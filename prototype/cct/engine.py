"""Real-time comms voice cleaner engine: 10 ms blocks, one or more talkers, shared PA references.

Per talker, every block:
  mic -> 80 Hz high-pass -> echo canceller (PA references, bulk-delayed) -> residual echo
  suppressor -> DeepFilterNet3 LL -> output
All three versions (dry, echo-only, full clean) are computed every block so switching between
them is a 10 ms crossfade with no restart. Dry is delayed to line up with the processed path.

Processing delay (algorithmic): 20 ms for echo-only and full clean (10 ms block + 10 ms STFT
overlap); add the audio interface's own input+output latency on top.
"""
import multiprocessing as mp
import queue
import threading
import time

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from .aec import PBFDAF
from .stream_dfn import StreamingDFN, StreamSTFT

FS = 48000
HOP = 480
NFFT = 960
MODES = ("dry", "echo", "clean")


def _db(p):
    return 10.0 * np.log10(p + 1e-12)


class HPF:
    def __init__(self, n_ch, fc=80.0, fs=FS):
        self.sos = butter(2, fc, btype="highpass", fs=fs, output="sos")
        z = sosfilt_zi(self.sos)
        self.zi = np.repeat(z[:, None, :], n_ch, axis=1) * 0.0      # (sections, ch, 2)

    def __call__(self, x):
        """x: (ch, n)."""
        y, self.zi = sosfilt(self.sos, x, axis=1, zi=self.zi)
        return y


class Ring:
    """Multichannel history buffer. write() appends a block; read(delay) returns the block that
    was written `delay` samples before the most recent one; last(n) returns the newest n samples."""

    def __init__(self, n_ch, length):
        self.buf = np.zeros((n_ch, length))
        self.L = length
        self.w = 0                      # total samples written

    def write(self, x):
        n = x.shape[1]
        i = self.w % self.L
        end = i + n
        if end <= self.L:
            self.buf[:, i:end] = x
        else:
            k = self.L - i
            self.buf[:, i:] = x[:, :k]
            self.buf[:, :end - self.L] = x[:, k:]
        self.w += n

    def _get(self, start, n):
        idx = (np.arange(start, start + n)) % self.L
        return self.buf[:, idx]

    def read(self, delay, n):
        return self._get(self.w - n - delay, n)

    def last(self, n):
        n = min(n, self.w, self.L)
        return self._get(self.w - n, n)


# --------------------------------------------------------------------------- delay finding
def _delay_worker(q_in, q_out):
    """Separate process so long FFTs never hold up the audio thread."""
    import signal
    from .delay import track_delay
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)       # the main app handles Ctrl+C
    except ValueError:
        pass                                                # not the main thread (inline mode)
    while True:
        job = q_in.get()
        if job is None:
            return
        tid, mic, refs = job
        res = []
        for r in refs:
            est = track_delay(mic, r, FS, win_s=3.0, hop_s=1.0, min_peak=0.01)
            if est:
                lags = np.array([e[1] for e in est])
                peaks = np.array([e[2] for e in est])
                good = lags[peaks >= np.median(peaks)]
                res.append((int(np.median(good)), float(np.median(peaks)), len(est),
                            float(np.std(lags) / FS * 1000)))
        q_out.put((tid, res))


class DelayTracker:
    """Measures the PA flight time per talker every few seconds, in a worker process.
    A new delay is applied once three measurements in a row agree within 20 ms and it differs
    from the current one by more than 20 ms. A manual delay disables tracking for that talker."""

    def __init__(self, engine, every_s=2.0, window_s=6.0, inline=False):
        """inline=True: no worker process or timer; call step() yourself (offline simulation)."""
        self.engine = engine
        self.every = every_s
        self.win = int(window_s * FS)
        self.hist = {}
        self.inline = inline
        if inline:
            return
        ctx = mp.get_context("spawn")
        self.q_in, self.q_out = ctx.Queue(maxsize=4), ctx.Queue()
        self.proc = ctx.Process(target=_delay_worker, args=(self.q_in, self.q_out), daemon=True)
        self.proc.start()
        self.stop_ev = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _jobs(self):
        eng = self.engine
        if eng.refs_ring.w < self.win:
            return []
        refs = eng.refs_ring.last(self.win)
        if np.max(np.abs(refs)) < 1e-4:              # no PA programme, nothing to measure
            return []
        return [(tid, t.mic_ring.last(self.win)[0], refs)
                for tid, t in enumerate(eng.talkers) if t.delay_mode != "manual"]

    def step(self):
        """Inline measurement (simulation)."""
        import io
        import contextlib
        for job in self._jobs():
            q_in, q_out = queue.Queue(), queue.Queue()
            q_in.put(job)
            q_in.put(None)
            with contextlib.redirect_stdout(io.StringIO()):
                _delay_worker(q_in, q_out)
            self._apply(*q_out.get())

    def _loop(self):
        while not self.stop_ev.wait(self.every):
            sent = 0
            for job in self._jobs():
                try:
                    self.q_in.put_nowait(job)
                    sent += 1
                except queue.Full:
                    pass
            for _ in range(sent):
                try:
                    tid, res = self.q_out.get(timeout=10)
                except queue.Empty:
                    break
                self._apply(tid, res)

    def _apply(self, tid, res):
        t = self.engine.talkers[tid]
        if not res:
            t.delay_status = "no PA in mic yet"
            return
        lag = min(r[0] for r in res)
        conf = max(r[1] for r in res)
        t.delay_measured_ms = 1000.0 * lag / FS
        t.delay_conf = conf
        h = self.hist.setdefault(tid, [])
        h.append(lag)
        del h[:-3]
        # The canceller starts 60 ms before the delay and spans 400 ms after it, so a few ms
        # either way is harmless. Lock once three measurements in a row sit within 20 ms of each
        # other, using the earliest (the direct sound; later peaks are reflections).
        tol = int(0.020 * FS)
        if len(h) < 3 or (max(h) - min(h)) > tol:
            t.delay_status = "measuring"
            return
        new = min(h)
        if t.delay_samples is None or abs(new - t.delay_samples) > tol:
            t.set_delay(new, source="auto")
        t.delay_status = "locked"

    def stop(self):
        if self.inline:
            return
        self.stop_ev.set()
        try:
            self.q_in.put_nowait(None)
        except Exception:
            pass
        self.proc.join(timeout=1)
        if self.proc.is_alive():
            self.proc.terminate()


# --------------------------------------------------------------------------- one talker
class Talker:
    def __init__(self, name, n_ref, dfn_proto=None, model_dir=None, tail_s=0.4, margin_s=0.06,
                 res_beta=1.5, res_floor_db=-18.0, res_smooth=0.5, delay_ms=None):
        self.name = name
        self.n_ref = n_ref
        self.tail_s = tail_s
        self.margin = int(margin_s * FS)
        self.hpf = HPF(1)
        self.aec = PBFDAF(n_ref, FS, block=HOP, tail_s=tail_s)
        self.st_e, self.st_y = StreamSTFT(NFFT, HOP), StreamSTFT(NFFT, HOP)
        self.syn_echo, self.syn_clean = StreamSTFT(NFFT, HOP), StreamSTFT(NFFT, HOP)
        self.dfn = None
        if model_dir is not None:
            self.dfn = StreamingDFN(model_dir, share_from=dfn_proto)
        self.res_beta = res_beta
        self.res_floor2 = 10 ** (res_floor_db / 10)
        self.res_smooth = res_smooth
        self.G = np.ones(NFFT // 2 + 1)
        self.dry_delay = np.zeros(NFFT - HOP)             # align dry with the STFT path
        self.mic_ring = Ring(1, 10 * FS)                  # high-passed mic, for delay finding
        self.mode = "clean" if self.dfn is not None else "echo"
        self.w = np.array([0.0, 0.0, 0.0])
        self.w[MODES.index(self.mode)] = 1.0
        # delay
        self.delay_samples = None
        self.delay_mode = "auto"
        self.delay_status = "waiting for PA"
        self.delay_measured_ms = None
        self.delay_conf = None
        self.bulk = 0
        self._pending_bulk = None
        self._reset_req = False
        if delay_ms is not None:
            self.set_delay(int(delay_ms * FS / 1000), source="manual")
        # health and meters
        self.diverge_blocks = 0
        self.gated_blocks = 0
        self.resets = 0
        self.fault = None
        self.fault_until = 0
        self.m = dict(mic=-120.0, out=-120.0, erle=0.0, lsnr=0.0)
        self._pd = self._pe = 1e-12

    # -- delay
    def set_delay(self, samples, source="auto"):
        """Request a new PA delay. Safe from any thread: applied by the audio thread at the start
        of the next block (see _apply_pending_delay)."""
        self.delay_samples = int(samples)
        self._pending_bulk = max(0, (self.delay_samples - self.margin) // HOP * HOP)
        if source == "manual":
            self.delay_mode = "manual"
            self.delay_status = "manual"

    def _apply_pending_delay(self, refs_ring):
        new = self._pending_bulk
        self._pending_bulk = None
        if new == self.bulk:
            return
        k = (new - self.bulk) // HOP
        a = self.aec
        if abs(k) >= a.P or refs_ring.w < (new + (a.P + 1) * HOP):
            self.bulk = new
            self.reset_aec()
            return
        # Keep what the canceller has learnt: the echo path moves k partitions earlier/later in
        # the filter, and the reference history is rebuilt for the new bulk delay.
        Wo = np.zeros_like(a.W)                    # W[:, p] holds lag p (partitions)
        if k > 0:
            Wo[:, :a.P - k] = a.W[:, k:]
        else:
            Wo[:, -k:] = a.W[:, :a.P + k]
        a.W = Wo
        # Called after this block's reference was written, before the canceller runs, so the
        # canceller's history ends one block back.
        for p in range(a.P):
            frame = refs_ring.read(new + HOP + p * HOP, 2 * HOP)
            a.X[:, (a.head - p) % a.P, :] = np.fft.rfft(frame, axis=1)
        a.xbuf = refs_ring.read(new + HOP, 2 * HOP)
        self.bulk = new

    def clear_manual_delay(self):
        self.delay_mode = "auto"
        self.delay_status = "measuring"

    def reset_aec(self):
        self.aec = PBFDAF(self.n_ref, FS, block=HOP, tail_s=self.tail_s)
        self.diverge_blocks = 0

    # -- processing
    def process(self, mic, refs_ring, now_block):
        """mic: (HOP,) raw. Returns dict of the three output versions, each (HOP,)."""
        if self._reset_req:
            self._reset_req = False
            self.reset_aec()
        if self._pending_bulk is not None:
            self._apply_pending_delay(refs_ring)
        d = self.hpf(mic[None])[0]
        self.mic_ring.write(d[None])
        x = refs_ring.read(self.bulk, HOP)
        # Panel and beltpack mics are gated by the talk keys: digital silence between calls.
        # Don't learn from silence (it would unlearn the PA path) and pass the silence through.
        gated = float(np.max(np.abs(mic))) < 1e-5
        e, y = self.aec.process_block(x, d, adapt=not gated)
        if gated:
            e = d.copy()
            self.gated_blocks += 1

        # Health: a diverging canceller makes the output louder than its input.
        pd = float(np.mean(d * d)) + 1e-12
        pe = float(np.mean(e * e))
        if gated:
            pass
        elif not np.isfinite(pe):
            self.reset_aec()
            self.resets += 1
            self.fault, self.fault_until = "echo canceller reset (invalid output)", now_block + 200
            e = d.copy()
            y = np.zeros_like(d)
            pe = pd
        elif pe > 4.0 * pd and pd > 1e-8:
            self.diverge_blocks += 1
            if self.diverge_blocks > 50:              # 0.5 s louder than the mic: reset
                self.reset_aec()
                self.resets += 1
                self.fault, self.fault_until = "echo canceller reset (diverged)", now_block + 200
        else:
            self.diverge_blocks = 0
        if self.fault and now_block > self.fault_until:
            self.fault = None

        E = self.st_e.analyse(e)
        Y = self.st_y.analyse(y)
        See, Syy = np.abs(E) ** 2, np.abs(Y) ** 2
        g = np.sqrt(np.maximum(1.0 - self.res_beta * Syy / (See + 1e-12), self.res_floor2))
        self.G = self.res_smooth * self.G + (1 - self.res_smooth) * g
        Z = E * self.G
        echo_only = self.syn_echo.synthesise(Z)
        if self.dfn is not None:
            clean = self.syn_clean.synthesise(self.dfn.process_spec(Z))
            self.m["lsnr"] = self.dfn.lsnr
        else:
            clean = echo_only
        dry = np.concatenate([self.dry_delay, mic])
        self.dry_delay = dry[HOP:]
        dry = dry[:HOP]

        a = 0.995
        if not gated:
            self._pd = a * self._pd + (1 - a) * pd
            self._pe = a * self._pe + (1 - a) * min(pe, 4 * pd)
        self.m["erle"] = _db(self._pd) - _db(self._pe)
        return {"dry": dry, "echo": echo_only, "clean": clean}

    def select(self, outs):
        """Crossfade (one block) towards the selected mode; any invalid output -> dry."""
        target = np.zeros(3)
        mode = self.mode
        if mode != "dry" and not (np.all(np.isfinite(outs[mode]))):
            mode = "dry"
        target[MODES.index(mode)] = 1.0
        ramp = np.linspace(0.0, 1.0, HOP, endpoint=False) + 1.0 / HOP
        w = self.w[:, None] + (target - self.w)[:, None] * ramp[None, :]
        y = sum(w[i] * np.nan_to_num(outs[k]) for i, k in enumerate(MODES))
        self.w = target
        self.m["mic"] = _db(float(np.mean(outs["dry"] ** 2)))
        self.m["out"] = _db(float(np.mean(y ** 2)))
        return y


# --------------------------------------------------------------------------- whole engine
class Engine:
    def __init__(self, talker_names, n_ref, model_dir=None, tail_s=0.4, delay_ms=None,
                 track_delay="process"):
        """track_delay: "process" (live, worker process), "inline" (simulation; call
        self.tracker.step()), or None."""
        self.n_ref = n_ref
        self.refs_hpf = HPF(n_ref)
        self.refs_ring = Ring(n_ref, 12 * FS)
        proto = None
        self.talkers = []
        for name in talker_names:
            t = Talker(name, n_ref, dfn_proto=proto, model_dir=model_dir, tail_s=tail_s,
                       delay_ms=delay_ms)
            if t.dfn is not None and proto is None:
                proto = t.dfn
            self.talkers.append(t)
        self.block = 0
        self.proc_ms = 0.0
        self.proc_ms_max = 0.0
        self.ref_db = -120.0
        self.tracker = DelayTracker(self, inline=(track_delay == "inline")) if track_delay else None

    def process(self, mics, refs):
        """mics: (n_talkers, HOP), refs: (n_ref, HOP). Returns (selected outputs (n_talkers, HOP),
        all versions per talker)."""
        t0 = time.perf_counter()
        r = self.refs_hpf(refs)
        self.refs_ring.write(r)
        outs, allv = [], []
        for i, t in enumerate(self.talkers):
            v = t.process(mics[i], self.refs_ring, self.block)
            outs.append(t.select(v))
            allv.append(v)
        self.block += 1
        self.ref_db = _db(float(np.mean(refs ** 2)))
        dt = 1000 * (time.perf_counter() - t0)
        self.proc_ms = 0.98 * self.proc_ms + 0.02 * dt
        self.proc_ms_max = max(self.proc_ms_max * 0.999, dt)
        return np.stack(outs), allv

    def stop(self):
        if self.tracker:
            self.tracker.stop()

    def state(self):
        return {
            "proc_ms": round(self.proc_ms, 2), "proc_ms_max": round(self.proc_ms_max, 2),
            "ref_db": round(self.ref_db, 1),
            "talkers": [{
                "name": t.name, "mode": t.mode, "mic_db": round(t.m["mic"], 1),
                "out_db": round(t.m["out"], 1), "erle_db": round(t.m["erle"], 1),
                "speech_snr_db": round(t.m["lsnr"], 1),
                "delay_ms": None if t.delay_samples is None else round(1000 * t.delay_samples / FS, 1),
                "delay_m": None if t.delay_samples is None else round(t.delay_samples / FS * 343, 1),
                "delay_measured_ms": None if t.delay_measured_ms is None else round(t.delay_measured_ms, 1),
                "delay_status": t.delay_status, "delay_mode": t.delay_mode,
                "resets": t.resets, "fault": t.fault, "has_dfn": t.dfn is not None,
                "gated": t.gated_blocks > 0 and t.m["mic"] < -100,
            } for t in self.talkers],
        }
