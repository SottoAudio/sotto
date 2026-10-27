"""Sotto's processing in the bench app: echo canceller, residual suppressor, speech model and a
plugin insert per talker, run on its own thread so the audio callback never waits for it.

Per talker, every 10 ms block:
  mic -> high-pass -> echo canceller (PGM reference, bulk-delayed) -> residual suppressor
      -> speech model (all talkers in one batched call) -> attenuation limit
      -> mode select (Bypass / Echo removal / Echo + noise / Full clean, 10 ms crossfade)
      -> insert plugin (always running; bypass path delayed by its latency) -> trim -> out
Every version is computed every block, so switching modes or the insert is instant and clean.

Timing: the callback hands block k to the worker and takes back block k-1, so the processing
adds one block (10 ms) to the 20 ms the chain itself needs. If a result isn't ready the callback
sends the dry signal at the same delay and counts a miss; three misses in a row after start-up
make the engine unhealthy, which disarms Sotto (latched) and the frame goes to dry.

Per-talker faults (invalid output, a diverging echo canceller, a failing plugin) are reported to
the app, which trips that talker to dry until it is re-armed.
"""
import base64
import collections
import os
import queue
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "prototype"))

from cct.aec import PBFDAF                                  # noqa: E402
from cct.engine import Ring, DelayTracker                    # noqa: E402
from cct.stream_dfn import StreamSTFT, BatchedDFN            # noqa: E402
from scipy.signal import butter, sosfilt                     # noqa: E402

from . import macos                                          # noqa: E402
from .diagnostics import LOG                                 # noqa: E402

FS = 48000
HOP = 480
NFFT = 960
MODES = ("bypass", "aec", "aecns", "full")
XFADE = np.linspace(0.0, 1.0, HOP, endpoint=False) + 1.0 / HOP


def model_dir():
    """The batched speech model: bundled with the app, or built by prototype/tools/get_models.py."""
    env = os.environ.get("SOTTO_MODEL_DIR")
    cands = [env] if env else []
    if getattr(sys, "frozen", False):
        cands.append(os.path.join(getattr(sys, "_MEIPASS", ""), "models", "DeepFilterNet3_ll_stream_batch"))
    cands.append(os.path.join(os.path.dirname(os.path.dirname(HERE)), "prototype", "models", "DeepFilterNet3_ll_stream_batch"))
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "enc.onnx")):
            return c
    return None


def _db(p):
    return 10.0 * np.log10(p + 1e-12)


class Settings:
    """What one talker's chain should do. Built from the project (operator profile, channel and
    project defaults) by the app."""

    def __init__(self, mode="full", supp_db=12.0, atten_db=18.0, hpf_hz=100.0, trim_db=0.0, tail_ms=256,
                 insert=None):
        self.mode = mode if mode in MODES else "full"
        self.supp_db = float(supp_db)
        self.atten_db = float(atten_db)
        self.hpf_hz = float(hpf_hz)
        self.trim_db = float(trim_db)
        self.tail_ms = int(tail_ms)
        self.insert = insert or {}          # {"path": ..., "class": "", "on": bool, "state": base64}


class TalkerChain:
    """Echo canceller and suppressor for one talker. Attribute names match what cct's
    DelayTracker expects (mic_ring, delay_mode, set_delay, ...)."""

    def __init__(self, name, settings):
        self.name = name
        self.s = settings
        self.mic_ring = Ring(1, 10 * FS)
        self.st_e, self.st_y = StreamSTFT(NFFT, HOP), StreamSTFT(NFFT, HOP)
        self.syn_ns, self.syn_full = StreamSTFT(NFFT, HOP), StreamSTFT(NFFT, HOP)
        self.G = np.ones(NFFT // 2 + 1)
        self.dry_delay = np.zeros(NFFT - HOP)
        self.aec_delay = np.zeros(NFFT - HOP)
        self.w = np.zeros(len(MODES))
        self.w[MODES.index(settings.mode)] = 1.0
        self.delay_samples = None
        self.delay_mode = "auto"
        self.delay_status = "waiting for PGM"
        self.delay_measured_ms = None
        self.delay_conf = None
        self.bulk = 0
        self._pending_bulk = None
        self.diverge_blocks = 0
        self.erle = 0.0
        self._pd = self._pe = 1e-12
        self._hpf_hz = None
        self._sos = self._zi = None
        self.tail_ms = None
        self.aec = None
        self.apply(settings)
        # insert
        self.plugin = None
        self.plug_delay = np.zeros(0, np.float32)
        self.ins_w = 0.0
        self.insert_error = None
        self.level_in = self.level_out = -120.0

    def apply(self, s):
        self.s = s
        if s.hpf_hz != self._hpf_hz:
            self._hpf_hz = s.hpf_hz
            if s.hpf_hz > 0:
                self._sos = butter(2, s.hpf_hz, btype="highpass", fs=FS, output="sos")
                self._zi = np.zeros((self._sos.shape[0], 2))
            else:
                self._sos = None
        if s.tail_ms != self.tail_ms:
            self.tail_ms = s.tail_ms
            self.reset_aec()

    def reset_aec(self):
        self.aec = PBFDAF(1, FS, block=HOP, tail_s=self.tail_ms / 1000.0)
        self.diverge_blocks = 0

    def set_delay(self, samples, source="auto"):
        self.delay_samples = int(samples)
        self._pending_bulk = max(0, (self.delay_samples - int(0.06 * FS)) // HOP * HOP)
        if source == "manual":
            self.delay_mode = "manual"

    def _apply_pending_delay(self, refs_ring):
        new, self._pending_bulk = self._pending_bulk, None
        if new == self.bulk:
            return
        k = (new - self.bulk) // HOP
        a = self.aec
        if abs(k) >= a.P or refs_ring.w < (new + (a.P + 1) * HOP):
            self.bulk = new
            self.reset_aec()
            return
        wo = np.zeros_like(a.W)
        if k > 0:
            wo[:, :a.P - k] = a.W[:, k:]
        else:
            wo[:, -k:] = a.W[:, :a.P + k]
        a.W = wo
        for p in range(a.P):
            frame = refs_ring.read(new + HOP + p * HOP, 2 * HOP)
            a.X[:, (a.head - p) % a.P, :] = np.fft.rfft(frame, axis=1)
        a.xbuf = refs_ring.read(new + HOP, 2 * HOP)
        self.bulk = new

    def front(self, mic, refs_ring):
        """Stage 1: high-pass, echo canceller, suppressor. Returns the spectrum for the speech
        model and the versions that don't need it. Also returns a fault string or None."""
        fault = None
        if self._pending_bulk is not None:
            self._apply_pending_delay(refs_ring)
        if self._sos is not None:
            d, self._zi = sosfilt(self._sos, mic, zi=self._zi)
        else:
            d = mic.astype(np.float64)
        self.mic_ring.write(d[None])
        x = refs_ring.read(self.bulk, HOP)
        gated = float(np.max(np.abs(mic))) < 1e-5       # panel mics: digital silence between calls
        e, y = self.aec.process_block(x, d, adapt=not gated)
        if gated:
            e = d.copy()
        pd = float(np.mean(d * d)) + 1e-12
        pe = float(np.mean(e * e))
        if not gated:
            if not np.isfinite(pe):
                self.reset_aec()
                fault = "echo canceller produced invalid audio"
                e, y, pe = d.copy(), np.zeros_like(d), pd
            elif pe > 4.0 * pd and pd > 1e-8:
                self.diverge_blocks += 1
                if self.diverge_blocks > 50:
                    self.reset_aec()
                    fault = "echo canceller diverged (output louder than the mic for 0.5 s)"
            else:
                self.diverge_blocks = 0
            a = 0.995
            self._pd = a * self._pd + (1 - a) * pd
            self._pe = a * self._pe + (1 - a) * min(pe, 4 * pd)
            self.erle = _db(self._pd) - _db(self._pe)
        E = self.st_e.analyse(e)
        Y = self.st_y.analyse(y)
        if self.s.supp_db > 0:
            floor2 = 10 ** (-self.s.supp_db / 10)
            g = np.sqrt(np.maximum(1.0 - 1.5 * np.abs(Y) ** 2 / (np.abs(E) ** 2 + 1e-12), floor2))
            self.G = 0.5 * self.G + 0.5 * g
        else:
            self.G[:] = 1.0
        Z = E * self.G
        aec_only = np.concatenate([self.aec_delay, e])
        self.aec_delay, aec_only = aec_only[HOP:], aec_only[:HOP]
        dry = np.concatenate([self.dry_delay, mic])
        self.dry_delay, dry = dry[HOP:], dry[:HOP]
        return Z, {"bypass": dry, "aec": aec_only, "aecns": self.syn_ns.synthesise(Z)}, fault

    def back(self, versions, Zclean, Z):
        """Stage 2: the speech model's output, the mode crossfade, the insert and the trim."""
        if Zclean is not None:
            lim = 10 ** (-self.s.atten_db / 20)
            versions["full"] = self.syn_full.synthesise((1 - lim) * Zclean + lim * Z)
        else:
            versions["full"] = versions["aecns"]
        target = np.zeros(len(MODES))
        mode = self.s.mode
        if not np.all(np.isfinite(versions[mode])):
            mode = "bypass"
        target[MODES.index(mode)] = 1.0
        w = self.w[:, None] + (target - self.w)[:, None] * XFADE[None, :]
        self.w = target
        y = sum(w[i] * np.nan_to_num(versions[k]) for i, k in enumerate(MODES))
        fault = None
        ins_on = bool(self.plugin is not None and self.s.insert.get("on"))
        if self.plugin is not None:
            n = self.plugin.latency
            yp = self.plugin.process(y.astype(np.float32)).astype(np.float64)
            if not np.all(np.isfinite(yp)) or self.plugin.failures:
                self.plugin.failures = 0
                fault = f"insert {self.insert_name} stopped returning valid audio"
                yp = np.zeros_like(y)
                ins_on = False
            if n > 0:
                buf = np.concatenate([self.plug_delay, y])
                self.plug_delay, yb = buf[HOP:], buf[:HOP]
            else:
                yb = y
            t = 1.0 if ins_on else 0.0
            g = self.ins_w + (t - self.ins_w) * XFADE
            self.ins_w = t
            y = g * yp + (1 - g) * yb
        y = y * 10 ** (self.s.trim_db / 20)
        return y, versions["bypass"], fault

    def set_plugin(self, plugin, name):
        self.plugin = plugin
        self.insert_name = name
        self.plug_delay = np.zeros(plugin.latency if plugin else 0)
        self.ins_w = 1.0 if plugin is not None and self.s.insert.get("on") else 0.0


class Processor:
    """All talkers. exchange() is called from the audio callback; everything heavy runs on the
    worker thread."""

    def __init__(self, names, settings, model=None, vst3_lib=None, track_delay=True, threads=1):
        self.names = list(names)
        self.n = len(self.names)
        self.chains = [TalkerChain(nm, s) for nm, s in zip(self.names, settings)]
        self.talkers = self.chains                      # for cct.DelayTracker
        self.refs_ring = Ring(1, 12 * FS)
        self.model_path = model
        self.dfn = BatchedDFN(model, self.n, threads=threads) if model else None
        self._dfn_rows = [False] * self.n
        self.lib = vst3_lib
        self.q_in = queue.Queue(maxsize=2)
        self.q_out = queue.Queue(maxsize=4)
        self.history = collections.deque(maxlen=3)      # inputs k-2..k, for the miss fallback
        self.blocks = 0
        self.misses = 0
        self.miss_run = 0
        self.max_miss_run = 0
        self.good_run = 0
        self.proc_ms = 0.0
        self.proc_ms_max = 0.0
        self.stage_ms = {"front": 0.0, "model": 0.0, "back": 0.0}   # where the time goes
        self.errors = 0
        self.cpu_ms = 0.0                                # thread CPU time: if far below proc_ms, the
        self.thread_note = ""                            # thread is waiting, not computing
        self.faults = []                                 # (talker, why), drained by the app
        self.ok = True
        self.why = ""
        self._pending = None
        self._lock = threading.Lock()
        self._to_close = []
        self.stop_ev = threading.Event()
        self.th = threading.Thread(target=self._loop, name="sotto-processing", daemon=True)
        self.th.start()
        self.tracker = DelayTracker(self) if track_delay else None
        self.ref_db = -120.0

    # ------------------------------------------------------------ audio callback side
    def exchange(self, mics, ref):
        """mics: (n, HOP) float, ref: (HOP,) float. Returns (outputs, dry), each (n, HOP), for the
        block handed over one callback ago, and aligned with it."""
        k = self.blocks
        self.blocks += 1
        self.history.append(mics.copy())
        try:
            self.q_in.put_nowait((k, mics.copy(), np.asarray(ref, dtype=np.float64).copy()))
        except queue.Full:
            pass
        # Take the result for block k-1; anything older is stale and dropped, so the delay through
        # the processing never grows.
        while True:
            try:
                tag, out, dry = self.q_out.get_nowait()
            except queue.Empty:
                break
            if tag == k - 1:
                self.miss_run = 0
                self.good_run += 1
                if not self.ok and self.good_run >= 100:   # 1 s on time: the check can go green again
                    self.ok, self.why = True, ""
                return out, dry
        self.good_run = 0
        if self.blocks > 50:                             # ignore start-up
            self.misses += 1
            self.miss_run += 1
            self.max_miss_run = max(self.max_miss_run, self.miss_run)
            if self.miss_run >= 3 and self.ok:
                self.ok, self.why = False, "processing can't keep up (3 blocks late in a row)"
        # Dry at the same delay as the processed path: input k-2 (one block queue + one block STFT).
        fallback = self.history[0] if len(self.history) == 3 else np.zeros_like(mics)
        return fallback, fallback

    def health(self):
        return self.ok, self.why

    def clear_health(self):
        """Re-arming is still manual: this only lets the checks go green again once processing is
        keeping up."""
        if not self.ok and self.miss_run == 0:
            self.ok, self.why = True, ""

    def drain_faults(self):
        f, self.faults = self.faults, []
        return f

    # ------------------------------------------------------------ settings (any thread)
    def configure(self, settings, plugins):
        """settings: list of Settings; plugins: list per talker of (Plugin or None, name, changed).
        Applied by the worker at the start of its next block."""
        with self._lock:
            self._pending = (settings, plugins)

    # ------------------------------------------------------------ worker
    def _apply_pending(self):
        with self._lock:
            p, self._pending = self._pending, None
        if not p:
            return
        settings, plugins = p
        for c, s, pl in zip(self.chains, settings, plugins):
            c.apply(s)
            plug, name, changed = pl
            if changed:
                if c.plugin is not None:
                    self._to_close.append(c.plugin)
                c.set_plugin(plug, name)

    def _loop(self):
        _, self.thread_note = macos.fast_thread()
        LOG.info("Processing thread started for %d talkers (%s)", self.n, self.thread_note)
        while not self.stop_ev.is_set():
            try:
                self._one_block()
            except Exception as e:
                # Never let the thread die silently: no output means the callback counts misses,
                # the engine goes unhealthy and Sotto disarms (the frame goes to dry).
                self.errors += 1
                if self.errors == 1 or self.errors % 100 == 0:
                    LOG.exception("Processing error (%d so far)", self.errors)
                    for i in range(self.n):
                        self.faults.append((i, f"processing error: {e!r}"))

    def _one_block(self):
        try:
            tag, mics, ref = self.q_in.get(timeout=0.2)
            while True:                              # running late: skip to the newest block
                try:
                    tag, mics, ref = self.q_in.get_nowait()
                except queue.Empty:
                    break
        except queue.Empty:
            return
        t0, c0 = time.perf_counter(), time.thread_time()
        self._apply_pending()
        self.refs_ring.write(ref[None])
        self.ref_db = _db(float(np.mean(ref * ref)))
        fronts = []
        t_front = time.perf_counter()
        for i, c in enumerate(self.chains):
            Z, versions, fault = c.front(mics[i].astype(np.float64), self.refs_ring)
            c.level_in = _db(float(np.mean(mics[i] ** 2)))
            fronts.append((Z, versions))
            if fault:
                self.faults.append((i, fault))
        t_model = time.perf_counter()
        Zc = None
        # The speech model is the expensive part: run it only while someone is on Full clean
        # (or crossfading to or from it). Rows that rejoin start from a clean model state.
        need = [c.s.mode == "full" or c.w[MODES.index("full")] > 0 for c in self.chains]
        if self.dfn is not None and any(need):
            for i, n_ in enumerate(need):
                if n_ and not self._dfn_rows[i]:
                    self.dfn.reset_row(i)
                self._dfn_rows[i] = n_
            Zs = np.stack([f[0] for f in fronts])
            try:
                Zc = self.dfn.process_specs(Zs, rows=[i for i, n_ in enumerate(need) if n_])
            except Exception as e:                   # the model failing is a fault for everyone
                LOG.exception("Speech model error")
                Zc = None
                for i in range(self.n):
                    self.faults.append((i, f"speech model error: {e!r}"))
        t_back = time.perf_counter()
        outs, drys = np.zeros((self.n, HOP)), np.zeros((self.n, HOP))
        for i, c in enumerate(self.chains):
            y, dry, fault = c.back(fronts[i][1], None if Zc is None else Zc[i], fronts[i][0])
            if fault:
                self.faults.append((i, fault))
            outs[i], drys[i] = y, dry
            c.level_out = _db(float(np.mean(y ** 2)))
        try:
            self.q_out.put_nowait((tag, outs, drys))
        except queue.Full:
            pass
        t_end = time.perf_counter()
        dt = 1000 * (t_end - t0)
        for k, a_, b_ in (("front", t_front, t_model), ("model", t_model, t_back), ("back", t_back, t_end)):
            self.stage_ms[k] = 0.98 * self.stage_ms[k] + 0.02 * 1000 * (b_ - a_)
        self.cpu_ms = 0.98 * self.cpu_ms + 0.02 * 1000 * (time.thread_time() - c0)
        self.proc_ms = 0.98 * self.proc_ms + 0.02 * dt
        self.proc_ms_max = max(self.proc_ms_max * 0.999, dt)
        while self._to_close:
            try:
                self._to_close.pop().close()
            except Exception:
                pass

    def stop(self):
        self.stop_ev.set()
        self.th.join(timeout=2)
        if self.tracker:
            self.tracker.stop()
        for c in self.chains:
            if c.plugin is not None:
                try:
                    c.plugin.close()
                except Exception:
                    pass

    def state(self):
        return {
            "on": True, "model": self.dfn is not None, "proc_ms": round(self.proc_ms, 2),
            "proc_ms_max": round(self.proc_ms_max, 2), "cpu_ms": round(self.cpu_ms, 2),
            "thread": self.thread_note, "stage_ms": {k: round(v, 2) for k, v in self.stage_ms.items()},
            "model_rows": sum(self._dfn_rows), "misses": self.misses, "ok": self.ok, "why": self.why,
            "latency_ms": round(1000 * (2 * HOP) / FS, 1), "pgm_db": round(self.ref_db, 1),
            "talkers": [{
                "name": c.name, "mode": c.s.mode, "erle_db": round(c.erle, 1),
                "speech_snr_db": None if self.dfn is None else round(float(self.dfn.lsnr[i]), 1),
                "delay_ms": None if c.delay_samples is None else round(1000 * c.delay_samples / FS, 1),
                "delay_status": c.delay_status,
                "insert": None if c.plugin is None else {
                    "name": c.insert_name, "on": bool(c.s.insert.get("on")), "latency": c.plugin.latency},
                "insert_error": c.insert_error,
            } for i, c in enumerate(self.chains)],
        }


def settings_from_project(project, names):
    """Turns the console's project (channels, operator profiles, processing defaults) into one
    Settings per talker. Returns (settings, display names)."""
    proc = (project or {}).get("proc", {})
    chs = (project or {}).get("ch", [])
    ops = {o.get("id"): o for o in (project or {}).get("operators", [])}
    out, labels = [], []
    for i, nm in enumerate(names):
        ch = chs[i] if i < len(chs) else {}
        op = ops.get(ch.get("op")) if ch else None
        mode = (op or ch or {}).get("mode", "default")
        if mode == "default":
            mode = proc.get("mode", "full")
        ins = {}
        if op and op.get("ins", {}).get("plugin"):
            path = op["ins"]["plugin"]
            ins = {"path": path, "on": bool(op["ins"].get("on")),
                   "state": (op["ins"].get("vst3") or {}).get(path, "")}
        out.append(Settings(mode=mode, supp_db=proc.get("supp", 12), atten_db=proc.get("atten", 18),
                            hpf_hz=proc.get("hpf", 100), tail_ms=proc.get("tail", 256),
                            trim_db=float(proc.get("trim", 0)) + float((op or {}).get("trim", 0) or 0), insert=ins))
        labels.append((op or {}).get("name") or ch.get("name") or nm)
    return out, labels


def b64(data):
    return base64.b64encode(data).decode() if data else ""


def unb64(s):
    try:
        return base64.b64decode(s) if s else b""
    except ValueError:
        return b""
