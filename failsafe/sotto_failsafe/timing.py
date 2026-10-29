"""A detailed record of audio timing, for the diagnostics file.

Every soundcard cycle and every 10 ms block leaves one row in a fixed-size ring (the last ~40 s),
so the report can say exactly where audio was lost: cycles Core Audio skipped (seen as jumps in
the capture time it reports), callbacks that ran too long for the buffer, blocks whose processing
came back late, Python garbage collection pauses, and PortAudio's under/overflow flags.

Audio-thread safe: the rings are allocated once at start-up; a cycle or a block only writes a
handful of numbers into them. Reading (summary(), csv()) happens on other threads; a row being
written at that moment can be half old, half new, which is fine for a report.
"""
import gc
import threading
import time

import numpy as np

# PortAudio callback status bits
IN_UNDERFLOW, IN_OVERFLOW, OUT_UNDERFLOW, OUT_OVERFLOW, PRIMING = 1, 2, 4, 8, 16
STAGES = ("tests", "exchange", "core", "phones", "record")


class TimingTrace:
    OUTAGE_S = 1.0          # a capture-time jump longer than this is the soundcard stopping, not skipping
    def __init__(self, fs=48000, io_frames=480, block=480, seconds=40):
        self.fs, self.io, self.block = fs, int(io_frames), int(block)
        n_dev = int(seconds * fs / max(16, self.io)) + 1
        n_blk = int(seconds * fs / self.block) + 1
        # one row per soundcard cycle: wall time, capture time, frames, callback ms, status bits
        self.dev = np.full((n_dev, 6), np.nan)          # + margin: engine output waiting (native side)
        self.native_stats = None
        self.dev_i = 0
        # one row per block: wall time, engine ms, per-stage ms, workers late, queue level
        self.blk = np.full((n_blk, 4 + len(STAGES)), np.nan)
        self.blk_i = 0
        self.gc_rows = np.full((512, 3), np.nan)            # wall time, generation, pause ms
        self.gc_i = 0
        self._gc_t0 = None
        self.t_start = time.perf_counter()
        self._last_adc = None
        self.skipped = 0                                    # cycles Core Audio skipped since start
        self.skip_events = 0
        self.outages = 0                                    # the soundcard stopped (DVS restarted): not skips
        self.outage_s = 0.0
        self.started_wall = time.time()
        self.lock = threading.Lock()                        # summary()/csv() only, never the audio thread

    # ------------------------------------------------------------ audio thread
    def cycle(self, t, adc, frames, cb_ms, status_bits, margin=float("nan")):
        if adc == adc and adc > 0:                          # not NaN: count skipped cycles as they happen
            if self._last_adc is not None:
                d = (adc - self._last_adc) * self.fs / self.io
                if d * self.io / self.fs > self.OUTAGE_S:   # a stop and restart, counted on its own
                    self.outages += 1
                    self.outage_s += d * self.io / self.fs
                elif d > 1.5:
                    self.skipped += int(d + 0.5) - 1
                    self.skip_events += 1
            self._last_adc = adc
        r = self.dev[self.dev_i % len(self.dev)]
        r[0], r[1], r[2], r[3], r[4], r[5] = t, adc, frames, cb_ms, status_bits, margin
        self.dev_i += 1

    def block_done(self, t, engine_ms, stages, late_workers, queue):
        r = self.blk[self.blk_i % len(self.blk)]
        r[0], r[1], r[2], r[3] = t, engine_ms, late_workers, queue
        for k, v in enumerate(stages):
            r[4 + k] = v
        self.blk_i += 1

    # ------------------------------------------------------------ garbage collection
    def watch_gc(self):
        if self._gc_cb not in gc.callbacks:
            gc.callbacks.append(self._gc_cb)

    def stop(self):
        try:
            gc.callbacks.remove(self._gc_cb)
        except ValueError:
            pass

    def _gc_cb(self, phase, info):
        if phase == "start":
            self._gc_t0 = time.perf_counter()
        elif self._gc_t0 is not None:
            r = self.gc_rows[self.gc_i % len(self.gc_rows)]
            r[0], r[1], r[2] = self._gc_t0, info.get("generation", -1), 1000 * (time.perf_counter() - self._gc_t0)
            self.gc_i += 1
            self._gc_t0 = None

    # ------------------------------------------------------------ reading
    @staticmethod
    def _rows(ring, n):
        k = min(n, len(ring))
        if n <= len(ring):
            rows = ring[:k].copy()
        else:
            i = n % len(ring)
            rows = np.concatenate([ring[i:], ring[:i]])
        return rows[~np.isnan(rows[:, 0])]

    @staticmethod
    def _pct(x, unit=" ms"):
        x = x[~np.isnan(x)]
        if not len(x):
            return "no data"
        p = np.percentile(x, [50, 90, 99])
        return f"median {p[0]:.2f}{unit}, 90% {p[1]:.2f}{unit}, 99% {p[2]:.2f}{unit}, worst {x.max():.2f}{unit} (n={len(x)})"

    def summary(self):
        """Plain-text section for the diagnostics file."""
        with self.lock:
            dev = self._rows(self.dev, self.dev_i)
            blk = self._rows(self.blk, self.blk_i)
            gcr = self._rows(self.gc_rows, self.gc_i)
        out = []
        if len(dev) < 3:
            return "Not enough audio yet (the soundcard has run fewer than 3 cycles)."
        span = dev[-1, 0] - dev[0, 0]
        period_ms = 1000 * self.io / self.fs
        out.append(f"Covers the last {span:.1f} s: {len(dev)} soundcard cycles of {self.io} samples "
                   f"({period_ms:.2f} ms each), {len(blk)} blocks of {self.block} samples.")
        # what should have happened vs what did, from the wall clock
        expected = span * self.fs / self.io
        out.append(f"Soundcard cycles: expected about {expected:.0f}, ran {len(dev) - 1} "
                   f"({100 * (len(dev) - 1) / max(expected, 1):.1f} %).")
        # skipped cycles: jumps in the capture time Core Audio reports for each cycle
        adc = dev[:, 1]
        good = ~np.isnan(adc)
        if good.sum() > 2:
            d = np.diff(adc[good]) * self.fs / self.io          # in cycles; 1.0 = no gap
            gone = d * self.io / self.fs > self.OUTAGE_S
            if gone.any():
                out.append(f"The soundcard stopped {int(gone.sum())} time(s) in this stretch "
                           f"({np.sum(d[gone]) * self.io / self.fs:.1f} s without audio); not counted as skipped cycles.")
                d = d[~gone]
            skipped = np.clip(np.round(d) - 1, 0, None)
            lost = skipped.sum()
            events = int((skipped > 0).sum())
            out.append(f"Skipped cycles (capture time jumped): {int(lost)} cycles in {events} places = "
                       f"{1000 * lost * self.io / self.fs:.0f} ms of audio lost "
                       f"({100 * lost / max(len(d) + lost, 1):.2f} % of the time).")
            if events:
                big = np.argsort(d)[::-1][:5]
                out.append("  Biggest jumps: " + ", ".join(f"{int(np.round(d[i])) - 1} cycles "
                                                          f"({1000 * (d[i] - 1) * self.io / self.fs:.0f} ms)"
                                                          for i in big if d[i] > 1.5))
        else:
            out.append("Skipped cycles: the soundcard reports no capture times, so they can't be counted here.")
        gaps = np.diff(dev[:, 0]) * 1000
        out.append(f"Time between callbacks: {self._pct(gaps)}; expected {period_ms:.2f} ms.")
        cb = dev[:, 3]
        over = int((cb > period_ms).sum())
        out.append(f"Callback time (all of Sotto Voce's work in the soundcard's thread): {self._pct(cb)}. "
                   f"Over the {period_ms:.2f} ms budget: {over} of {len(cb)} ({100 * over / len(cb):.1f} %).")
        st = dev[:, 4].astype(int)
        flags = {name: int(((st & bit) > 0).sum()) for name, bit in
                 (("input underflow", IN_UNDERFLOW), ("input overflow", IN_OVERFLOW),
                  ("output underflow", OUT_UNDERFLOW), ("output overflow", OUT_OVERFLOW))}
        out.append("PortAudio status flags: " + ", ".join(f"{k} {v}" for k, v in flags.items()) + ".")
        m = dev[:, 5]
        if np.any(~np.isnan(m)):
            ms = m * 1000.0 / self.fs
            mm = ms[~np.isnan(ms)]
            low = np.percentile(mm, [1, 10, 50])
            out.append(f"Native engine margin (engine output ready ahead of each cycle; below 0 = played dry): "
                       f"median {low[2]:.2f} ms, lowest 10% {low[1]:.2f} ms, lowest 1% {low[0]:.2f} ms, "
                       f"lowest {mm.min():.2f} ms; {int((mm < 0).sum())} cycles below 0.")
        ns = self.native_stats
        if ns:
            out.append(f"Native soundcard side since start: {ns['cycles']} cycles, {ns['underrun_cycles']} with the "
                       f"engine late ({1000 * ns['fallback_frames'] / self.fs:.0f} ms played dry), "
                       f"{ns['dropped_blocks']} blocks skipped as too late, {ns['overruns']} input overruns, "
                       f"{ns['resyncs']} resyncs.")
        if len(blk):
            out.append(f"Engine time per 10 ms block: {self._pct(blk[:, 1])}.")
            for k, name in enumerate(STAGES):
                col = blk[:, 4 + k]
                if np.nanmax(col) > 0:
                    out.append(f"  {name}: {self._pct(col)}")
            late = blk[:, 2]
            n_late = int((late > 0).sum())
            out.append(f"Blocks where processing came back late (unprocessed audio played instead): "
                       f"{n_late} of {len(blk)} ({100 * n_late / len(blk):.1f} %).")
            q = blk[:, 3]
            if np.nanmax(q) > 0:
                out.append(f"Output queue (regrouping) before each block, samples: {self._pct(q, '')}.")
        if len(gcr):
            in_span = gcr[gcr[:, 0] >= dev[0, 0]]
            out.append(f"Python garbage collection: {len(in_span)} pauses, total {np.nansum(in_span[:, 2]):.1f} ms; "
                       f"{self._pct(in_span[:, 2])}.")
        else:
            out.append("Python garbage collection: no pauses recorded.")
        # the worst moments, with what else happened around them
        worst = np.argsort(gaps)[::-1][:8]
        lines = []
        for i in sorted(worst):
            t = dev[i + 1, 0]
            g = [r for r in gcr if abs(r[0] - t) < 0.2]
            b = blk[(blk[:, 0] > t - 0.05) & (blk[:, 0] <= t + 0.01)] if len(blk) else []
            note = []
            if g:
                note.append(f"garbage collection {max(r[2] for r in g):.1f} ms nearby")
            if len(b):
                note.append(f"block engine {np.nanmax(b[:, 1]):.1f} ms")
            lines.append(f"  at +{t - self.t_start:8.2f} s: {gaps[i]:.1f} ms without a callback; "
                         f"that callback took {cb[i + 1]:.1f} ms" + ("; " + ", ".join(note) if note else ""))
        out.append("Worst gaps between callbacks:\n" + "\n".join(lines))
        return "\n".join(out)

    def csv(self, seconds=10):
        """The last few seconds, one row per soundcard cycle and per block, for a closer look."""
        with self.lock:
            dev = self._rows(self.dev, self.dev_i)
            blk = self._rows(self.blk, self.blk_i)
        if not len(dev):
            return ""
        t_end = dev[-1, 0]
        rows = ["kind,t_s,capture_s,frames,callback_ms,status,margin_frames,engine_ms,late_workers,queue," + ",".join(STAGES)]
        for r in dev[dev[:, 0] >= t_end - seconds]:
            rows.append(f"cycle,{r[0] - self.t_start:.5f},{r[1]:.5f},{int(r[2])},{r[3]:.3f},{int(r[4])},"
                        f"{'' if r[5] != r[5] else int(r[5])},,,," + "," * (len(STAGES) - 1))
        for r in blk[blk[:, 0] >= t_end - seconds] if len(blk) else []:
            rows.append(f"block,{r[0] - self.t_start:.5f},,,,,,{r[1]:.3f},{int(r[2])},{int(r[3])},"
                        + ",".join(f"{v:.3f}" for v in r[4:]))
        rows[1:] = sorted(rows[1:], key=lambda s: float(s.split(",")[1]))
        return "\n".join(rows)


def status_bits(status):
    """PortAudio's CallbackFlags (or an int) to our bits."""
    if status is None:
        return 0
    try:
        return ((IN_UNDERFLOW if status.input_underflow else 0) | (IN_OVERFLOW if status.input_overflow else 0)
                | (OUT_UNDERFLOW if status.output_underflow else 0) | (OUT_OVERFLOW if status.output_overflow else 0)
                | (PRIMING if getattr(status, "priming_output", False) else 0))
    except AttributeError:
        return int(status) if isinstance(status, int) else 0
