"""Measure latency: the real time from a talker's input to what a listener hears, with a loop.

While disarmed, a speech test source plays into talker 1. The engine's recording ring already
carries, block by block on one timeline, talker 1's input (as the engine saw it), talker 1's clean
output (as the engine wrote it) and the listener feed input (--listen-in), which on the bench is
talker 1's output looped back: Dante out, an interface's DAC, a cable, its ADC, Dante back in.

Cross-correlating them (GCC-PHAT: the phase alone, so level and EQ don't matter) gives:
  * processing: talker 1's input -> its output, inside Sotto Voce;
  * the loop: talker 1's output -> the listener feed, i.e. the engine's margin, the soundcard
    out, the network, the interface, and the soundcard back in;
  * total = processing + loop. That is the full mic-to-headset delay through the Mac, with the
    loop's converters and network standing in for the panel's.
"""

import threading
import time

import numpy as np

SETTLE_S = 0.6           # after the test source starts, before listening
CAPTURE_S = 4.0          # how much to correlate
MAX_MS = 600.0           # the longest delay looked for
MIN_CONFIDENCE = 4.0     # the peak against the next best one; below this the answer isn't trusted


def gcc_phat(ref, sig, fs, max_ms=MAX_MS):
    """How much later `sig` is than `ref`, in ms (>= 0), and how clear the peak is."""
    ref = np.asarray(ref, np.float64) - np.mean(ref)
    sig = np.asarray(sig, np.float64) - np.mean(sig)
    if not (np.any(ref) and np.any(sig)):
        return None, 0.0
    n = 1 << int(np.ceil(np.log2(len(ref) + len(sig))))
    cross = np.fft.rfft(sig, n) * np.conj(np.fft.rfft(ref, n))
    cross /= np.abs(cross) + 1e-12
    r = np.fft.irfft(cross, n)
    m = min(int(fs * max_ms / 1000), n // 2 - 2)
    seg = r[:m + 1]
    k = int(np.argmax(seg))
    peak = seg[k]
    if 0 < k < m:                                  # parabolic: a fraction of a sample
        a, b, c = seg[k - 1], seg[k], seg[k + 1]
        d = (a - 2 * b + c)
        frac = 0.5 * (a - c) / d if d else 0.0
    else:
        frac = 0.0
    guard = max(1, int(fs * 0.002))                # the next best peak, 2 ms or more away
    rest = np.concatenate([seg[:max(0, k - guard)], seg[k + guard + 1:]])
    second = float(np.max(np.abs(rest))) if len(rest) else 0.0
    conf = float(peak / second) if second > 0 else float("inf")
    return 1000.0 * (k + frac) / fs, conf


class LatencyMeter:
    """One measurement at a time. feed() is called with each recorded block (from the C engine's
    recording ring, on the control thread); the maths runs on its own thread."""

    def __init__(self, fs):
        self.fs = fs
        self.lock = threading.Lock()
        self.active = None            # {"cols": (in, out, listen), "skip": n, "need": n, "blocks": [...]}
        self.result = None            # the last answer, for the console and the diagnostics
        self.on_done = None           # called with the result when the maths is done

    def start(self, cols, processing_ms):
        """cols: the recording's columns for talker 1's input, talker 1's output and the listener feed."""
        with self.lock:
            if self.active is not None:
                return False
            self.active = {"cols": cols, "skip": int(SETTLE_S * self.fs), "need": int(CAPTURE_S * self.fs),
                           "have": 0, "blocks": [], "processing_ms": processing_ms, "t0": time.time()}
            self.result = {"state": "measuring", "started": time.time()}
            return True

    def cancel(self, why):
        with self.lock:
            if self.active is None:
                return
            self.active = None
        self._publish({"state": "failed", "error": why, "when": time.time()})

    def _publish(self, res):
        """on_done first (it puts talker 1's test source back), then the answer is shown: so anything
        that sees "done" also sees the test source as it was."""
        if self.on_done:
            try:
                self.on_done(res)
            except Exception:
                pass
        with self.lock:
            self.result = res

    def feed(self, blk):
        with self.lock:
            a = self.active
            if a is None:
                return
            n = blk.shape[0]
            if a["skip"] > 0:
                a["skip"] -= n
                return
            a["blocks"].append(blk[:, list(a["cols"])].copy())
            a["have"] += n
            if a["have"] < a["need"]:
                return
            self.active = None
        threading.Thread(target=self._finish, args=(a,), name="latency-measure", daemon=True).start()

    def _finish(self, a):
        x = np.concatenate(a["blocks"], axis=0)
        res = measure(x[:, 0], x[:, 1], x[:, 2], self.fs, a["processing_ms"])
        res["when"] = time.time()
        self._publish(res)

    def state(self):
        with self.lock:
            return None if self.result is None else dict(self.result)


def level_db(x):
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    return -120.0 if r <= 1e-6 else round(float(20 * np.log10(r)), 1)


def measure(talker_in, talker_out, listen, fs, processing_ms_nominal):
    """The three signals on the engine's timeline -> the answer (see the module docstring)."""
    lv = {"input_db": level_db(talker_in), "output_db": level_db(talker_out), "return_db": level_db(listen)}
    if lv["return_db"] < -70:
        return {"state": "failed", **lv,
                "error": "Nothing is coming back on the listener feed input. Check that talker 1's return is "
                         "looped back to it (Dante out, the interface, a cable, Dante back in)."}
    if lv["output_db"] < -70:
        return {"state": "failed", **lv, "error": "Talker 1's output is silent, so there was nothing to time."}
    loop_ms, loop_conf = gcc_phat(talker_out, listen, fs)
    if loop_ms is None or loop_conf < MIN_CONFIDENCE:
        return {"state": "failed", **lv, "confidence": round(loop_conf, 1),
                "error": "The listener feed doesn't match talker 1's output clearly enough to time it. Check it "
                         "carries talker 1's return and nothing else."}
    proc_ms, proc_conf = gcc_phat(talker_in, talker_out, fs)
    measured_proc = proc_ms is not None and proc_conf >= MIN_CONFIDENCE
    p = proc_ms if measured_proc else processing_ms_nominal
    return {"state": "done", **lv,
            "total_ms": round(p + loop_ms, 1),
            "processing_ms": round(p, 1), "processing_measured": bool(measured_proc),
            "loop_ms": round(loop_ms, 1), "confidence": round(min(loop_conf, proc_conf if measured_proc else loop_conf), 1)}
