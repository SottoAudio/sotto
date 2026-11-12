import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "failsafe"))
from sotto_failsafe import latency as L  # noqa: E402

FS = 48000


def speechlike(seconds, seed=1):
    """Bursts of band-limited noise with gaps: enough like speech for the correlation."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = rng.standard_normal(n)
    env = np.repeat(rng.random(n // 4800 + 1) > 0.4, 4800)[:n].astype(float)
    k = np.ones(48) / 48
    return 0.1 * np.convolve(x, k, "same") * env


def delayed(x, ms, gain=1.0, noise=0.0, seed=2):
    d = int(round(ms * FS / 1000))
    y = np.concatenate([np.zeros(d), x[:len(x) - d]]) * gain
    if noise:
        y += noise * np.random.default_rng(seed).standard_normal(len(y))
    return y


class GccPhatTests(unittest.TestCase):
    def test_finds_the_delay_whatever_the_level(self):
        x = speechlike(4)
        for ms in (0.0, 10.0, 37.5, 61.25, 140.0):
            got, conf = L.gcc_phat(x, delayed(x, ms, gain=0.05, noise=0.0005), FS)
            self.assertAlmostEqual(got, ms, delta=0.05, msg=f"{ms} ms")
            self.assertGreater(conf, L.MIN_CONFIDENCE)

    def test_unrelated_audio_is_not_trusted(self):
        a, b = speechlike(4, seed=1), speechlike(4, seed=9)
        _, conf = L.gcc_phat(a, b, FS)
        self.assertLess(conf, L.MIN_CONFIDENCE)


class MeasureTests(unittest.TestCase):
    def test_processing_plus_the_loop(self):
        x = speechlike(4)
        out = delayed(x, 10.0, gain=0.7)                  # Sotto Voce: one block
        back = delayed(out, 63.0, gain=0.3, noise=1e-4)   # margin + DVS + network + converters
        r = L.measure(x, out, back, FS, 10.0)
        self.assertEqual(r["state"], "done", r)
        self.assertAlmostEqual(r["processing_ms"], 10.0, delta=0.1)
        self.assertTrue(r["processing_measured"])
        self.assertAlmostEqual(r["loop_ms"], 63.0, delta=0.1)
        self.assertAlmostEqual(r["total_ms"], 73.0, delta=0.2)

    def test_nothing_coming_back(self):
        x = speechlike(4)
        r = L.measure(x, delayed(x, 10.0), np.zeros(len(x)), FS, 10.0)
        self.assertEqual(r["state"], "failed")
        self.assertIn("listener feed", r["error"])

    def test_processing_that_changes_the_signal_falls_back_to_nominal(self):
        x = speechlike(4)
        out = speechlike(4, seed=5)                        # unrecognisable after processing
        back = delayed(out, 50.0)
        r = L.measure(x, out, back, FS, 20.0)
        self.assertEqual(r["state"], "done", r)
        self.assertFalse(r["processing_measured"])
        self.assertEqual(r["processing_ms"], 20.0)
        self.assertAlmostEqual(r["total_ms"], 70.0, delta=0.2)


class MeterTests(unittest.TestCase):
    def test_settles_captures_and_answers(self):
        m = L.LatencyMeter(FS)
        done = []
        m.on_done = done.append
        self.assertTrue(m.start((0, 3, 2), 10.0))           # in, out, listen
        self.assertFalse(m.start((0, 3, 2), 10.0))          # one at a time
        self.assertEqual(m.state()["state"], "measuring")
        x = speechlike(L.SETTLE_S + L.CAPTURE_S + 0.5)
        out = delayed(x, 10.0)
        back = delayed(out, 45.0, gain=0.5)
        cols = np.stack([x, np.zeros_like(x), back, out], axis=1).astype(np.float32)   # in, PROBE, listen, out
        for k in range(0, len(x) - 480, 480):
            m.feed(cols[k:k + 480])
        t0 = time.time()
        while not done and time.time() - t0 < 10:
            time.sleep(0.05)
        self.assertTrue(done)
        r = m.state()
        self.assertEqual(r["state"], "done", r)
        self.assertAlmostEqual(r["total_ms"], 55.0, delta=0.3)

    def test_cancel(self):
        m = L.LatencyMeter(FS)
        done = []
        m.on_done = done.append
        m.start((0, 1, 2), 10.0)
        m.cancel("armed")
        self.assertEqual(m.state()["state"], "failed")
        self.assertEqual(done[0]["error"], "armed")
        self.assertIsNone(m.active)


if __name__ == "__main__":
    unittest.main()


class AppTests(unittest.TestCase):
    """The button end to end on the simulated frame: it starts, finishes, says why if it can't
    match, and puts talker 1's test source back as it was. (The simulated listener hears the
    frame's own mic while disarmed, so there's no real loop to time here.)"""

    def test_measure_from_the_console(self):
        import tempfile
        import threading
        from sotto_failsafe import app as A, cengine
        if cengine.library() is None:
            self.skipTest("native library not built")
        with tempfile.TemporaryDirectory() as d:
            a = A.parse_args(["--simulate", "--no-processing", "--port", "0", "--out-dir", d, "--project",
                              os.path.join(d, "project.json")])
            b = A.BenchApp(a)
            th = threading.Thread(target=b.run, daemon=True)
            th.start()
            try:
                t0 = time.monotonic()
                while time.monotonic() - t0 < 30 and b.cengine is None:
                    time.sleep(0.1)
                self.assertIsNotNone(b.cengine)
                b.test_source({"channel": 1, "kind": "pink"})
                r = b.command({"cmd": "measure_latency"})
                self.assertTrue(r["ok"], r)
                # One at a time is checked in MeterTests: here the first measurement can already have
                # finished on a slow machine (the recording ring catches up in one go), so a second is fine.
                t0 = time.monotonic()
                while time.monotonic() - t0 < 15 and b.state()["latency_measure"]["state"] == "measuring":
                    time.sleep(0.2)
                st = b.state()["latency_measure"]
                self.assertIn(st["state"], ("done", "failed"), st)
                self.assertGreater(st.get("return_db", -120), -70, st)             # the feed was heard
                self.assertEqual(b.tests.desc[0]["kind"], "pink")                   # put back as it was
                self.assertTrue(any(r[2] == "latency_measured" for r in b.recent))
            finally:
                b.quit_ev.set()
                th.join(5)
