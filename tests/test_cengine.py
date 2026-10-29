"""The engine in C: its processing hand-over against workers.Pool.exchange, and the whole engine
running in the simulated frame."""
import ctypes
import os
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "failsafe"))
os.environ.setdefault("SOTTO_AUDIO_IO_DIR", os.path.join(ROOT, "build", "audio-io"))

from sotto_failsafe import cengine  # noqa: E402
from sotto_failsafe import processing as P  # noqa: E402
from sotto_failsafe import workers as W  # noqa: E402
from sotto_failsafe.core import Config  # noqa: E402

B = 480


@unittest.skipUnless(cengine.library() is not None, "native library not built")
class ExchangeTests(unittest.TestCase):
    def test_same_results_as_the_python_hand_over(self):
        n, n_in = 4, 8
        tin, pgm = [0, 1, 2, 3], 5
        mk = lambda: W.Pool([f"t{i}" for i in range(n)], [P.Settings(mode="aec", hpf_hz=100) for _ in range(n)],  # noqa: E731
                            model=None, workers=2, track_delay=False, headroom=3)
        a, b = mk(), mk()
        lib = cengine.library()
        try:
            self.assertTrue(all(w.ready for w in a.workers + b.workers))
            e = lib.se_create(None, None, n_in, 8, B, 48000, n, cengine._ints(tin), pgm, 0, cengine._ints([]),
                              cengine._ints([]), -1, -1)
            ce = cengine.CEngine.__new__(cengine.CEngine)
            ce.lib, ce.h, ce._keep = lib, e, []
            for w in b.workers:
                ce._stage_worker(w)
                lib.se_set_ready(e, w.wid, 1)
            lib.se_set_proc(e, 1, len(b.workers), b.headroom, 1)
            lib.se_exchange_test.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                             ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            rng = np.random.default_rng(5)
            ret = np.zeros((B, n))
            dry = np.zeros((B, n))
            stats = (ctypes.c_double * len(cengine.STATS))()
            compared = 0
            missed = False                          # after a late block the two runs can differ for good
            for k in range(200):
                x = (rng.standard_normal((B, n_in)) * 0.1).astype(np.float32)
                out_a, dry_a = a.exchange(np.asarray(x[:, tin], np.float64).T, x[:, pgm])
                lib.se_exchange_test(e, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     ret.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     dry.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                lib.se_stats(e, stats)
                if k > 3 and (a.last_missed or stats[7]):
                    missed = True
                if not missed and k > 3:
                    np.testing.assert_array_equal(ret.T.astype(np.float32), out_a, f"block {k}: processed")
                    np.testing.assert_array_equal(dry.T.astype(np.float32), dry_a, f"block {k}: dry")
                    compared += 1
                time.sleep(0.02)                    # both sets of workers keep up
            lib.se_destroy(e)
            if compared < 60:                       # every compared block matched; too few to count
                self.skipTest(f"only {compared} blocks before this machine fell behind")
        finally:
            a.stop()
            b.stop()

    def test_a_late_worker_gives_dry_at_the_same_delay_with_a_fade(self):
        n, n_in = 2, 4
        pool = W.Pool(["a", "b"], [P.Settings(mode="bypass", hpf_hz=0) for _ in range(n)], model=None, workers=1,
                      track_delay=False, headroom=1)
        lib = cengine.library()
        try:
            e = lib.se_create(None, None, n_in, 4, B, 48000, n, cengine._ints([0, 1]), -1, 0, cengine._ints([]),
                              cengine._ints([]), -1, -1)
            ce = cengine.CEngine.__new__(cengine.CEngine)
            ce.lib, ce.h, ce._keep = lib, e, []
            w = pool.workers[0]
            ce._stage_worker(w)
            lib.se_set_ready(e, 0, 1)
            lib.se_set_proc(e, 1, 1, 1, 1)
            ret, dry = np.zeros((B, n)), np.zeros((B, n))
            blocks = [np.full((B, n_in), 0.01 * (k + 1), np.float32) for k in range(6)]
            for k in range(6):                       # no sleep: the worker can't have answered
                lib.se_exchange_test(e, blocks[k].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     ret.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                     dry.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                if k >= 2:                           # dry from the same delay: block k - headroom - 1
                    self.assertAlmostEqual(float(dry[0, 0]), 0.01 * (k - 1), places=6)
            lib.se_destroy(e)
        finally:
            pool.stop()


class _Args:
    pass


@unittest.skipUnless(cengine.library() is not None, "native library not built")
class SimulatedFrameTests(unittest.TestCase):
    def test_the_c_engine_runs_the_frame(self):
        import tempfile
        import threading
        from sotto_failsafe import app as A
        with tempfile.TemporaryDirectory() as d:
            a = A.parse_args(["--simulate", "--no-processing", "--port", "0", "--out-dir", d, "--project",
                              os.path.join(d, "project.json")])
            b = A.BenchApp(a)
            th = threading.Thread(target=b.run, daemon=True)
            th.start()
            t0 = time.monotonic()
            while time.monotonic() - t0 < 30 and not b.core.arm.can_arm(b.core.t):
                time.sleep(0.1)
            self.assertIsNotNone(b.cengine, "the C engine should be running")
            self.assertTrue(b.core.arm.can_arm(b.core.t), b.core.health_why)
            ok = b.arm(None, "test")
            self.assertTrue(ok["ok"], ok)
            time.sleep(0.5)
            st = b.state()
            self.assertTrue(st["armed"] and st["heartbeat_on"])
            self.assertTrue(st["frame"]["clean_ok"])
            b.command({"cmd": "fault", "name": "silence_all", "seconds": 0.5})
            time.sleep(1.0)
            self.assertFalse(b.state()["armed"])         # PROBE lost: disarmed, and it stays that way
            time.sleep(1.0)
            self.assertFalse(b.state()["armed"])
            self.assertEqual(b.cengine.stats()["late_total"], 0)
            b.quit_ev.set()
            th.join(5)


if __name__ == "__main__":
    unittest.main()
