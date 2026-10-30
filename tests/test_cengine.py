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
            b.quit_ev.set()
            th.join(5)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(cengine.library() is not None, "native library not built")
class ArmingSafetyTests(unittest.TestCase):
    """From the review: the engine must refuse an arm while any check outside its core is red,
    while no audio is arriving, or when the request is stale; a red check restarts the 5 s."""

    def setUp(self):
        from sotto_failsafe import nativeio
        from sotto_failsafe.core import FailsafeCore
        self.cfg = Config(talker_in=[0, 1], talker_out=[0, 1], talker_names=["a", "b"], probe_in=[6], probe_out=6,
                          heartbeat_out=7, monitor_out=[])
        self.n_in, self.n_out = 8, 8
        self.nat = nativeio.NativeIO(self.n_in, self.n_out, B, B, 48000, margin=4800)   # 100 ms: a busy test machine never makes it late by accident
        self.core = FailsafeCore(self.cfg)
        self.ce = cengine.CEngine(self.nat, self.cfg, self.n_in, self.n_out, B)
        self.ce.attach(self.core)
        self.ce.start()
        self.y = np.zeros((B, self.n_out), np.float32)
        self.hb = []

    def tearDown(self):
        self.ce.destroy()
        self.nat.destroy()

    def feed(self, seconds):
        """Soundcard cycles in real time, PROBE looped back, the C engine running on its thread."""
        rng = np.random.default_rng(0)
        for _ in range(int(seconds * 100)):
            x = (rng.standard_normal((B, self.n_in)) * 0.01).astype(np.float32)
            x[:, 6] = self.y[:, 6]
            self.nat.cycle(x, self.y, B)
            self.hb.append(float(np.max(np.abs(self.y[:, 7]))))
            time.sleep(0.01)
        self.ce.sync()

    def arm(self):
        return self.core.arm.request_arm(self.core.t, "test")

    def test_refusals_and_a_clean_arm(self):
        self.feed(6.5)
        self.assertTrue(self.core.arm.can_arm(self.core.t), self.core.health_why)
        self.ce.set_ext(False, "Dante network port: en7 has no link.")
        time.sleep(0.05)
        ok, msg = self.arm()
        self.assertFalse(ok)
        self.assertIn("en7 has no link", msg)
        self.ce.set_ext(True)
        self.feed(0.5)
        ok, msg = self.arm()                    # the red check restarted the 5 s
        self.assertFalse(ok, msg)
        self.feed(5.5)
        ok, msg = self.arm()
        self.assertTrue(ok, msg)
        self.feed(0.5)
        self.assertTrue(self.core.arm.armed)
        self.assertGreater(max(self.hb[-20:]), 0.05, "HEARTBEAT plays while armed")
        self.core.arm.request_disarm(self.core.t, "test", "test")
        time.sleep(0.2)                          # no audio arriving now
        ok, msg = self.arm()
        self.assertFalse(ok)
        self.assertIn("no audio", msg)
        n = self.ce.lib.se_command(self.ce.h, cengine.CMD_ARM, -1, 0, self.ce.lib.se_clock() - 1.0)
        res = ctypes.c_double(0)
        t0 = time.monotonic()
        while not self.ce.lib.se_result(self.ce.h, n, ctypes.byref(res)) and time.monotonic() - t0 < 1:
            time.sleep(0.001)
        self.assertEqual(res.value, -2, "a stale arm request is refused")
        self.ce.sync()
        self.assertFalse(self.core.arm.armed)

    def test_engine_late_disarms_and_the_latch_clears_only_on_a_manual_rearm(self):
        self.feed(6.5)
        ok, _ = self.arm()
        self.assertTrue(ok)
        self.feed(0.3)
        self.core.inject("freeze", ms=400)       # the engine thread misses its deadlines (beyond the 100 ms margin)
        self.feed(0.8)
        st = self.ce.stats()
        self.assertGreater(st["late_total"], 0)
        self.assertFalse(self.core.arm.armed)
        self.assertTrue(self.nat.heartbeat_blocked())
        ok, msg = self.arm()                     # not green for 5 s after a late engine
        self.assertFalse(ok, msg)
        self.feed(5.5)
        ok, msg = self.arm()                     # then 5 s of green, as in the Python engine
        self.assertFalse(ok, msg)
        self.feed(5.0)
        ok, msg = self.arm()
        self.assertTrue(ok, msg)
        self.feed(0.2)
        self.assertFalse(self.nat.heartbeat_blocked())
