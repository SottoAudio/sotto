"""Golden tests: the C fail-safe core must do exactly what the Python core does, block by block."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "failsafe"))
os.environ.setdefault("SOTTO_AUDIO_IO_DIR", os.path.join(ROOT, "build", "audio-io"))

from sotto_failsafe import ccore  # noqa: E402
from sotto_failsafe.core import Config, FailsafeCore  # noqa: E402

B = 480
TOL = 1e-5


def make(n_talkers=4, monitor=True, listen=False):
    cfg = Config(talker_in=list(range(n_talkers)), talker_out=list(range(n_talkers)),
                 talker_names=[f"t{i}" for i in range(n_talkers)], probe_in=[60], probe_out=61, heartbeat_out=60,
                 pgm_in=61, monitor_out=[62, 63] if monitor else [], listen_in=10 if listen else None)
    return cfg, 62, 64


@unittest.skipUnless(ccore.library() is not None, "native library not built (cmake -S engine/audio-io -B build/audio-io)")
class GoldenTests(unittest.TestCase):
    def run_script(self, cfg, n_in, n_out, seconds, script, processed=False, seed=1):
        """script(k, py, c) -> None, called before block k; it acts on both cores the same way.
        Returns the number of blocks compared."""
        rng = np.random.default_rng(seed)
        py = FailsafeCore(cfg)
        c = ccore.CCore(cfg, n_in, n_out)
        prev_probe = np.zeros(B)
        loop = {"on": True}
        nt = len(cfg.talker_in)
        for k in range(int(seconds * cfg.fs / B)):
            script(k, py, c, loop)
            x = (rng.standard_normal((B, n_in)) * 0.05).astype(np.float32)
            for i in range(nt):
                x[:, cfg.talker_in[i]] *= 1 + i
            x[:, cfg.probe_in[0]] = prev_probe if loop["on"] else 0.0
            ret = dry = None
            if processed:
                ret = (rng.standard_normal((B, nt)) * 0.1).astype(np.float32)
                dry = (rng.standard_normal((B, nt)) * 0.1).astype(np.float32)
            o1 = np.zeros((B, n_out), np.float32)
            o2 = np.zeros((B, n_out), np.float32)
            py.process(x, o1, None, returns=ret, dry=dry)
            c.process(x, o2, 0, ret, dry)
            d = float(np.max(np.abs(o1 - o2)))
            self.assertLessEqual(d, TOL, f"block {k}: outputs differ by {d} (output {int(np.argmax(np.max(np.abs(o1 - o2), axis=0))) + 1})")
            s = c.state()
            a = py.arm
            self.assertEqual((s["armed"], s["green"], s["talker_armed"]),
                             (a.armed, a.green_since is not None, list(a.talker_armed)), f"block {k}: state")
            self.assertEqual(s["heartbeat_on"], a.heartbeat_on and "silence_all" not in py.faults)
            prev_probe = o1[:, cfg.probe_out].astype(np.float64)
        c.destroy()
        return k + 1

    def test_start_arm_trip_and_rearm(self):
        cfg, n_in, n_out = make()

        def script(k, py, c, loop):
            if k in (300, 700, 900):                     # 3 s: too soon; 7 s: fine; 9 s: talker after trip
                t = None if k != 900 else 2
                ok, _ = py.arm.request_arm(py.t, "test", t)
                self.assertEqual(c.request_arm(t), ok, f"arm at block {k}")
            if k == 800:
                py.arm.trip_talker(py.t, 2, "test")
                c.trip_talker(2)
            if k == 1000:
                py.arm.request_disarm(py.t, "test")
                c.request_disarm()
        self.assertEqual(self.run_script(cfg, n_in, n_out, 11, script), 1100)

    def test_probe_lost_disarms_and_latches(self):
        cfg, n_in, n_out = make()

        def script(k, py, c, loop):
            if k == 700:
                ok, _ = py.arm.request_arm(py.t, "test")
                self.assertTrue(ok)
                self.assertTrue(c.request_arm())
            if k == 800:
                loop["on"] = False                        # the Riedel loop breaks
            if k == 900:
                loop["on"] = True                         # and comes back: still disarmed
        self.run_script(cfg, n_in, n_out, 10, script)

    def test_processed_returns_and_dry_when_tripped(self):
        cfg, n_in, n_out = make()

        def script(k, py, c, loop):
            if k == 600:
                py.arm.request_arm(py.t, "test")
                c.request_arm()
            if k == 650:
                py.arm.trip_talker(py.t, 1, "test")
                c.trip_talker(1)
        self.run_script(cfg, n_in, n_out, 8, script, processed=True)

    def test_injected_faults(self):
        cfg, n_in, n_out = make()

        def script(k, py, c, loop):
            if k == 600:
                py.arm.request_arm(py.t, "test")
                c.request_arm()
            if k == 620:
                py.inject("silence_all", seconds=0.5)
                c.inject("silence_all", 0.5)
            if k == 700:
                py.inject("mute_returns")
                c.inject("mute_returns")
            if k == 750:
                py.clear_faults()
                c.clear_faults()
        self.run_script(cfg, n_in, n_out, 9, script)

    def test_phones_bus_modes_dim_and_limiter(self):
        cfg, n_in, n_out = make()

        def script(k, py, c, loop):
            plans = {10: dict(ids=["i1", "i3"], mode="clicked", level_db=0.0),
                     100: dict(mode="split"),
                     200: dict(mode="clean", level_db=12.0),             # loud: the limiter works
                     300: dict(dim=True, dim_db=-20.0),
                     400: dict(ids=["o2", "i4"], mode="dry", limit_db=-12.0, dim=False),
                     500: dict(ids=[])}
            if k in plans:
                py.set_solo(**plans[k])
                c.set_solo(py._solo_plan, py.solo)
        self.run_script(cfg, n_in, n_out, 6, script)

    def test_listen_input_and_one_monitor(self):
        cfg, n_in, n_out = make(n_talkers=2, listen=True)
        cfg.monitor_out = [62]

        def script(k, py, c, loop):
            if k == 5:
                py.set_solo(ids=["i1", "i2"], mode="clicked", level_db=-6.0)
                c.set_solo(py._solo_plan, py.solo)
        self.run_script(cfg, n_in, n_out, 3, script)

    def test_events(self):
        cfg, n_in, n_out = make()
        c = ccore.CCore(cfg, n_in, n_out)
        self.assertFalse(c.request_arm())
        ev = c.events()
        self.assertEqual(ev[0][1], "arm_refused")
        c.destroy()


@unittest.skipUnless(ccore.library() is not None, "native library not built")
class ShadowTests(unittest.TestCase):
    def test_shadow_counts_matches_and_differences(self):
        cfg, n_in, n_out = make()
        py = FailsafeCore(cfg)
        sh = ccore.Shadow(py, n_in, n_out)
        rng = np.random.default_rng(3)
        for k in range(50):
            x = (rng.standard_normal((B, n_in)) * 0.05).astype(np.float32)
            o = np.zeros((B, n_out), np.float32)
            py.process(x, o, None)
            if k == 40:
                o[:, 5] += 0.5                           # pretend the Python core wrote something else
            sh.block(x, o, 0, None, None, True)
        self.assertEqual(sh.blocks, 50)
        self.assertEqual(sh.out_bad, 1)
        self.assertEqual(sh.matched, 49)
        self.assertIn("output 6 differs", sh.report())

    def test_mirrored_commands_from_another_thread(self):
        import threading
        cfg, n_in, n_out = make()
        py = FailsafeCore(cfg)
        sh = ccore.Shadow(py, n_in, n_out)
        prev = np.zeros(B)
        rng = np.random.default_rng(4)
        for k in range(900):
            if k == 700:                                   # the console arms from its own thread
                def go():
                    ok, _ = py.arm.request_arm(py.t, "console")
                    sh.arm(None, ok)
                th = threading.Thread(target=go)
                th.start()
                th.join()
            if k == 800:
                py.arm.request_disarm(py.t, "auto", "test")
                sh.disarm()
            x = (rng.standard_normal((B, n_in)) * 0.05).astype(np.float32)
            x[:, 60] = prev
            o = np.zeros((B, n_out), np.float32)
            py.process(x, o, None)
            sh.block(x, o, 0, None, None, True)
            prev = o[:, 61]
        self.assertEqual((sh.matched, sh.blocks), (900, 900), sh.report())


if __name__ == "__main__":
    unittest.main()
