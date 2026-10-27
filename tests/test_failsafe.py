"""Fail-safe bench tool: tone detection, the arming latch, and an end-to-end run against the
simulated Riedel frame (VOX hold, PROBE loop, listener switching)."""
import csv
import os
import sys
import tempfile
import unittest
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "failsafe"))

from sotto_failsafe.arming import ArmState  # noqa: E402
from sotto_failsafe.core import Config, FailsafeCore  # noqa: E402
from sotto_failsafe.signals import Tone, tone_present  # noqa: E402
from sotto_failsafe.sim import SimRiedel  # noqa: E402
from sotto_failsafe import analyse  # noqa: E402
from sotto_failsafe.app import _int24  # noqa: E402

FS = 48000


class ToneTests(unittest.TestCase):
    def test_tone_detected(self):
        x = Tone(2000, -20).block(480)
        self.assertTrue(tone_present(x, 2000, FS))

    def test_noise_and_other_tones_rejected(self):
        rng = np.random.default_rng(1)
        self.assertFalse(tone_present(rng.standard_normal(480) * 0.1, 2000, FS))
        self.assertFalse(tone_present(Tone(1000, -20).block(480), 2000, FS))
        self.assertFalse(tone_present(np.zeros(480), 2000, FS))

    def test_tone_continuous_across_blocks(self):
        t = Tone(1000, -6)
        a, b = t.block(100), t.block(100)
        whole = Tone(1000, -6).block(200)
        self.assertTrue(np.allclose(np.concatenate([a, b]), whole))


class ArmingTests(unittest.TestCase):
    def test_starts_disarmed_and_needs_hold(self):
        s = ArmState(4, arm_hold_s=5)
        self.assertFalse(s.heartbeat_on)
        s.update_health(0.0, True)
        ok, _ = s.request_arm(3.0, "console")
        self.assertFalse(ok)
        ok, _ = s.request_arm(5.1, "console")
        self.assertTrue(ok)
        self.assertTrue(s.heartbeat_on and all(s.talker_armed))

    def test_fault_latches_until_engineer_rearms(self):
        s = ArmState(4, arm_hold_s=5)
        s.update_health(0.0, True)
        s.request_arm(6.0, "console")
        s.update_health(7.0, False, "PROBE missing")
        self.assertFalse(s.armed)
        s.update_health(7.5, True)                    # fault clears
        self.assertFalse(s.armed)                     # still disarmed: never automatic
        self.assertFalse(s.request_arm(9.0, "panel")[0])   # only 1.5 s green
        self.assertTrue(s.request_arm(12.6, "companion")[0])
        sources = [e.source for e in s.drain_events() if e.kind in ("armed", "arm_refused")]
        self.assertEqual(sources, ["console", "panel", "companion"])

    def test_talker_trip_latches_and_rearms_alone(self):
        s = ArmState(4, arm_hold_s=5)
        s.update_health(0.0, True)
        s.request_arm(6.0, "console")
        s.trip_talker(7.0, 2, "silent output")
        self.assertEqual(s.talker_armed, [True, True, False, True])
        self.assertTrue(s.armed)                      # others stay clean, heartbeat stays on
        s.request_arm(8.0, "console", talker=2)
        self.assertEqual(s.talker_armed, [True] * 4)

    def test_talker_arm_refused_while_disarmed(self):
        s = ArmState(4, arm_hold_s=5)
        s.update_health(0.0, True)
        ok, msg = s.request_arm(6.0, "console", talker=1)
        self.assertFalse(ok)
        self.assertEqual(s.talker_armed, [False] * 4)


def run_bench(core, sim, seconds, block, record=None, actions=None):
    """Drive the core against the simulated frame, faster than real time."""
    cfg = core.cfg
    n_in = max(cfg.talker_in + cfg.probe_in + [cfg.listen_in]) + 1
    n_out = sim.n_out
    actions = sorted(actions or [], key=lambda a: a[0])
    timeline = []
    for _ in range(int(seconds * FS / block)):
        while actions and core.t >= actions[0][0]:
            actions.pop(0)[1]()
        x = sim.make_input(block, n_in)
        out = np.zeros((block, n_out), np.float32)
        core.process(x, out)
        sim.take_output(out)
        timeline.append((core.t, core.arm.armed, sim.clean_ok))
        if record is not None:
            record.append(np.stack([x[:, cfg.listen_in], out[:, cfg.heartbeat_out]], axis=1))
    return timeline


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(listen_in=5)
        self.core = FailsafeCore(self.cfg)
        self.sim = SimRiedel(self.cfg, vox_hold_ms=400, logic_ms=20)
        self.block = 240

    def first_time(self, timeline, after, idx, value):
        for t, armed, ok in timeline:
            if t >= after and (armed, ok)[idx] == value:
                return t
        return None

    def test_full_cycle(self):
        core, rec, res = self.core, [], {}
        acts = [
            (3.0, lambda: res.__setitem__("early", core.arm.request_arm(core.t, "console")[0])),
            (6.5, lambda: res.__setitem__("arm1", core.arm.request_arm(core.t, "console")[0])),
            (9.0, lambda: core.inject("stop_heartbeat")),
            (10.0, lambda: res.__setitem__("arm2", core.arm.request_arm(core.t, "panel")[0])),
            (13.0, lambda: core.inject("silence_all", seconds=2.0)),
            (16.0, lambda: res.__setitem__("arm_too_soon", core.arm.request_arm(core.t, "companion")[0])),
            (21.5, lambda: res.__setitem__("arm3", core.arm.request_arm(core.t, "companion")[0])),
        ]
        tl = run_bench(core, self.sim, 24.0, self.block, rec, acts)
        self.assertFalse(res["early"], "arming must wait for 5 s of green")
        self.assertTrue(res["arm1"])
        self.assertTrue(res["arm2"])
        self.assertFalse(res["arm_too_soon"], "no re-arm until 5 s green after the fault")
        self.assertTrue(res["arm3"])

        # Riedel follows the heartbeat: clean after arming, dry ~hold + logic after it stops
        on1 = self.first_time(tl, 6.5, 1, True)
        self.assertIsNotNone(on1)
        self.assertLess(on1 - 6.5, 0.1)
        off1 = self.first_time(tl, 9.0, 1, False)
        self.assertGreaterEqual(off1 - 9.0, 0.40)
        self.assertLess(off1 - 9.0, 0.50)

        # Laptop "dies" at 13 s: PROBE vanishes, app disarms itself; Riedel drops to dry
        disarm = self.first_time(tl, 13.0, 0, False)
        self.assertLess(disarm - 13.0, 0.3)
        off2 = self.first_time(tl, 13.0, 1, False)
        self.assertLess(off2 - 13.0, 0.55)
        # ...and nothing goes back to clean on its own after the fault clears at 15 s
        self.assertTrue(all(not ok for t, _, ok in tl if 15.0 <= t < 21.5))
        on3 = self.first_time(tl, 21.5, 1, True)
        self.assertIsNotNone(on3)

        # The timing analysis reads the same story from a recording
        with tempfile.TemporaryDirectory() as d:
            take = "test"
            data = np.concatenate(rec)
            with wave.open(os.path.join(d, f"{take}_bench.wav"), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(3)
                w.setframerate(FS)
                w.writeframes(_int24(data.reshape(-1)))
            with open(os.path.join(d, f"{take}_channels.txt"), "w") as f:
                f.write("1\tin: LISTEN\n2\tout: HEARTBEAT\n")
            with open(os.path.join(d, f"{take}_events.csv"), "w", newline="") as f:
                cw = csv.writer(f)
                cw.writerow(["audio_time_s", "clock", "event", "detail", "source"])
                cw.writerow(["0.0", "", "recording_started", "", "test"])
                cw.writerow(["9.0", "", "fault_injected", "stop heartbeat", "test"])
                cw.writerow(["13.0", "", "fault_injected", "silence_all", "test"])
            res = analyse.analyse(d)
        hb, sil = res[0], res[1]
        self.assertTrue(hb["marker_before"])
        self.assertTrue(400 <= hb["switch_ms"] <= 520, hb)
        self.assertLess(hb["silence_ms"], 60, "stop-heartbeat should switch without silence")
        self.assertTrue(350 <= sil["silence_ms"] <= 560, sil)   # the dropout a listener hears
        self.assertIsNotNone(sil["dry_back_ms"])


if __name__ == "__main__":
    unittest.main()
