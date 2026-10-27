"""Sotto Voce processing: timing and alignment, modes, the batched speech model, the per-talker
fault path, operator profiles from the project, and (when the plugin host library has been built)
a real VST3 insert."""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "failsafe"))
sys.path.insert(0, os.path.join(ROOT, "prototype"))

from sotto_failsafe import processing as P  # noqa: E402
from sotto_failsafe import vst3  # noqa: E402

MODEL = P.model_dir()
LIB = vst3.load_library()
ADELAY = None
for d in filter(None, os.environ.get("SOTTO_EXTRA_VST3", "").split(os.pathsep)):
    if os.path.isdir(os.path.join(d, "adelay.vst3")):
        ADELAY = os.path.join(d, "adelay.vst3")


def run(proc, blocks, make, pace=False):
    outs = []
    t0 = time.perf_counter()
    for b in range(blocks):
        mics, ref = make(b)
        o, _ = proc.exchange(mics, ref)
        outs.append(o.copy())
        if pace:
            time.sleep(max(0.0, t0 + (b + 1) * 0.01 - time.perf_counter()))
        else:
            for _ in range(200):                 # wait for the worker, like a real-time callback would
                if proc.q_out.qsize() or proc.q_in.qsize() == 0:
                    break
                time.sleep(0.001)
            time.sleep(0.002)
    return np.concatenate(outs, axis=1)


class ProcessingTests(unittest.TestCase):
    def test_bypass_delay_is_two_blocks(self):
        pr = P.Processor(["a"], [P.Settings(mode="bypass", hpf_hz=0)], model=None, track_delay=False)
        try:
            def make(b):
                m = np.zeros((1, 480))
                if b == 20:
                    m[0, 0] = 0.5
                return m, np.zeros(480)
            y = run(pr, 60, make)
            self.assertEqual(int(np.argmax(np.abs(y[0]))), 20 * 480 + 960)   # 20 ms through the chain
        finally:
            pr.stop()

    def test_echo_removed_from_pgm_bleed(self):
        rng = np.random.default_rng(0)
        n_blocks = 900
        ref = rng.standard_normal(n_blocks * 480) * 0.1
        bleed = np.concatenate([np.zeros(2400), ref[:-2400]]) * 0.5       # PA 50 ms away
        pr = P.Processor(["a"], [P.Settings(mode="aec", hpf_hz=0, tail_ms=256)], model=None, track_delay=False)
        try:
            pr.chains[0].set_delay(2400)
            y = run(pr, n_blocks, lambda b: (bleed[None, b * 480:(b + 1) * 480], ref[b * 480:(b + 1) * 480]))
            last = slice(-200 * 480, None)
            drop = 10 * np.log10(np.mean(bleed[last] ** 2) / np.mean(y[0, last] ** 2))
            self.assertGreater(drop, 10.0, f"only {drop:.1f} dB of the PGM bleed removed")
        finally:
            pr.stop()

    def test_settings_from_project_follow_the_operator(self):
        project = {
            "proc": {"mode": "full", "supp": 10, "atten": 20, "hpf": 80, "tail": 512, "trim": 1},
            "ch": [{"name": "Talker 1", "op": "op-x", "mode": "default"}, {"name": "Talker 2", "op": None, "mode": "aec"}],
            "operators": [{"id": "op-x", "name": "Show caller", "mode": "aecns", "trim": 2,
                           "ins": {"plugin": "/p/CEDAR Voxis.vst3", "on": True, "vst3": {"/p/CEDAR Voxis.vst3": "AAAA"}}}],
        }
        s, names = P.settings_from_project(project, ["t1", "t2"])
        self.assertEqual(names, ["Show caller", "Talker 2"])
        self.assertEqual((s[0].mode, s[0].trim_db, s[0].insert["path"], s[0].insert["state"]), ("aecns", 3.0, "/p/CEDAR Voxis.vst3", "AAAA"))
        self.assertEqual((s[1].mode, s[1].insert, s[1].tail_ms, s[1].hpf_hz), ("aec", {}, 512, 80.0))

    def test_late_processing_makes_the_engine_unhealthy(self):
        pr = P.Processor(["a"], [P.Settings(mode="bypass")], model=None, track_delay=False)
        try:
            pr.stop_ev.set()
            pr.th.join()                                   # a stalled worker
            for _ in range(60):
                pr.exchange(np.zeros((1, 480)), np.zeros(480))
            ok, why = pr.health()
            self.assertFalse(ok)
            self.assertIn("keep up", why)
        finally:
            pr.stop()

    @unittest.skipUnless(MODEL, "speech model not downloaded (prototype/tools/get_models.py)")
    def test_full_clean_runs_and_modes_switch(self):
        rng = np.random.default_rng(1)
        pr = P.Processor(["a", "b"], [P.Settings(mode="full"), P.Settings(mode="aecns")], model=MODEL, track_delay=False)
        try:
            y = run(pr, 80, lambda b: (rng.standard_normal((2, 480)) * 0.02, rng.standard_normal(480) * 0.1))
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertEqual(pr.drain_faults(), [])
            pr.configure([P.Settings(mode="aecns"), P.Settings(mode="full")], [(None, "", False)] * 2)
            y = run(pr, 40, lambda b: (rng.standard_normal((2, 480)) * 0.02, rng.standard_normal(480) * 0.1))
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertEqual([c.s.mode for c in pr.chains], ["aecns", "full"])
            st = pr.state()                                # computing time is reported next to the total
            self.assertGreater(st["cpu_ms"], 0)
            self.assertLessEqual(st["cpu_ms"], st["proc_ms"] * 1.5 + 0.5)
            self.assertTrue(st["thread"])
            self.assertEqual(set(st["stage_ms"]), {"front", "model", "back"})
            self.assertGreater(st["stage_ms"]["model"], 0)
        finally:
            pr.stop()

    @unittest.skipUnless(LIB and ADELAY, "plugin host library or the SDK's ADelay example not built")
    def test_vst3_insert_runs_with_constant_latency(self):
        plug = vst3.Plugin(LIB, ADELAY, block=480)
        pid = [q["id"] for q in plug.info()["params"] if q["name"] == "Delay"][0]
        plug.set_param(pid, 0.01)                         # 10 ms
        s = P.Settings(mode="bypass", hpf_hz=0, insert={"path": ADELAY, "on": True})
        pr = P.Processor(["a"], [s], model=None, track_delay=False)
        try:
            pr.configure([s], [(plug, "ADelay", True)])

            def make(b):
                m = np.zeros((1, 480))
                if b == 20:
                    m[0, 0] = 0.5
                return m, np.zeros(480)
            y = run(pr, 60, make)
            self.assertEqual(int(np.argmax(np.abs(y[0]))), 20 * 480 + 960 + 480)   # chain + the plugin's 10 ms
            state = plug.get_state()
            self.assertTrue(state.startswith(b"SOTTOVST3STATE01"))
        finally:
            pr.stop()


if __name__ == "__main__":
    unittest.main()
