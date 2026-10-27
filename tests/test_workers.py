"""Processing in worker processes: the same timing as the in-process thread, settings and inserts
routed to the right worker, and a worker that dies is noticed, faulted and restarted."""
import os
import signal
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "failsafe"))
sys.path.insert(0, os.path.join(ROOT, "prototype"))

from sotto_failsafe import processing as P  # noqa: E402
from sotto_failsafe import vst3  # noqa: E402
from sotto_failsafe import workers as W  # noqa: E402

LIB = vst3.load_library()
ADELAY = None
for d in filter(None, os.environ.get("SOTTO_EXTRA_VST3", "").split(os.pathsep)):
    if os.path.isdir(os.path.join(d, "adelay.vst3")):
        ADELAY = os.path.join(d, "adelay.vst3")


def paced(pool, blocks, make, on_block=None):
    """Calls exchange() every 10 ms, like the audio callback."""
    outs = []
    t0 = time.perf_counter()
    for b in range(blocks):
        mics, ref = make(b)
        o, _ = pool.exchange(mics.astype(np.float32), ref.astype(np.float32))
        outs.append(o.copy())
        if on_block:
            on_block(b)
        time.sleep(max(0.0, t0 + (b + 1) * 0.01 - time.perf_counter()))
    return np.concatenate(outs, axis=1)


class SplitTests(unittest.TestCase):
    def test_groups(self):
        self.assertEqual(W.split(5, 2), [[0, 1, 2], [3, 4]])
        self.assertEqual(W.split(2, 4), [[0], [1]])
        self.assertEqual(W.auto_workers(4, cores=8), 2)
        self.assertEqual(W.auto_workers(12, cores=4), 1)


class PoolTests(unittest.TestCase):
    def test_same_delay_as_the_thread_and_settings_reach_each_worker(self):
        n = 4
        pool = W.Pool([f"t{i}" for i in range(n)], [P.Settings(mode="bypass", hpf_hz=0) for _ in range(n)],
                      model=None, workers=2, track_delay=False)
        try:
            self.assertEqual(len(pool.workers), 2)
            self.assertTrue(all(w.ready for w in pool.workers))

            def make(b):
                m = np.zeros((n, 480))
                if b == 20:
                    m[:, 0] = [0.1, 0.2, 0.3, 0.4]
                return m, np.zeros(480)
            y = paced(pool, 60, make)
            for i in range(n):
                self.assertEqual(int(np.argmax(np.abs(y[i]))), 20 * 480 + 960)   # same as processing.Processor
                self.assertAlmostEqual(float(np.max(np.abs(y[i]))), 0.1 * (i + 1), places=5)
            r = pool.apply([P.Settings(mode="aec") for _ in range(n)], ["a", "b", "c", "d"])
            self.assertEqual(r["errors"], [])
            time.sleep(0.8)
            st = pool.state()
            self.assertEqual([t["name"] for t in st["talkers"]], ["a", "b", "c", "d"])
            self.assertEqual(st["workers"], 2)
            self.assertEqual(pool.insert_info(3), {"loaded": False, "error": None})
        finally:
            pool.stop()

    def test_a_worker_that_dies_is_faulted_and_restarted(self):
        pool = W.Pool([f"t{i}" for i in range(4)], [P.Settings(mode="bypass") for _ in range(4)],
                      model=None, workers=2, track_delay=False)
        try:
            victim = pool.workers[1]

            def kill(b):
                if b == 60:
                    os.kill(victim.pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
            paced(pool, 200, lambda b: (np.zeros((4, 480)), np.zeros(480)), on_block=kill)
            faults = pool.drain_faults()
            self.assertEqual(sorted({i for i, _ in faults}), [2, 3])            # its talkers only
            self.assertTrue(all("stopped" in why for _, why in faults))
            t0 = time.time()
            while not victim.ready and time.time() - t0 < 30:
                time.sleep(0.1)
            self.assertEqual(victim.restarts, 1)
            self.assertTrue(victim.ready)
        finally:
            pool.stop()

    @unittest.skipUnless(LIB and ADELAY, "plugin host library or the SDK's ADelay example not built")
    def test_insert_loads_in_the_worker_with_its_latency(self):
        s = P.Settings(mode="bypass", hpf_hz=0, insert={"path": ADELAY, "on": True})
        pool = W.Pool(["a", "b"], [P.Settings(mode="bypass", hpf_hz=0), s], model=None, workers=2, track_delay=False)
        try:
            r = pool.apply([P.Settings(mode="bypass", hpf_hz=0), s], ["a", "b"])
            self.assertEqual(r["errors"], [])
            info = pool.insert_info(1)
            self.assertTrue(info["loaded"])
            pid = [q["id"] for q in info["params"] if q["name"] == "Delay"][0]
            self.assertTrue(pool.plugin_param(1, pid, 0.01)["ok"])                # 10 ms
            time.sleep(0.2)

            def make(b):
                m = np.zeros((2, 480))
                if b == 30:
                    m[1, 0] = 0.5
                return m, np.zeros(480)
            y = paced(pool, 80, make)
            self.assertEqual(int(np.argmax(np.abs(y[1]))), 30 * 480 + 960 + 480)
            states = pool.plugin_states()
            self.assertEqual(list(states), [1])
            self.assertTrue(states[1][1].startswith(b"SOTTOVST3STATE01"))
        finally:
            pool.stop()


if __name__ == "__main__":
    unittest.main()
