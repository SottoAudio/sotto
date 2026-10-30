"""Processing in worker processes: the same timing as the in-process thread, settings and inserts
routed to the right worker, and a worker that dies is noticed, faulted and restarted."""
import os
import signal
import sys
import threading
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

    def test_headroom_absorbs_bunched_callbacks(self):
        """Some soundcards deliver 10 ms blocks in bunches (two back to back, then a 20 ms gap).
        With one block of headroom the second of each pair finds no result; with two it does."""
        n = 2
        pool = W.Pool(["a", "b"], [P.Settings(mode="bypass", hpf_hz=0) for _ in range(n)],
                      model=None, workers=1, track_delay=False)
        try:
            def bunched(blocks):
                outs = []
                t0 = time.perf_counter()
                for b in range(blocks):
                    m = np.zeros((n, 480), np.float32)
                    if b == 40:
                        m[:, 0] = 0.5
                    o, _ = pool.exchange(m, np.zeros(480, np.float32))
                    outs.append(o.copy())
                    if b % 2:                          # pairs: 0 ms apart, then a 20 ms gap
                        time.sleep(max(0.0, t0 + (b + 1) * 0.01 - time.perf_counter()))
                return np.concatenate(outs, axis=1)

            bunched(100)
            self.assertGreater(pool.misses, 15)                   # 1 block of headroom: late every other block (counted after block 50)
            pool.set_headroom(2)
            before = pool.misses
            y = bunched(120)
            late = pool.misses - before
            if late > 6 and os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted":
                self.skipTest(f"{late} of 120 blocks late: this shared CI virtual machine can't keep 10 ms timing")
            self.assertLessEqual(late, 6)                        # 2 blocks: (almost) none late, on a busy CI machine
            self.assertEqual(int(np.argmax(np.abs(y[0]))), 40 * 480 + 960 + 480)   # 10 ms more delay
            self.assertEqual(pool.state()["latency_ms"], 30.0)
        finally:
            pool.stop()

    def test_workers_can_run_the_delay_tracker(self):
        # The delay tracker starts a helper process of its own, which daemon processes can't.
        # (The first M2 bench run failed exactly here: both workers stopped at start.)
        pool = W.Pool(["a", "b"], [P.Settings(mode="aec") for _ in range(2)], model=None, workers=2, track_delay=True)
        try:
            self.assertTrue(all(w.ready for w in pool.workers), [w.failed for w in pool.workers])
            paced(pool, 50, lambda b: (np.zeros((2, 480)), np.zeros(480)))
            self.assertTrue(pool.ok, pool.why)
        finally:
            pool.stop()
        for w in pool.workers:
            self.assertFalse(w.proc.is_alive())

    def test_a_worker_that_cant_start_is_not_restarted_forever(self):
        pool = W.Pool(["a", "b"], [P.Settings() for _ in range(2)], model=os.path.join(ROOT, "no-such-model"),
                      workers=1, track_delay=False, ready_timeout=30)
        try:
            w = pool.workers[0]
            self.assertFalse(w.ready)
            self.assertTrue(w.failed)                                       # the reason, from the worker
            t0 = time.time()
            while not w.given_up and time.time() - t0 < 10:
                time.sleep(0.1)
            time.sleep(1.5)
            self.assertTrue(w.given_up)
            self.assertEqual(w.restarts, 0)
            ok, why = pool.health()
            self.assertFalse(ok)
            self.assertIn("can't run", why)
            self.assertEqual(sorted({i for i, _ in pool.drain_faults()}), [0, 1])     # both talkers to dry
        finally:
            pool.stop()

    def test_a_plugin_that_crashes_a_worker_is_not_loaded_again(self):
        pool = W.Pool(["a", "b"], [P.Settings(mode="bypass") for _ in range(2)], model=None, workers=1,
                      track_delay=False)
        try:
            w = pool.workers[0]
            bad = "/Library/Audio/Plug-Ins/VST3/Crashy.vst3"
            # As if the worker were in the middle of opening that plugin when it died:
            w.loading = {bad}
            slot = {"ev": threading.Event(), "wid": w.wid}
            with pool._wait_lock:
                pool._waiting[99] = slot
            t0 = time.time()
            os.kill(w.pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
            self.assertTrue(slot["ev"].wait(5), "the waiting request wasn't told")
            self.assertLess(time.time() - t0, 5)
            self.assertFalse(slot["ok"])
            self.assertIn(bad, pool.crashed)
            while w.restarts < 1 and time.time() - t0 < 10:
                time.sleep(0.05)
            # Choosing it again (sent straight away: it waits for the restarted worker): Sotto Voce leaves it out and says why, instead of crashing the worker again.
            s = P.Settings(mode="bypass", insert={"path": bad, "on": True})
            r = pool.apply([s, P.Settings(mode="bypass")], ["a", "b"])
            self.assertTrue(any("Crashy crashed" in e for e in r["errors"]), r)
            time.sleep(0.8)
            self.assertIn("Crashy crashed", pool.state()["talkers"][0]["insert_error"])
            self.assertEqual(w.restarts, 1)
        finally:
            pool.stop()

    @unittest.skipUnless(LIB and ADELAY, "plugin host library or the SDK's ADelay example not built")
    def test_trial_open_in_a_throwaway_process(self):
        ok, why = vst3.trial(ADELAY)
        self.assertTrue(ok, why)
        ok, why = vst3.trial("/no/such/plugin.vst3")
        self.assertFalse(ok)

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


class AnswerTimingTests(unittest.TestCase):
    def test_buckets_and_report(self):
        t = W.AnswerTiming()
        for a in (0.5, 3, 7, 9, 12, 30):
            t.add(int(0.2e6), int(a * 1e6))
        t.add(-1, 5)                                   # no hand-over time: ignored
        t.skipped = 2
        r = t.report()
        self.assertEqual(r["blocks"], 6)
        self.assertEqual(r["over_8_ms"], 3)
        self.assertEqual(r["over_10_ms"], 2)
        self.assertEqual(r["skipped"], 2)
        self.assertEqual(r["answer_ms_max"], 30.0)
        self.assertEqual(sum(r["within_ms"].values()), 6)

    def test_workers_report_how_quickly_they_answer(self):
        """The workers time each block from the hand-over; with a wait that really sleeps, they
        wake within a millisecond or two of it (a polling wait could be many ms late)."""
        pool = W.Pool(["a", "b"], [P.Settings(mode="bypass", hpf_hz=0) for _ in range(2)], model=None, workers=2,
                      track_delay=False, headroom=1)
        try:
            self.assertTrue(all(w.ready for w in pool.workers))
            rng = np.random.default_rng(1)
            paced(pool, 300, lambda b: (rng.standard_normal((2, W.HOP)) * 0.1, np.zeros(W.HOP)))
            time.sleep(0.8)                              # the next state reports
            for w in pool.state()["worker_detail"]:
                an = w["answer"]
                self.assertIsNotNone(an, w)
                self.assertGreater(an["blocks"], 200)
                self.assertLess(an["wake_ms_avg"], 3.0, an)
                self.assertLess(an["answer_ms_avg"], 5.0, an)
        finally:
            t0 = time.monotonic()
            pool.stop()
            self.assertLess(time.monotonic() - t0, 10.0)    # stopping wakes the waiting workers


class DeclickTests(unittest.TestCase):
    def test_switching_to_unprocessed_audio_is_faded_not_stepped(self):
        p = W.Pool.__new__(W.Pool)                         # just the hand-over's fade, no processes
        n = 2
        p.out_buf = np.zeros((n, W.HOP), np.float32)
        p._was_missed = np.zeros(n, bool)
        p._last_out = np.zeros(n, np.float32)
        p._fade = np.linspace(1.0, 0.0, W.RAMP, endpoint=False).astype(np.float32)
        p.out_buf[:] = 0.8                                 # processed: ends at 0.8
        p._declick(np.array([False, False]))
        p.out_buf[:] = -0.5                                # talker 1 late: unprocessed audio, a 1.3 step
        p._declick(np.array([True, False]))
        self.assertAlmostEqual(float(p.out_buf[0, 0]), 0.8, places=5)          # starts where it left off
        self.assertLess(float(np.max(np.abs(np.diff(p.out_buf[0])))), 0.02)    # no step
        self.assertAlmostEqual(float(p.out_buf[0, -1]), -0.5, places=5)
        self.assertAlmostEqual(float(p.out_buf[1, 0]), -0.5, places=5)         # talker 2 didn't switch: untouched
