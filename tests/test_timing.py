"""The detailed audio timing record behind the diagnostics file."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "failsafe"))

from sotto_failsafe.timing import OUT_UNDERFLOW, TimingTrace  # noqa: E402


class TimingTraceTests(unittest.TestCase):
    def test_skipped_cycles_overruns_and_late_processing_are_reported(self):
        tr = TimingTrace(48000, 160, 480, seconds=5)
        period = 160 / 48000
        t = adc = 100.0
        k = 0
        for i in range(600):
            adc += period
            t += period
            if i == 300:                                    # Core Audio skips 6 cycles (20 ms)
                adc += 6 * period
                t += 6 * period
            cb = 12.0 if i == 301 else 0.5                  # one callback runs far over its 3.3 ms
            tr.cycle(t, adc, 160, cb, OUT_UNDERFLOW if i == 301 else 0)
            if i % 3 == 2:
                tr.block_done(t, 2.0, [0.1, 0.5, 0.8, 0.0, 0.1], 1 if k % 50 == 0 else 0, 320)
                k += 1
        self.assertEqual((tr.skipped, tr.skip_events), (6, 1))
        s = tr.summary()
        self.assertIn("Skipped cycles (capture time jumped): 6 cycles in 1 places = 20 ms", s)
        self.assertIn("Over the 3.33 ms budget: 1 of", s)
        self.assertIn("output underflow 1", s)
        self.assertIn("processing came back late (unprocessed audio played instead): 4 of 200", s)
        csv = tr.csv(10).splitlines()
        self.assertTrue(csv[0].startswith("kind,t_s,capture_s"))
        self.assertEqual(sum(1 for r in csv if r.startswith("cycle,")), 600)

    def test_a_soundcard_restart_is_not_skipped_cycles(self):
        tr = TimingTrace(48000, 480, 480, seconds=5)
        period = 480 / 48000
        t = adc = 50.0
        for i in range(200):
            adc += period
            t += period
            if i == 100:                                    # DVS restarted: 11.7 s without audio
                adc += 11.7
                t += 11.7
            if i == 150:                                    # and one ordinary skip of 2 cycles
                adc += 2 * period
                t += 2 * period
            tr.cycle(t, adc, 480, 0.03, 0)
        self.assertEqual((tr.skipped, tr.outages), (2, 1))
        self.assertAlmostEqual(tr.outage_s, 11.71, places=1)
        s = tr.summary()
        self.assertIn("The soundcard stopped 1 time(s)", s)
        self.assertIn("Skipped cycles (capture time jumped): 2 cycles in 1 places", s)

    def test_stalls_are_explained(self):
        tr = TimingTrace(48000, 480, 480, seconds=5)
        t = 10.0
        for i in range(300):
            t += 0.01
            if i == 100:                       # held up: 25 ms, but the engine only ran for 1 ms
                with_span = (t + 0.002, t + 0.020)
                tr.span_done("web GET /api/state", *with_span)
                tr.block_done(t, 25.0, [0.1, 0.2, 24.5, 0.0, 0.2], 0, 0, cpu_ms=1.0)
            elif i == 200:                     # working: 12 ms of its own CPU, nothing else running
                tr.block_done(t, 12.0, [0.1, 11.0, 0.7, 0.0, 0.2], 0, 0, cpu_ms=11.5)
            else:
                tr.block_done(t, 1.0, [0.1, 0.2, 0.5, 0.0, 0.2], 0, 0, cpu_ms=0.9)
        r = tr.stall_report()
        self.assertIn("Engine blocks over 8 ms since start: 2", r)
        self.assertIn("Working (the engine thread used the CPU for most of it): 1", r)
        self.assertIn("Waiting (it wasn't running for most of it", r)
        self.assertIn("web GET /api/state 1", r)
        self.assertIn("nothing else recorded during 1", r)
        self.assertIn("core 2", "".join(r.split("biggest stage each time): ")[1:]) or r)
        with tr.span("main: stats line"):
            pass
        self.assertIn("main: stats line", tr.span_names)
        self.assertIn("No engine block took longer", TimingTrace(48000, 480, 480, seconds=1).stall_report())

    def test_the_ring_wraps_without_growing(self):
        tr = TimingTrace(48000, 480, 480, seconds=1)
        n = len(tr.dev)
        for i in range(5 * n):
            tr.cycle(float(i) / 100, float(i) / 100, 480, 1.0, 0)
        self.assertEqual(len(tr.dev), n)
        self.assertIn(f"{n} soundcard cycles", tr.summary())

    def test_garbage_collection_pauses_are_timed(self):
        import gc
        tr = TimingTrace()
        tr.watch_gc()
        try:
            junk = [[i] for i in range(200000)]
            gc.collect()
            del junk
        finally:
            tr.stop()
        self.assertGreater(tr.gc_i, 0)
        self.assertTrue(np.nanmax(tr.gc_rows[:, 2]) >= 0)


if __name__ == "__main__":
    unittest.main()
