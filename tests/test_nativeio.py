"""The native soundcard side: rings, safety margin, dry fallback with crossfade, HEARTBEAT latch.

These drive the C library's real-time function by hand (no soundcard needed), with the engine on
its own thread exactly as the app runs it.
"""
import os
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "failsafe"))

from sotto_failsafe import nativeio  # noqa: E402

N_IN, N_OUT, B, IO = 4, 4, 480, 160
TALK_IN, TALK_OUT, HB = [0], [1], 3


def ramp_input(start, frames):
    """Input channel 0 = a sample counter (so the output shows exactly which input it came from),
    channel 2 = a 300 Hz tone for the crossfade checks."""
    k = np.arange(start, start + frames)
    x = np.zeros((frames, N_IN), np.float32)
    x[:, 0] = (k % 10000000) / 10000000.0 * 100
    x[:, 2] = 0.5 * np.sin(2 * np.pi * 300 * k / 48000)
    return x


@unittest.skipUnless(nativeio.library(), "native audio library not built (cmake -S engine/audio-io -B build/audio-io)")
class NativeIOTests(unittest.TestCase):
    def make(self, margin=480):
        io = nativeio.NativeIO(N_IN, N_OUT, B, IO, margin=margin)
        io.set_fallback(TALK_IN, TALK_OUT, HB)
        self.addCleanup(io.destroy)
        return io

    def engine(self, io, stall=None):
        """The engine thread: output 1 = input 0 + 1.0 (the 'processed' talker), output 3 = HEARTBEAT (1.0).
        stall: (block number, seconds) to stop for once."""
        stop = threading.Event()
        state = {"blocks": 0}

        def run():
            inb = np.zeros((B, N_IN), np.float32)
            outb = np.zeros((B, N_OUT), np.float32)
            while not stop.is_set():
                if io.read(inb, 50) != 1:
                    continue
                if stall and state["blocks"] == stall[0]:
                    time.sleep(stall[1])
                outb[:] = 0
                outb[:, 1] = inb[:, 0] + 1.0               # 'processed': a full 1.0 away from dry
                outb[:, 3] = 1.0
                io.write(outb)
                state["blocks"] += 1

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return stop, t, state

    def soundcard(self, io, cycles, start=0, pace=True):
        """Calls the real-time side every IO samples, in real time (like Core Audio)."""
        outs = []
        t0 = time.perf_counter()
        for c in range(cycles):
            if pace:
                wait = t0 + c * IO / 48000 - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            x = ramp_input(start + c * IO, IO)
            y = np.zeros((IO, N_OUT), np.float32)
            io.cycle(x, y, IO, adc=(start + c * IO) / 48000)
            outs.append(y)
        return np.concatenate(outs)

    def test_output_trails_input_by_exactly_the_safety_margin(self):
        # A wide margin (100 ms): this checks the logic, not how evenly a shared test machine
        # schedules Python threads.
        io = self.make(margin=4800)
        stop, t, _ = self.engine(io)
        try:
            y = self.soundcard(io, 600)                       # 2 s
        finally:
            stop.set(); t.join()
        st = io.stats()
        self.assertEqual(st["underrun_cycles"], 0)
        self.assertFalse(io.heartbeat_blocked())
        # output 1 at sample n = input 0 at sample n - prefill (480 - 160 + 4800 = 5120)
        self.assertEqual(io.prefill, 5120)
        n = np.arange(len(y))
        late = n >= 8000
        expect = ((n - io.prefill) % 10000000) / 10000000.0 * 100 + 1.0
        self.assertLess(float(np.max(np.abs(y[late, 1] - expect[late]))), 1e-6)
        self.assertTrue(np.all(y[late, 3] == 1.0))            # HEARTBEAT through

    def test_late_engine_plays_dry_with_crossfade_and_latches_heartbeat(self):
        io = self.make(margin=480)
        stop, t, _ = self.engine(io, stall=(100, 0.06))      # block 100 (1 s in): 60 ms late
        try:
            y = self.soundcard(io, 900)                       # 3 s
        finally:
            stop.set(); t.join()
        st = io.stats()
        self.assertGreater(st["underrun_cycles"], 0)
        self.assertGreater(st["fallback_frames"], 0)
        self.assertTrue(io.heartbeat_blocked())
        hb = y[:, 3]
        first_off = 2000 + int(np.argmax(hb[2000:] == 0.0)) if np.any(hb[2000:] == 0) else -1
        self.assertGreater(first_off, 0)
        self.assertTrue(np.all(hb[first_off + IO:] == 0.0), "HEARTBEAT must stay off once latched")
        # the talker output never jumps: dry and processed are crossfaded (input 0 is a slow ramp,
        # so any step bigger than a few ramp increments would be a click)
        jumps = np.abs(np.diff(y[2000:, 1]))
        self.assertGreater(float(np.min(y[2000:, 1] - (np.arange(2000, len(y)) % 10000000) / 1e5)), -0.01)
        self.assertLess(float(jumps.max()), 1.0 / 100, "a hard switch would jump by 1.0; a 120-sample fade by ~0.008")
        self.assertTrue(np.any(y[2000:, 1] < 0.5), "dry audio was played while the engine was late")
        io.clear_heartbeat()                                  # a manual re-arm
        self.assertFalse(io.heartbeat_blocked())

    def test_a_long_stall_skips_late_blocks_and_keeps_the_margin(self):
        io = self.make(margin=4800)
        stop, t, _ = self.engine(io, stall=(50, 0.5))        # 500 ms: far beyond the 100 ms margin
        try:
            y = self.soundcard(io, 900)
        finally:
            stop.set(); t.join()
        st = io.stats()
        self.assertGreater(st["dropped_blocks"], 0)
        n = np.arange(len(y))
        tail = n >= len(y) - 48000 // 2                      # the last half second: back to normal
        expect = ((n - io.prefill) % 10000000) / 10000000.0 * 100 + 1.0
        self.assertLess(float(np.max(np.abs(y[tail, 1] - expect[tail]))), 1e-6, "same delay as before the stall")

    def test_timing_record(self):
        io = self.make()
        stop, t, _ = self.engine(io)
        try:
            self.soundcard(io, 300)
        finally:
            stop.set(); t.join()
        rows = io.new_trace_rows()
        self.assertEqual(len(rows), 300)
        self.assertTrue(np.all(rows[:, 2] == IO))
        self.assertAlmostEqual(float(rows[1, 1] - rows[0, 1]), IO / 48000, places=6)
        self.assertEqual(len(io.new_trace_rows()), 0)          # only new rows next time


if __name__ == "__main__":
    unittest.main()
