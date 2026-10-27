"""Small soundcard buffers regrouped into Sotto's 10 ms blocks: nothing lost or repeated, a fixed
delay of exactly the prefill, and the engine always sees whole 480-sample blocks."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "failsafe"))

from sotto_failsafe.regroup import Regrouper  # noqa: E402


class RegroupTests(unittest.TestCase):
    def run_through(self, io, total=48000):
        rg = Regrouper(2, 2, 480, io)
        x = np.random.default_rng(1).standard_normal((total, 2)).astype(np.float32)
        y = np.zeros_like(x)
        sizes = []

        def engine(inb, outb):                           # a pass-through engine
            sizes.append(len(inb))
            outb[:] = inb

        for s in range(0, total - io + 1, io):
            rg.process(x[s:s + io], y[s:s + io], io, engine)
        return rg, x, y, sizes

    def test_every_buffer_size_is_lossless_with_a_fixed_delay(self):
        for io in (64, 96, 128, 160, 240, 256, 480, 512):
            rg, x, y, sizes = self.run_through(io)
            d = rg.prefill
            self.assertEqual(set(sizes), {480}, io)
            self.assertEqual(rg.underruns, 0, io)
            self.assertEqual(rg.overruns, 0, io)
            n = len(x) - d - 2 * io
            np.testing.assert_array_equal(y[d:d + n], x[:n], err_msg=f"io={io}")

    def test_added_delay(self):
        self.assertEqual(Regrouper(1, 1, 480, 160).prefill, 320)        # 6.7 ms
        self.assertEqual(Regrouper(1, 1, 480, 128).prefill, 448)        # 9.3 ms
        self.assertEqual(Regrouper(1, 1, 480, 240).prefill, 240)        # 5 ms
        self.assertEqual(Regrouper(1, 1, 480, 480).prefill, 0)


if __name__ == "__main__":
    unittest.main()
