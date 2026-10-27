"""Runs the soundcard on a small buffer while Sotto keeps its 10 ms (480-sample) blocks.

Asking a soundcard for 480-sample callbacks when its own I/O buffer is a different size (Dante
Virtual Soundcard and most Core Audio devices use powers of two) makes the audio library cut the
stream up for us. That adds a buffer on each side, and it delivers the 10 ms pieces unevenly: two
back to back, then a gap. Opening the soundcard at its own small buffer and regrouping here keeps
the callbacks even and the soundcard latency low.

Every soundcard callback copies its samples into the current block. When a block is full, the
whole 10 ms of Sotto's engine runs on it (fail-safe core, processing hand-over, phones bus), and
the 480 output samples join a queue the soundcard reads from.

The queue starts with (480 - gcd(io, 480)) samples of silence. That is the least that guarantees
it never runs dry: the soundcard always finds enough samples, whatever the phase between its
buffers and Sotto's blocks. It is the only delay the regrouping adds: 6.7 ms for a 160-sample
buffer, 9.3 ms for 128, 5 ms for 240.

Audio-thread safe: every buffer is allocated up front; process() only copies.
"""
from math import gcd

import numpy as np


class Regrouper:
    def __init__(self, n_in, n_out, block=480, io=160):
        self.block, self.io = int(block), int(io)
        self.prefill = self.block - gcd(self.io, self.block)
        self.inb = np.zeros((self.block, n_in), np.float32)
        self.outb = np.zeros((self.block, n_out), np.float32)
        self.fill = 0
        self.cap = 2 * self.block + max(self.io, self.block)
        self.ring = np.zeros((self.cap, n_out), np.float32)
        self.r = 0
        self.count = self.prefill                        # silence to start with
        self.w = self.prefill % self.cap
        self.underruns = 0
        self.overruns = 0
        self.blocks = 0

    @property
    def added_ms(self):
        return 1000.0 * self.prefill / 48000.0

    def _push(self, x):
        n = len(x)
        if self.count + n > self.cap:                    # can't happen with steady callbacks
            self.overruns += 1
            drop = self.count + n - self.cap
            self.r = (self.r + drop) % self.cap
            self.count -= drop
        first = min(n, self.cap - self.w)
        self.ring[self.w:self.w + first] = x[:first]
        if first < n:
            self.ring[:n - first] = x[first:]
        self.w = (self.w + n) % self.cap
        self.count += n

    def _pop(self, out, n):
        take = min(n, self.count)
        first = min(take, self.cap - self.r)
        out[:first] = self.ring[self.r:self.r + first]
        if first < take:
            out[first:take] = self.ring[:take - first]
        if take < n:                                     # ran dry: silence for the rest
            out[take:n] = 0.0
            self.underruns += 1
        self.r = (self.r + take) % self.cap
        self.count -= take

    def process(self, indata, outdata, frames, run_block):
        """indata (frames, n_in), outdata (frames, n_out) from the soundcard. run_block(inb, outb)
        is Sotto's 10 ms engine; it is called once for every full block."""
        i = 0
        while i < frames:
            take = min(self.block - self.fill, frames - i)
            self.inb[self.fill:self.fill + take] = indata[i:i + take]
            self.fill += take
            i += take
            if self.fill == self.block:
                self.outb.fill(0.0)
                run_block(self.inb, self.outb)
                self._push(self.outb)
                self.fill = 0
                self.blocks += 1
        self._pop(outdata, frames)
