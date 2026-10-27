"""A simulated Riedel frame and audio driver, for trying the bench tool without hardware and
for the automated tests.

What the simulated frame does, every block:
  * talkers: talker 1 is a latched key with a noise source near the mic (so the dry path is
    audible); the others are quieter;
  * PROBE: whatever the app sends on PROBE out comes straight back on every PROBE input
    (a fixed crosspoint), after the network delay;
  * HEARTBEAT: a VOX watches the app's HEARTBEAT output; CLEAN_OK stays on until the VOX has
    been off for the hold time (400 ms by default), then logic takes `logic_ms` to act;
  * LISTEN: a listener hears talker 1's clean return from the app while CLEAN_OK is on,
    otherwise talker 1's dry mic straight through the matrix.

  python -m sotto_failsafe --simulate        # console at http://127.0.0.1:8766
"""
import threading
import time

import numpy as np


class SimRiedel:
    def __init__(self, cfg, fs=48000, net_ms=4.0, vox_hold_ms=400.0, vox_on_dbfs=-40.0,
                 logic_ms=20.0, seed=0):
        self.cfg = cfg
        self.fs = fs
        self.delay = int(net_ms * fs / 1000)
        self.hold = vox_hold_ms / 1000.0
        self.vox_on = vox_on_dbfs
        self.logic = logic_ms / 1000.0
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.last_vox = -1e9
        self.clean_ok = False
        self.clean_ok_since = None
        self.pending = None            # (time it takes effect, new state)
        n_out = max(cfg.talker_out + [cfg.probe_out, cfg.heartbeat_out] + list(cfg.monitor_out)) + 1
        self.out_hist = np.zeros((self.delay + 4096, n_out))
        self.n_out = n_out
        self.lp = 0.0

    def _mic(self, frames, gain):
        w = self.rng.standard_normal(frames) * gain
        y = np.empty(frames)
        a = 0.97
        for i in range(frames):               # gentle low-pass: noise that sounds like a room
            self.lp = a * self.lp + (1 - a) * w[i]
            y[i] = self.lp * 6.0
        return y

    def make_input(self, frames, n_in):
        """Builds the app's input block from the frame's point of view."""
        cfg = self.cfg
        x = np.zeros((frames, n_in))
        mics = []
        for i, ch in enumerate(cfg.talker_in):
            m = self._mic(frames, 0.05 if i == 0 else 0.01)
            mics.append(m)
            x[:, ch] = m
        # Delayed app outputs (what the frame received a few ms ago)
        past = self.out_hist[-self.delay - frames:-self.delay] if self.delay else self.out_hist[-frames:]
        for ch in cfg.probe_in:
            x[:, ch] = past[:, cfg.probe_out]
        # VOX on the heartbeat, hold, then logic
        hb = past[:, cfg.heartbeat_out]
        level = 20 * np.log10(np.sqrt(np.mean(hb ** 2)) + 1e-12)
        if level > self.vox_on:
            self.last_vox = self.t
        vox = (self.t - self.last_vox) <= self.hold
        if vox != self.clean_ok and (self.pending is None or self.pending[1] != vox):
            self.pending = (self.t + self.logic, vox)
        if self.pending and self.t >= self.pending[0]:
            self.clean_ok = self.pending[1]
            self.pending = None
        if cfg.listen_in is not None:
            clean_ret = past[:, cfg.talker_out[0]]
            x[:, cfg.listen_in] = clean_ret if self.clean_ok else mics[0]
        self.t += frames / self.fs
        return x.astype(np.float32)

    def take_output(self, outdata):
        self.out_hist = np.concatenate([self.out_hist[len(outdata):], np.asarray(outdata, dtype=np.float64)])


class SimStream:
    """Stands in for sounddevice.Stream: drives the callback at real time (or faster)."""

    def __init__(self, sim, n_in, n_out, blocksize, callback, speed=1.0):
        self.sim, self.n_in, self.n_out = sim, n_in, n_out
        self.bs, self.cb, self.speed = blocksize, callback, speed
        self.latency = (0.004, 0.004)
        self._stop = threading.Event()

    def __enter__(self):
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.th.join(timeout=2)

    def _run(self):
        t0 = time.perf_counter()
        b = 0
        while not self._stop.is_set():
            x = self.sim.make_input(self.bs, self.n_in)
            out = np.zeros((self.bs, self.n_out), np.float32)
            self.cb(x, out, self.bs, None, None)
            self.sim.take_output(out)
            b += 1
            if self.speed:
                wait = t0 + b * self.bs / self.sim.fs / self.speed - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
