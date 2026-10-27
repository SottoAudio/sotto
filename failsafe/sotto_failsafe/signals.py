"""Test tones and narrow-band tone detection for HEARTBEAT, PROBE and the path marker."""
import math

import numpy as np


def dbfs(x):
    return 20.0 * math.log10(max(float(x), 1e-12))


class Tone:
    """Phase-continuous sine, block by block."""

    def __init__(self, freq_hz, level_dbfs, fs=48000):
        self.freq = float(freq_hz)
        self.amp = 10 ** (level_dbfs / 20.0)
        self.fs = fs
        self.phase = 0.0

    def block(self, n):
        w = 2 * math.pi * self.freq / self.fs
        out = self.amp * np.sin(self.phase + w * np.arange(n))
        self.phase = (self.phase + w * n) % (2 * math.pi)
        return out


def goertzel_power(x, freq, fs):
    """Power of `x` at `freq` (normalised so a full-scale sine reads about 0.25)."""
    n = len(x)
    if n == 0:
        return 0.0
    w = 2 * math.pi * freq / fs
    X = np.dot(np.asarray(x, dtype=np.float64), np.exp(-1j * w * np.arange(n)))
    return float(abs(X) ** 2) / (n * n)


def tone_present(x, freq, fs, min_dbfs=-50.0, dominance_db=12.0):
    """True if a steady tone at `freq` dominates `x`: its level is above `min_dbfs` and
    `dominance_db` above the signal power around it (so speech or noise doesn't count)."""
    if len(x) == 0:
        return False
    p = goertzel_power(np.asarray(x, dtype=np.float64), freq, fs)
    tone_rms = math.sqrt(2 * p)                      # sine of amplitude A: p ~ A^2/4, rms A/sqrt(2)
    if dbfs(tone_rms) < min_dbfs:
        return False
    total = float(np.mean(np.square(x))) + 1e-20
    rest = max(total - tone_rms ** 2, 1e-20)
    return 10 * math.log10(tone_rms ** 2 / rest) >= dominance_db


class ToneWatch:
    """Tracks whether a tone keeps arriving. `missing_s` is how long it has been absent."""

    def __init__(self, freq_hz, fs=48000, timeout_s=0.2, min_dbfs=-50.0, dominance_db=12.0):
        self.freq = freq_hz
        self.fs = fs
        self.timeout = timeout_s
        self.min_dbfs = min_dbfs
        self.dominance = dominance_db
        self.last_seen = None        # audio time (s) the tone was last detected
        self.ever_seen = False

    def update(self, x, t_now):
        if tone_present(x, self.freq, self.fs, self.min_dbfs, self.dominance):
            self.last_seen = t_now
            self.ever_seen = True

    def missing_s(self, t_now):
        if self.last_seen is None:
            return float("inf")
        return max(0.0, t_now - self.last_seen)

    def ok(self, t_now):
        return self.missing_s(t_now) <= self.timeout
