"""Partitioned-block frequency-domain adaptive filter (PBFDAF / MDF) echo canceller.

Causal, block-based, the same structure the real-time box would run. Supports several
references (PA L, PA R, sub...) each with its own filter. A variable step size slows
adaptation while the talker is speaking (double talk), so the filter learns the PA path
from the gaps between words.
"""
import numpy as np


class PBFDAF:
    def __init__(self, n_ref, fs=48000, block=256, tail_s=0.4, mu=0.5, reg=1e-8,
                 constrain_per_block=4, smooth=0.9):
        self.N = block
        self.M = 2 * block
        self.K = block + 1
        self.P = int(np.ceil(tail_s * fs / block))
        self.n_ref = n_ref
        self.mu = mu
        self.reg = reg
        self.cpb = constrain_per_block
        self.smooth = smooth
        self.W = np.zeros((n_ref, self.P, self.K), dtype=np.complex128)
        self.X = np.zeros((n_ref, self.P, self.K), dtype=np.complex128)  # circular history
        self.head = 0
        self.xbuf = np.zeros((n_ref, self.M))
        self.Pxx = np.full(self.K, 1e-6)
        self.See = np.full(self.K, 1e-6)
        self.Syy = np.full(self.K, 1e-6)
        self._cidx = 0

    def _ordered_X(self):
        idx = (self.head - np.arange(self.P)) % self.P
        return self.X[:, idx, :]

    def process_block(self, x, d, adapt=True):
        """x: (n_ref, N) reference block; d: (N,) mic block. Returns (e, y) time blocks.
        adapt=False runs the filter without learning (used while the mic is gated off, so the
        learnt PA path is kept between calls instead of being unlearnt against silence)."""
        N, M = self.N, self.M
        self.xbuf = np.concatenate([self.xbuf[:, N:], x], axis=1)
        Xn = np.fft.rfft(self.xbuf, axis=1)
        self.head = (self.head + 1) % self.P
        self.X[:, self.head, :] = Xn
        Xo = self._ordered_X()
        Y = np.sum(self.W * Xo, axis=(0, 1))
        y = np.fft.irfft(Y, M)[N:]
        e = d - y
        E = np.fft.rfft(np.concatenate([np.zeros(N), e]))
        Yf = np.fft.rfft(np.concatenate([np.zeros(N), y]))

        if not adapt:
            return e, y
        a = self.smooth
        self.Pxx = a * self.Pxx + (1 - a) * np.sum(np.abs(Xn) ** 2, axis=0)
        self.See = a * self.See + (1 - a) * np.abs(E) ** 2
        self.Syy = a * self.Syy + (1 - a) * np.abs(Yf) ** 2
        # Variable step: large when the error is mostly echo, small during talk.
        vss = np.clip(self.Syy / (self.See + 1e-12), 0.05, 1.0)
        # Warm start: before the filter has learnt anything, Syy ~ 0, so allow adaptation.
        if np.sum(self.Syy) < 1e-9:
            vss = np.ones_like(vss)
        # NLMS normalisation per bin by the energy held in the whole filter window.
        psum = np.sum(np.abs(Xo) ** 2, axis=(0, 1))
        psum = np.maximum(psum, 1e-3 * np.mean(psum) + 1e-12)
        step = self.mu * vss / psum
        self.W += step[None, None, :] * np.conj(Xo) * E[None, None, :]

        # Gradient constraint, a few partitions per block (round robin).
        for _ in range(self.cpb):
            p = self._cidx
            w = np.fft.irfft(self.W[:, p, :], M, axis=1)
            w[:, N:] = 0.0
            self.W[:, p, :] = np.fft.rfft(w, axis=1)
            self._cidx = (self._cidx + 1) % self.P
        return e, y

    def impulse_response(self):
        """Learnt filter in the time domain, (n_ref, P*N), oldest partition last."""
        N, M = self.N, self.M
        w = np.fft.irfft(self.W, M, axis=2)[:, :, :N]
        return w.reshape(self.n_ref, -1)

    def run(self, refs, mic):
        """refs: (n_ref, T); mic: (T,). Returns error (clean-ish) and echo estimate."""
        T = len(mic)
        nb = T // self.N
        e_out = np.zeros(nb * self.N)
        y_out = np.zeros(nb * self.N)
        for b in range(nb):
            s = slice(b * self.N, (b + 1) * self.N)
            e, y = self.process_block(refs[:, s], mic[s])
            e_out[s] = e
            y_out[s] = y
        return e_out, y_out


def residual_suppress(e, y, fs=48000, nfft=1024, hop=256, beta=1.5, floor_db=-18.0, smooth=0.7):
    """Spectral residual echo suppressor driven by the echo estimate y."""
    from scipy.signal import stft, istft
    _, _, E = stft(e, fs, nperseg=nfft, noverlap=nfft - hop)
    _, _, Y = stft(y, fs, nperseg=nfft, noverlap=nfft - hop)
    floor = 10 ** (floor_db / 20)
    See = np.abs(E) ** 2
    Syy = np.abs(Y) ** 2
    G = np.maximum(1.0 - beta * Syy / (See + 1e-12), floor ** 2)
    G = np.sqrt(G)
    for t in range(1, G.shape[1]):
        G[:, t] = smooth * G[:, t - 1] + (1 - smooth) * G[:, t]
    _, out = istft(E * G, fs, nperseg=nfft, noverlap=nfft - hop)
    return out[: len(e)]
