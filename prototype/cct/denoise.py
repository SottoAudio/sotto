"""Noise suppression stages: a classical spectral baseline and DeepFilterNet3 (ONNX).

DeepFilterNet (MIT / Apache-2.0, Schroeter et al.) is run from its published ONNX export
with onnxruntime. Feature extraction and synthesis follow libDF (Vorbis window, ERB band
energies with exponential mean normalisation, unit-normalised complex features, ERB mask
on all bins plus a 5-tap deep filter on the lowest 96 bins).
"""
import configparser
import math
import os
import numpy as np


# ---------------------------------------------------------------- classical baseline
def spectral_denoise(x, fs=48000, nfft=1024, hop=256, floor_db=-15.0, alpha=0.98, win_s=1.5):
    """Decision-directed Wiener filter; noise = bias-corrected trailing minimum of the
    smoothed power spectrum over `win_s` seconds (minimum statistics, simplified)."""
    from scipy.signal import stft, istft
    from scipy.ndimage import minimum_filter1d
    _, _, X = stft(x, fs, nperseg=nfft, noverlap=nfft - hop)
    P = np.abs(X) ** 2
    Ps = np.empty_like(P)
    acc = P[:, 0]
    for t in range(P.shape[1]):
        acc = 0.85 * acc + 0.15 * P[:, t]
        Ps[:, t] = acc
    W = max(3, int(win_s * fs / hop))
    noise = 1.8 * minimum_filter1d(Ps, size=W, axis=1, origin=(W - 1) // 2) + 1e-14
    G = np.ones_like(P)
    prev = np.zeros(P.shape[0])
    gmin = 10 ** (floor_db / 20)
    for t in range(P.shape[1]):
        post = P[:, t] / noise[:, t]
        prio = alpha * prev / noise[:, t] + (1 - alpha) * np.maximum(post - 1, 0)
        g = np.maximum(prio / (1 + prio), gmin)
        G[:, t] = g
        prev = (g ** 2) * P[:, t]
    _, y = istft(X * G, fs, nperseg=nfft, noverlap=nfft - hop)
    return y[: len(x)]


# ---------------------------------------------------------------- DeepFilterNet3 ONNX
def _freq2erb(f):
    return 9.265 * math.log1p(f / (24.7 * 9.265))


def _erb2freq(n):
    return 24.7 * 9.265 * (math.exp(n / 9.265) - 1)


def erb_widths(sr, fft_size, nb_bands, min_nb_freqs):
    nyq = sr // 2
    fw = sr / fft_size
    lo, hi = _freq2erb(0.0), _freq2erb(nyq)
    step = (hi - lo) / nb_bands
    erb = [0] * nb_bands
    prev, over = 0, 0
    for i in range(1, nb_bands + 1):
        f = _erb2freq(lo + i * step)
        fb = int(round(f / fw))
        nb = fb - prev - over
        if nb < min_nb_freqs:
            over = min_nb_freqs - nb
            nb = min_nb_freqs
        else:
            over = 0
        erb[i - 1] = nb
        prev = fb
    erb[-1] += 1
    too = sum(erb) - (fft_size // 2 + 1)
    if too > 0:
        erb[-1] -= too
    return np.array(erb)


class DeepFilterNetONNX:
    def __init__(self, model_dir):
        import onnxruntime as ort
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(model_dir, "config.ini"))
        d = cfg["df"]
        self.sr = int(d["sr"])
        self.n_fft = int(d["fft_size"])
        self.hop = int(d["hop_size"])
        self.nb_erb = int(d["nb_erb"])
        self.nb_df = int(d["nb_df"])
        self.df_order = int(d["df_order"])
        self.df_lookahead = int(d["df_lookahead"])
        self.conv_lookahead = int(cfg["deepfilternet"]["conv_lookahead"])
        tau = float(d["norm_tau"])
        a_ = math.exp(-(self.hop / self.sr) / tau)
        prec, a = 3, 1.0
        while a >= 1.0:
            a = round(a_, prec)
            prec += 1
        self.alpha = a
        self.erb = erb_widths(self.sr, self.n_fft, self.nb_erb, int(d["min_nb_erb_freqs"]))
        n = self.n_fft
        i = np.arange(n)
        s = np.sin(0.5 * np.pi * (i + 0.5) / (n // 2))
        self.window = np.sin(0.5 * np.pi * s * s)
        self.wnorm = 1.0 / (n ** 2 / (2 * self.hop))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.enc = ort.InferenceSession(os.path.join(model_dir, "enc.onnx"), so)
        self.erb_dec = ort.InferenceSession(os.path.join(model_dir, "erb_dec.onnx"), so)
        self.df_dec = ort.InferenceSession(os.path.join(model_dir, "df_dec.onnx"), so)

    @property
    def latency_ms(self):
        # STFT loop delay (window - hop) + one hop of buffering + look-ahead frames.
        la = max(self.df_lookahead, self.conv_lookahead)
        return 1000.0 * (self.n_fft + la * self.hop) / self.sr

    def analysis(self, x):
        n, h = self.n_fft, self.hop
        T = len(x) // h
        mem = np.zeros(n - h)
        spec = np.zeros((T, n // 2 + 1), dtype=np.complex128)
        for t in range(T):
            frame = np.concatenate([mem, x[t * h:(t + 1) * h]])
            mem = frame[h:]
            spec[t] = np.fft.rfft(frame * self.window) * self.wnorm
        return spec

    def synthesis(self, spec):
        n, h = self.n_fft, self.hop
        T = spec.shape[0]
        out = np.zeros(T * h)
        mem = np.zeros(n - h)
        for t in range(T):
            fr = np.fft.irfft(spec[t], n) * n * self.window
            out[t * h:(t + 1) * h] = fr[:h] + mem[:h]
            mem = np.concatenate([mem[h:], np.zeros(h)]) + fr[h:]
        return out

    def features(self, spec):
        T = spec.shape[0]
        # ERB features
        p = np.abs(spec) ** 2
        edges = np.concatenate([[0], np.cumsum(self.erb)])
        e = np.stack([p[:, edges[b]:edges[b + 1]].mean(axis=1) for b in range(self.nb_erb)], axis=1)
        e = 10 * np.log10(e + 1e-10)
        st = np.linspace(-60.0, -90.0, self.nb_erb)
        feat_erb = np.zeros_like(e)
        for t in range(T):
            st = e[t] * (1 - self.alpha) + st * self.alpha
            feat_erb[t] = (e[t] - st) / 40.0
        # complex features
        c = spec[:, : self.nb_df]
        su = np.linspace(0.001, 0.0001, self.nb_df)
        feat_c = np.zeros_like(c)
        for t in range(T):
            su = np.abs(c[t]) * (1 - self.alpha) + su * self.alpha
            feat_c[t] = c[t] / np.sqrt(su)
        fe = feat_erb[None, None].astype(np.float32)
        fs = np.stack([feat_c.real, feat_c.imag], axis=0)[None].astype(np.float32)
        return fe, fs

    def enhance(self, x):
        """Offline enhancement of a 48 kHz mono signal. Output is time-aligned to the input
        (the STFT delay is removed for comparison; the real-time latency is `latency_ms`)."""
        n, h = self.n_fft, self.hop
        orig = len(x)
        xp = np.concatenate([x, np.zeros(n)])
        spec = self.analysis(xp)
        fe, fs = self.features(spec)
        e0, e1, e2, e3, emb, c0, _lsnr = self.enc.run(None, {"feat_erb": fe, "feat_spec": fs})
        m = self.erb_dec.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})[0]
        coefs = self.df_dec.run(None, {"emb": emb, "c0": c0})[0]
        m = m[0, 0]                                   # (T, nb_erb)
        gains = np.repeat(m, self.erb, axis=1)        # (T, F)
        spec_m = spec * gains
        T = spec.shape[0]
        co = coefs[0].reshape(T, self.nb_df, self.df_order, 2)
        co = co[..., 0] + 1j * co[..., 1]             # (T, F, O)
        pad_past = self.df_order - 1 - self.df_lookahead
        lo = spec[:, : self.nb_df]
        padded = np.concatenate([np.zeros((pad_past, self.nb_df)), lo,
                                 np.zeros((self.df_lookahead, self.nb_df))], axis=0)
        df_out = np.zeros((T, self.nb_df), dtype=np.complex128)
        for o in range(self.df_order):
            df_out += padded[o:o + T] * co[:, :, o]
        spec_e = spec_m.copy()
        spec_e[:, : self.nb_df] = df_out
        y = self.synthesis(spec_e)
        d = n - h
        return y[d:d + orig]
