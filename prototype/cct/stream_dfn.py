"""DeepFilterNet3 low-latency, one 10 ms frame at a time.

Uses the models from tools/make_streaming.py (GRU states and causal-conv context are explicit
inputs/outputs). Feature extraction, the ERB mask and the deep filter follow the offline
DeepFilterNetONNX class exactly, so the two produce the same audio.
"""
import json
import os
import numpy as np



class StreamSTFT:
    """Vorbis-window STFT matching libDF: window n_fft, hop, analysis and overlap-add synthesis."""

    def __init__(self, n_fft=960, hop=480):
        self.n, self.h = n_fft, hop
        i = np.arange(n_fft)
        s = np.sin(0.5 * np.pi * (i + 0.5) / (n_fft // 2))
        self.window = np.sin(0.5 * np.pi * s * s)
        self.wnorm = 1.0 / (n_fft ** 2 / (2 * hop))
        self.amem = np.zeros(n_fft - hop)
        self.smem = np.zeros(n_fft - hop)

    def analyse(self, x):
        frame = np.concatenate([self.amem, x])
        self.amem = frame[self.h:]
        return np.fft.rfft(frame * self.window) * self.wnorm

    def synthesise(self, X):
        h = self.h
        fr = np.fft.irfft(X, self.n) * self.n * self.window
        out = fr[:h] + self.smem[:h]
        self.smem = np.concatenate([self.smem[h:], np.zeros(h)]) + fr[h:]
        return out


class StreamingDFN:
    def __init__(self, model_dir, threads=1, share_from=None):
        """share_from: another StreamingDFN whose (stateless) ONNX sessions are reused, so
        several talkers can run without loading the model several times."""
        import onnxruntime as ort
        self._init_cfg(model_dir)
        with open(os.path.join(model_dir, "states.json")) as f:
            self.meta = json.load(f)
        if share_from is not None:
            self.sess = share_from.sess
        else:
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            self.sess = {m: ort.InferenceSession(os.path.join(model_dir, f"{m}.onnx"), so)
                         for m in ("enc", "erb_dec", "df_dec")}
        self._out_names = {m: [o.name for o in s.get_outputs()] for m, s in self.sess.items()}
        self.reset()

    def _init_cfg(self, model_dir):
        import configparser
        import math
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
        assert self.df_lookahead == 0 and int(cfg["deepfilternet"]["conv_lookahead"]) == 0, \
            "streaming wrapper supports the zero look-ahead model only"
        tau = float(d["norm_tau"])
        a_ = math.exp(-(self.hop / self.sr) / tau)
        prec, a = 3, 1.0
        while a >= 1.0:
            a = round(a_, prec)
            prec += 1
        self.alpha = a
        from .denoise import erb_widths
        self.erb = erb_widths(self.sr, self.n_fft, self.nb_erb, int(d["min_nb_erb_freqs"]))
        self.edges = np.concatenate([[0], np.cumsum(self.erb)])

    @property
    def latency_ms(self):
        return 1000.0 * self.n_fft / self.sr

    def reset(self):
        self.states = {m: {s["in"]: np.zeros(s["shape"], np.float32) for s in self.meta[m]}
                       for m in self.meta}
        self.erb_state = np.linspace(-60.0, -90.0, self.nb_erb)
        self.unit_state = np.linspace(0.001, 0.0001, self.nb_df)
        self.hist = np.zeros((self.df_order, self.nb_df), dtype=np.complex128)  # oldest first
        self.lsnr = 0.0

    def _run(self, m, feeds):
        sess = self.sess[m]
        st = self.states[m]
        outs = sess.run(None, {**feeds, **st})
        res = dict(zip(self._out_names[m], outs))
        for s in self.meta[m]:
            st[s["in"]] = res[s["out"]]
        return res

    def process_spec(self, X):
        """X: one complex spectrum frame (n_fft//2+1,) from StreamSTFT. Returns enhanced frame."""
        a = self.alpha
        p = np.abs(X) ** 2
        e = np.add.reduceat(p, self.edges[:-1]) / self.erb
        e = 10 * np.log10(e + 1e-10)
        self.erb_state = e * (1 - a) + self.erb_state * a
        fe = ((e - self.erb_state) / 40.0).astype(np.float32)[None, None, None, :]
        c = X[: self.nb_df]
        self.unit_state = np.abs(c) * (1 - a) + self.unit_state * a
        fc = c / np.sqrt(self.unit_state)
        fsp = np.stack([fc.real, fc.imag])[None, :, None, :].astype(np.float32)

        enc = self._run("enc", {"feat_erb": fe, "feat_spec": fsp})
        m = self._run("erb_dec", {k: enc[k] for k in ("emb", "e3", "e2", "e1", "e0")})["m"]
        dfo = self._run("df_dec", {"emb": enc["emb"], "c0": enc["c0"]})
        coefs = dfo["coefs"]
        self.lsnr = float(enc["lsnr"].ravel()[0])

        gains = np.repeat(m[0, 0, 0], self.erb)
        Y = X * gains
        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = X[: self.nb_df]
        co = coefs[0, 0].reshape(self.nb_df, self.df_order, 2)
        co = co[..., 0] + 1j * co[..., 1]
        Y[: self.nb_df] = np.sum(self.hist.T * co, axis=1)
        return Y
