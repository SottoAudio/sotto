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


class BatchedDFN(StreamingDFN):
    """All talkers through the speech model in one ONNX call per 10 ms frame (the batched model
    from tools/make_batchable.py). Same maths as StreamingDFN, one row per talker; much cheaper
    than one call per talker."""

    def __init__(self, model_dir, n, threads=1):
        self.n = int(n)
        super().__init__(model_dir, threads=threads)

    def reset(self):
        n = self.n

        def shape(s):
            sh = list(s["shape"])
            sh[1 if s["kind"] == "gru" else 0] = n          # GRU states carry the batch on axis 1
            return sh
        self.states = {m: {s["in"]: np.zeros(shape(s), np.float32) for s in self.meta[m]} for m in self.meta}
        self.erb_state = np.tile(np.linspace(-60.0, -90.0, self.nb_erb), (n, 1))
        self.unit_state = np.tile(np.linspace(0.001, 0.0001, self.nb_df), (n, 1))
        self.hist = np.zeros((n, self.df_order, self.nb_df), dtype=np.complex128)
        self.lsnr = np.zeros(n)

    def reset_row(self, i):
        """Clears one talker's model state (after that talker's input is reset)."""
        for m in self.meta:
            for s in self.meta[m]:
                st = self.states[m][s["in"]]
                if s["kind"] == "gru":
                    st[:, i] = 0
                else:
                    st[i] = 0
        self.erb_state[i] = np.linspace(-60.0, -90.0, self.nb_erb)
        self.unit_state[i] = np.linspace(0.001, 0.0001, self.nb_df)
        self.hist[i] = 0

    def process_specs(self, X, rows=None):
        """X: (n, n_fft//2+1) complex, one spectrum frame per talker. Returns enhanced frames.
        rows: optional list of talker rows to run (the others keep their state and get X back),
        so talkers that don't need the model don't cost anything."""
        if rows is not None and len(rows) < self.n:
            rows = list(rows)
            Y = X.copy()
            if not rows:
                return Y
            full = (self.states, self.erb_state, self.unit_state, self.hist, self.lsnr, self.n)
            idx = np.array(rows)
            sub_states = {m: {} for m in self.meta}
            for m in self.meta:
                for s_ in self.meta[m]:
                    st = self.states[m][s_["in"]]
                    sub_states[m][s_["in"]] = st[:, idx] if s_["kind"] == "gru" else st[idx]
            self.states, self.n = sub_states, len(rows)
            self.erb_state, self.unit_state, self.hist = full[1][idx], full[2][idx], full[3][idx]
            try:
                Ysub = self.process_specs(X[idx])
            finally:
                sub = (self.states, self.erb_state, self.unit_state, self.hist, self.lsnr)
                self.states, self.erb_state, self.unit_state, self.hist, self.lsnr, self.n = full
            for m in self.meta:
                for s_ in self.meta[m]:
                    k = s_["in"]
                    if s_["kind"] == "gru":
                        self.states[m][k][:, idx] = sub[0][m][k]
                    else:
                        self.states[m][k][idx] = sub[0][m][k]
            self.erb_state[idx], self.unit_state[idx], self.hist[idx] = sub[1], sub[2], sub[3]
            self.lsnr = np.array(self.lsnr, dtype=float)
            self.lsnr[idx] = sub[4]
            Y[idx] = Ysub
            return Y
        a = self.alpha
        p = np.abs(X) ** 2
        e = np.add.reduceat(p, self.edges[:-1], axis=1) / self.erb
        e = 10 * np.log10(e + 1e-10)
        self.erb_state = e * (1 - a) + self.erb_state * a
        fe = ((e - self.erb_state) / 40.0).astype(np.float32)[:, None, None, :]
        c = X[:, : self.nb_df]
        self.unit_state = np.abs(c) * (1 - a) + self.unit_state * a
        fc = c / np.sqrt(self.unit_state)
        fsp = np.stack([fc.real, fc.imag], axis=1)[:, :, None, :].astype(np.float32)

        enc = self._run("enc", {"feat_erb": fe, "feat_spec": fsp})
        m = self._run("erb_dec", {k: enc[k] for k in ("emb", "e3", "e2", "e1", "e0")})["m"]
        dfo = self._run("df_dec", {"emb": enc["emb"], "c0": enc["c0"]})
        coefs = dfo["coefs"]
        self.lsnr = np.asarray(enc["lsnr"]).reshape(self.n, -1)[:, 0].astype(float)

        gains = np.repeat(m[:, 0, 0], self.erb, axis=1)
        Y = X * gains
        self.hist = np.roll(self.hist, -1, axis=1)
        self.hist[:, -1] = X[:, : self.nb_df]
        co = coefs[:, 0].reshape(self.n, self.nb_df, self.df_order, 2)
        co = co[..., 0] + 1j * co[..., 1]
        Y[:, : self.nb_df] = np.sum(np.transpose(self.hist, (0, 2, 1)) * co, axis=2)
        return Y
