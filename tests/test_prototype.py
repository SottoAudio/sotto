"""Prototype processing: echo canceller convergence, the ONNX editing helpers, and (when the
models have been downloaded with prototype/tools/get_models.py) the frame-by-frame and batched
speech model against the published whole-file model."""
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTO = os.path.join(ROOT, "prototype")
sys.path.insert(0, PROTO)
sys.path.insert(0, os.path.join(PROTO, "tools"))

MODELS = os.path.join(PROTO, "models")
HAVE_MODELS = all(os.path.isfile(os.path.join(MODELS, d, "enc.onnx"))
                  for d in ("DeepFilterNet3_ll", "DeepFilterNet3_ll_stream", "DeepFilterNet3_ll_stream_batch"))
FS = 48000


def scene(seconds=12.0, delay_ms=50.0, seed=0):
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    ref = lfilter([1.0], [1.0, -0.9], rng.standard_normal(n)) * 0.05
    ir = np.zeros(int(0.08 * FS))
    ir[0], ir[400], ir[1500] = 1.0, 0.5, -0.3
    d = int(delay_ms * FS / 1000)
    echo = np.concatenate([np.zeros(d), np.convolve(ref, ir)[: n - d]]) * 0.5
    mic = echo + rng.standard_normal(n) * 1e-4
    return ref, mic, echo


class EchoCancellerTests(unittest.TestCase):
    def test_engine_removes_echo(self):
        from cct.engine import Engine, HOP
        ref, mic, _ = scene()
        eng = Engine(["t1"], 1, model_dir=None, delay_ms=50.0, track_delay=None)
        outs = []
        for b in range(len(mic) // HOP):
            s = slice(b * HOP, (b + 1) * HOP)
            _, allv = eng.process(mic[None, s], ref[None, s])
            outs.append(allv[0]["echo"])
            self.assertTrue(np.all(np.isfinite(outs[-1])))
        out = np.concatenate(outs)
        tail = slice(len(out) - 3 * FS, len(out))
        erle = 10 * np.log10(np.mean(mic[tail] ** 2) / (np.mean(out[tail] ** 2) + 1e-20))
        self.assertGreater(erle, 10.0, f"echo removed only {erle:.1f} dB")   # measured 14.7 dB

    def test_delay_change_keeps_learning(self):
        from cct.engine import Engine, HOP
        ref, mic, _ = scene(delay_ms=150.0)
        eng = Engine(["t1"], 1, model_dir=None, delay_ms=60.0, track_delay=None)
        n = len(mic) // HOP
        for b in range(n):
            if b == n // 3:
                eng.talkers[0].set_delay(int(0.150 * FS))        # applied by the audio thread
            s = slice(b * HOP, (b + 1) * HOP)
            eng.process(mic[None, s], ref[None, s])
        self.assertEqual(eng.talkers[0].bulk, (int(0.150 * FS) - int(0.06 * FS)) // HOP * HOP)
        self.assertEqual(eng.talkers[0].resets, 0)


@unittest.skipUnless(HAVE_MODELS, "models not downloaded (python prototype/tools/get_models.py)")
class SpeechModelTests(unittest.TestCase):
    def test_protobuf_round_trip(self):
        import pb
        with open(os.path.join(MODELS, "DeepFilterNet3_ll", "enc.onnx"), "rb") as f:
            raw = f.read()
        self.assertEqual(pb.encode(pb.decode(raw)), raw)

    def test_streaming_matches_whole_file(self):
        from cct.denoise import DeepFilterNetONNX
        from cct.stream_dfn import StreamingDFN, StreamSTFT
        rng = np.random.default_rng(3)
        t = np.arange(FS * 3) / FS
        x = 0.1 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0) + 0.02 * rng.standard_normal(len(t))
        off = DeepFilterNetONNX(os.path.join(MODELS, "DeepFilterNet3_ll")).enhance(x)
        dfn = StreamingDFN(os.path.join(MODELS, "DeepFilterNet3_ll_stream"))
        st = StreamSTFT()
        y = np.concatenate([st.synthesise(dfn.process_spec(st.analyse(x[i * 480:(i + 1) * 480])))
                            for i in range(len(x) // 480)])[480:]
        n = min(len(y), len(off)) - 2000
        rel = np.sqrt(np.mean((off[:n] - y[:n]) ** 2)) / (np.sqrt(np.mean(off[:n] ** 2)) + 1e-20)
        self.assertLess(20 * np.log10(rel + 1e-20), -80.0)

    def test_batch_matches_single(self):
        import onnxruntime as ort
        d = os.path.join(MODELS, "DeepFilterNet3_ll_stream_batch")
        with open(os.path.join(d, "states.json")) as f:
            meta = json.load(f)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        enc = ort.InferenceSession(os.path.join(d, "enc.onnx"), so)
        names = [o.name for o in enc.get_outputs()]

        def states(B):
            out = {}
            for s in meta["enc"]:
                sh = list(s["shape"])
                sh[1 if s["kind"] == "gru" else 0] = B
                out[s["in"]] = np.zeros(sh, np.float32)
            return out

        rng = np.random.default_rng(0)
        B = 3
        sb, s1 = states(B), [states(1) for _ in range(B)]
        for _ in range(10):
            fe = rng.standard_normal((B, 1, 1, 32)).astype(np.float32)
            fs = rng.standard_normal((B, 2, 1, 96)).astype(np.float32)
            rb = dict(zip(names, enc.run(None, {"feat_erb": fe, "feat_spec": fs, **sb})))
            for s in meta["enc"]:
                sb[s["in"]] = rb[s["out"]]
            for b in range(B):
                r1 = dict(zip(names, enc.run(None, {"feat_erb": fe[b:b + 1], "feat_spec": fs[b:b + 1], **s1[b]})))
                for s in meta["enc"]:
                    s1[b][s["in"]] = r1[s["out"]]
                self.assertTrue(np.allclose(r1["emb"][0], rb["emb"][b], atol=1e-4))


if __name__ == "__main__":
    unittest.main()
