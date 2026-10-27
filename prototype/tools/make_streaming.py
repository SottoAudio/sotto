"""Turn the DeepFilterNet3 ONNX export (whole-sequence) into frame-by-frame models.

Two edits per graph, nothing else changes:
  * every GRU's initial hidden state becomes a graph input, and its final state a graph output;
  * every causal time-padding (Pad before a Conv with a time kernel of k) becomes
    Concat(previous k-1 frames, this frame), and the last k-1 frames become a graph output.
Run once T=1 frame at a time, feeding each state output back in as the next state input,
the result equals the whole-sequence model.

  python tools/make_streaming.py models/DeepFilterNet3_ll models/DeepFilterNet3_ll_stream
"""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb  # noqa: E402

FLOAT = 1


def load(path):
    model = pb.decode(open(path, "rb").read())
    gi = next(i for i, e in enumerate(model) if e[0] == 7)
    graph = pb.decode(model[gi][2])
    return model, gi, graph


def save(model, gi, graph, path):
    model[gi][2] = pb.encode(graph)
    with open(path, "wb") as f:
        f.write(pb.encode(model))


def graph_nodes(graph):
    return [pb.Node(v) for f, _, v in graph if f == 1]


def set_nodes(graph, nodes):
    rest = [e for e in graph if e[0] != 1]
    return [[1, 2, n.encode()] for n in nodes] + rest


def plan(nodes):
    """Find causal pads and GRUs. Returns list of state dicts."""
    consumers = {}
    for n in nodes:
        for i in n.inputs:
            consumers.setdefault(i, []).append(n)
    states = []
    for n in nodes:
        if n.op == "Pad":
            cons = consumers.get(n.outputs[0], [])
            conv = [c for c in cons if c.op == "Conv"]
            assert len(conv) == 1, n.name
            kt = conv[0].attrs()["kernel_shape"][0]
            assert kt > 1, n.name
            states.append(dict(kind="pad", node=n.name, src=n.inputs[0], dst=n.outputs[0], k=kt - 1))
        elif n.op == "GRU":
            states.append(dict(kind="gru", node=n.name, h0=n.inputs[5], yh=n.outputs[1]))
    for i, st in enumerate(states):
        st["in"] = f"state_in_{i}"
        st["out"] = f"state_out_{i}"
    return states


def probe_shapes(src_path, feeds, names):
    """Shapes of intermediate tensors, by exposing them as outputs of a temporary model."""
    import onnxruntime as ort
    model, gi, graph = load(src_path)
    for nm in names:
        graph.append([12, 2, pb.encode([[1, 2, nm.encode()],
                                         [2, 2, pb.encode([[1, 2, pb.encode([[1, 0, FLOAT]])]])]])])
    tmp = src_path + ".probe.onnx"
    save(model, gi, graph, tmp)
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(tmp, so)
        outs = sess.run(names, feeds)
    finally:
        os.remove(tmp)
    return {n: o.shape for n, o in zip(names, outs)}


def convert(src_path, dst_path, feeds):
    model, gi, graph = load(src_path)
    nodes = graph_nodes(graph)
    states = plan(nodes)
    shapes = probe_shapes(src_path, feeds, [s["src"] for s in states if s["kind"] == "pad"])
    new_nodes, extra_inits = [], []
    by_name = {s["node"]: s for s in states}
    for n in nodes:
        st = by_name.get(n.name)
        if st is None:
            new_nodes.append(n)
            continue
        if st["kind"] == "pad":
            shp = list(shapes[st["src"]])
            st["shape"] = [1, shp[1], st["k"], shp[3]]
            new_nodes.append(pb.make_node("Concat", [st["in"], st["src"]], [st["dst"]],
                                          n.name + "/stream_concat", {"axis": 2}))
            base = n.name.replace("/", "_") + "_stream"
            for suffix, vals in (("starts", [-st["k"]]), ("ends", [2 ** 62]), ("axes", [2])):
                extra_inits.append(pb.make_int64_tensor(f"{base}_{suffix}", vals))
            new_nodes.append(pb.make_node("Slice", [st["dst"], f"{base}_starts", f"{base}_ends", f"{base}_axes"],
                                          [st["out"]], n.name + "/stream_slice"))
        else:
            ins = n.inputs
            ins[5] = st["in"]
            outs = n.outputs
            n.set_io(inputs=ins, outputs=[outs[0], st["out"]])
            st["shape"] = [1, 1, int(n.attrs()["hidden_size"])]
            new_nodes.append(n)
    graph = set_nodes(graph, new_nodes)
    for t in extra_inits:
        graph.append([5, 2, t])
    for st in states:
        graph.append([11, 2, pb.make_value_info(st["in"], FLOAT, st["shape"])])
        graph.append([12, 2, pb.make_value_info(st["out"], FLOAT, st["shape"])])
    save(model, gi, graph, dst_path)
    return [{"in": s["in"], "out": s["out"], "shape": s["shape"], "kind": s["kind"], "node": s["node"]}
            for s in states]


def main(src, dst):
    import onnxruntime as ort
    os.makedirs(dst, exist_ok=True)
    T = 12
    rng = np.random.default_rng(0)
    fe = rng.standard_normal((1, 1, T, 32)).astype(np.float32)
    fsp = rng.standard_normal((1, 2, T, 96)).astype(np.float32)
    enc = ort.InferenceSession(os.path.join(src, "enc.onnx"))
    e0, e1, e2, e3, emb, c0, _ = enc.run(None, {"feat_erb": fe, "feat_spec": fsp})
    feeds = {
        "enc": {"feat_erb": fe, "feat_spec": fsp},
        "erb_dec": {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0},
        "df_dec": {"emb": emb, "c0": c0},
    }
    meta = {}
    for m in ("enc", "erb_dec", "df_dec"):
        meta[m] = convert(os.path.join(src, f"{m}.onnx"), os.path.join(dst, f"{m}.onnx"), feeds[m])
        print(m, [(s["kind"], s["shape"]) for s in meta[m]])
    with open(os.path.join(dst, "states.json"), "w") as f:
        json.dump(meta, f, indent=1)
    for fn in ("config.ini", "LICENSE-MIT"):
        if os.path.exists(os.path.join(src, fn)):
            shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
