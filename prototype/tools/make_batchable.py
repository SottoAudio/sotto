"""Make the frame-by-frame DeepFilterNet models accept a batch of talkers (one ONNX call per 10 ms
for all talkers). Only the declared input/output shapes change: the batch dimension becomes 'B'
(dimension 0, or dimension 1 for GRU states)."""
import os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb

FLOAT = 1


def relax(path_in, path_out):
    model = pb.decode(open(path_in, "rb").read())
    gi = next(i for i, e in enumerate(model) if e[0] == 7)
    g = pb.decode(model[gi][2])
    for e in g:
        if e[0] in (11, 12):
            name, et, shape = pb.value_info_shape(e[2])
            if not shape:
                continue
            idx = 1 if (name.startswith("state_") and len(shape) == 3) else 0
            shape = list(shape)
            shape[idx] = "B"
            e[2] = pb.make_value_info(name, et, shape)
    g = [e for e in g if e[0] != 13]          # drop stale intermediate shape hints
    model[gi][2] = pb.encode(g)
    open(path_out, "wb").write(pb.encode(model))


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for m in ("enc", "erb_dec", "df_dec"):
        relax(os.path.join(src, f"{m}.onnx"), os.path.join(dst, f"{m}.onnx"))
    for f in ("states.json", "config.ini", "LICENSE-MIT"):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
