"""Download the DeepFilterNet3 low-latency model and build the frame-by-frame versions Sotto uses.

  python tools/get_models.py

Fetches DeepFilterNet3_ll_onnx.tar.gz from the DeepFilterNet repository (MIT / Apache-2.0,
Schroeter et al.), checks its SHA-256, and writes:
  models/DeepFilterNet3_ll/               the published whole-sequence model (offline pipeline)
  models/DeepFilterNet3_ll_stream/        one 10 ms frame at a time (live engine)
  models/DeepFilterNet3_ll_stream_batch/  as above, any number of talkers per call
"""
import hashlib
import io
import os
import shutil
import sys
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

COMMIT = "d375b2d8309e0935d165700c91da9de862a99c31"
URL = f"https://github.com/Rikorose/DeepFilterNet/raw/{COMMIT}/models/DeepFilterNet3_ll_onnx.tar.gz"
LICENSE_URL = f"https://raw.githubusercontent.com/Rikorose/DeepFilterNet/{COMMIT}/LICENSE-MIT"
SHA256 = "5998e58e8ba0e09bb76986ef97b84afa065a571ef282d4a1222f341e3251cf3a"


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main():
    models = os.path.join(ROOT, "models")
    base = os.path.join(models, "DeepFilterNet3_ll")
    os.makedirs(base, exist_ok=True)
    if not os.path.isfile(os.path.join(base, "enc.onnx")):
        local = os.environ.get("SOTTO_MODEL_TARBALL")
        if local:
            print("Using local copy", local)
            data = open(local, "rb").read()
        else:
            print("Downloading", URL)
            data = fetch(URL)
        digest = hashlib.sha256(data).hexdigest()
        if digest != SHA256:
            sys.exit(f"Checksum mismatch: got {digest}, expected {SHA256}. Not using this download.")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            for m in tf.getmembers():
                name = os.path.basename(m.name)
                if m.isfile() and name in ("enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini"):
                    with tf.extractfile(m) as src, open(os.path.join(base, name), "wb") as dst:
                        shutil.copyfileobj(src, dst)
        try:
            text = fetch(LICENSE_URL)
            with open(os.path.join(base, "LICENSE-MIT"), "wb") as f:
                f.write(text)
        except Exception as e:  # licence text is a courtesy copy; the model works without it
            print("Could not fetch the licence text:", e)
    else:
        print("Model already present:", base)

    import make_streaming
    import make_batchable
    stream = os.path.join(models, "DeepFilterNet3_ll_stream")
    batch = os.path.join(models, "DeepFilterNet3_ll_stream_batch")
    print("Building frame-by-frame model ...")
    make_streaming.main(base, stream)
    print("Building batched model ...")
    os.makedirs(batch, exist_ok=True)
    for m in ("enc", "erb_dec", "df_dec"):
        make_batchable.relax(os.path.join(stream, f"{m}.onnx"), os.path.join(batch, f"{m}.onnx"))
    for f in ("states.json", "config.ini", "LICENSE-MIT"):
        if os.path.exists(os.path.join(stream, f)):
            shutil.copy(os.path.join(stream, f), os.path.join(batch, f))
    print("Done:", stream, "and", batch)


if __name__ == "__main__":
    main()
