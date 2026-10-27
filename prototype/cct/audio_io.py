"""WAV input/output and track-map handling."""
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def read_wav(path, fs_target=48000):
    """Read a WAV (int16/int24/int32/float) as float64 in [-1, 1], shape (n, ch)."""
    fs, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) / 128.0
    else:
        x = x.astype(np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if fs != fs_target:
        g = np.gcd(fs, fs_target)
        x = resample_poly(x, fs_target // g, fs // g, axis=0)
    return fs_target, x


def write_wav(path, x, fs=48000):
    """Write float audio as 32-bit float WAV."""
    x = np.asarray(x, dtype=np.float32)
    wavfile.write(path, fs, x)


def load_trackmap(path):
    """Track map JSON:
    {
      "files": ["show.wav"]            # one multichannel file, or one file per track
      "talkers":    {"caller": 0, "foh_audio": 1},     # name -> channel index (across files)
      "references": {"pa_l": 5, "pa_r": 6, "sub": 7},
      "segments": {"pa_only": [[10.0, 70.0]], "talk": [[120.0, 125.0]]}   # optional, seconds
    }
    """
    with open(path) as f:
        return json.load(f)


def load_tracks(trackmap, base_dir="."):
    import os
    chans = []
    fs = 48000
    for f in trackmap["files"]:
        fs, x = read_wav(os.path.join(base_dir, f))
        chans.append(x)
    n = min(c.shape[0] for c in chans)
    allch = np.concatenate([c[:n] for c in chans], axis=1)
    talkers = {k: allch[:, v] for k, v in trackmap["talkers"].items()}
    refs = {k: allch[:, v] for k, v in trackmap["references"].items()}
    return fs, talkers, refs
