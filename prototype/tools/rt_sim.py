"""Run the real-time engine block by block on the synthetic scene (or a track map) and score it
the same way as the offline pipeline. Checks the engine before it meets an audio interface.

  python tools/rt_sim.py --out out_rt_sim [--ser 0] [--distance 70] [--no-dfn]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from cct.engine import Engine, HOP, FS  # noqa: E402
from cct.pipeline import measure  # noqa: E402
from cct.audio_io import write_wav  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out_rt_sim")
    p.add_argument("--dur", type=float, default=40.0)
    p.add_argument("--distance", type=float, default=70.0)
    p.add_argument("--rt60", type=float, default=1.6)
    p.add_argument("--ser", type=float, default=0.0)
    p.add_argument("--crowd", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-dfn", action="store_true")
    p.add_argument("--delay-ms", type=float, default=None, help="manual delay instead of tracking")
    p.add_argument("--gated", action="store_true", help="mic is digital silence except while talking (talk-key tap)")
    a = p.parse_args()

    from cct.synth import make_scene
    sc = make_scene(dur=a.dur, distance_m=a.distance, rt60=a.rt60, ser_db=a.ser,
                    snr_crowd_db=a.crowd, seed=a.seed)
    model = None if a.no_dfn else os.path.join(HERE, "models", "DeepFilterNet3_ll_stream")
    eng = Engine(["caller"], 1, model_dir=model, delay_ms=a.delay_ms,
                 track_delay=None if a.delay_ms is not None else "inline")
    mic, ref = sc["mic"].copy(), sc["ref"]
    if a.gated:                      # key pressed 0.3 s before the call, released 0.5 s after
        gate = np.zeros(len(mic), bool)
        for t0, t1 in sc["segments"]["talk"]:
            gate[int((t0 - 0.3) * FS):int((t1 + 0.5) * FS)] = True
        mic[~gate] = 0.0
    nb = len(mic) // HOP
    outs = {k: np.zeros(nb * HOP) for k in ("dry", "echo", "clean")}
    times = []
    delay_log = []
    every = int(3.0 * FS / HOP)
    for b in range(nb):
        s = slice(b * HOP, (b + 1) * HOP)
        t0 = time.perf_counter()
        _, allv = eng.process(mic[None, s], ref[None, s])
        times.append(time.perf_counter() - t0)
        for k, v in allv[0].items():
            outs[k][s] = v
        if eng.tracker and b % every == every - 1:
            eng.tracker.step()
            t = eng.talkers[0]
            delay_log.append((round((b + 1) * HOP / FS, 1), t.delay_status,
                              None if t.delay_samples is None else round(1000 * t.delay_samples / FS, 1)))

    # Line the outputs up with the input for scoring: the STFT path delays by NFFT-HOP.
    d = 480
    n = nb * HOP - d
    from cct.pipeline import hpf
    stages = {"1_mic": hpf(outs["dry"], FS)[d:d + n], "3_echo": outs["echo"][d:d + n],
              "4_clean": outs["clean"][d:d + n]}
    truth = {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in sc.items()}
    met = measure(stages, FS, truth=truth)
    # echo removed for the processed stages (measure() only does it for offline stage names)
    from cct.pipeline import hpf
    quiet = np.abs(hpf(truth["speech"], FS)) < 1e-4
    quiet[: int(5 * FS)] = False
    base = stages["1_mic"]
    cr = hpf(truth["crowd"], FS)
    for k in ("3_echo",):
        v = stages[k]
        met[k]["echo_removed_db"] = round(10 * np.log10(
            np.mean((base[quiet] - cr[quiet]) ** 2) / (np.mean((v[quiet] - cr[quiet]) ** 2) + 1e-20)), 1)

    if a.gated:
        # Score only where the key is down: gaps are digital silence in and out.
        g = gate[d:d + n]
        sp = hpf(truth["speech"], FS)
        q = g & (np.abs(sp) < 1e-4)
        q[: int(5 * FS)] = False
        met = {}
        for k, v in stages.items():
            vv, ss = v[g], sp[g]
            al = np.dot(vv, ss) / np.dot(ss, ss)
            row = {"si_sdr_db_key_down": round(10 * np.log10(np.sum((al * ss) ** 2) / np.sum((vv - al * ss) ** 2)), 1),
                   "speech_gain_db": round(20 * np.log10(abs(al)), 1)}
            if k != "1_mic" and q.any():
                row["level_drop_in_hang_db"] = round(10 * np.log10(np.mean(base[q] ** 2) / (np.mean(v[q] ** 2) + 1e-20)), 1)
            met[k] = row
    t = np.array(times[50:]) * 1000
    true_ms = round(1000 * sc["delay_samples"] / FS, 1)
    rep = {"scene": vars(a), "true_delay_ms": true_ms, "delay_log": delay_log,
           "metrics": met, "block_ms": {"mean": round(t.mean(), 2), "p99": round(float(np.percentile(t, 99)), 2),
                                        "max": round(t.max(), 2), "budget": 10.0},
           "resets": eng.talkers[0].resets}
    os.makedirs(a.out, exist_ok=True)
    for k, v in stages.items():
        write_wav(os.path.join(a.out, f"{k}.wav"), v, FS)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    eng.stop()


if __name__ == "__main__":
    main()
