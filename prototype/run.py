#!/usr/bin/env python3
"""Comms clean-up offline test.

  python run.py demo --out out_demo                 # synthetic stadium scene, scored against truth
  python run.py process --map show.json --out out   # real show multitrack (see README)

Writes before/after WAVs per talker and stage, report.json, report.md and spectrogram PNGs.
"""
import argparse
import json
import os
import time
import numpy as np

from cct.audio_io import write_wav, load_trackmap, load_tracks
from cct.pipeline import process_talker, measure

HERE = os.path.dirname(os.path.abspath(__file__))
DFN_DIR = os.path.join(HERE, "models", "DeepFilterNet3_ll")


def load_dfn(disable):
    if disable or not os.path.isfile(os.path.join(DFN_DIR, "enc.onnx")):
        return None
    from cct.denoise import DeepFilterNetONNX
    return DeepFilterNetONNX(DFN_DIR)


def spectrograms(stages, fs, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(stages)
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 2.2 * len(keys)), sharex=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        ax.specgram(stages[k] + 1e-7 * np.random.default_rng(0).standard_normal(len(stages[k])), NFFT=1024, Fs=fs, noverlap=768, cmap="magma", vmin=-140, vmax=-20)
        ax.set_ylim(0, 12000)
        ax.set_ylabel("Hz")
        ax.set_title(k, fontsize=9, loc="left")
    np.atleast_1d(axes)[-1].set_xlabel("seconds")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(out, results, meta):
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump({"meta": meta, "talkers": results}, f, indent=2)
    lines = [f"# Comms clean-up offline test", "", f"Run: {meta['when']}  ", f"Source: {meta['source']}", ""]
    for name, r in results.items():
        info = r["info"]
        lines += [f"## {name}", ""]
        for k, d in info["delays"].items():
            lines.append(f"- Reference **{k}**: delay {d['ms']} ms (about {d['metres']} m)"
                         + (f", spread {d['spread_ms']} ms over {d['windows']} windows" if d["spread_ms"] is not None else ""))
        lines.append(f"- Direct PA sound found by the echo canceller at {info['filter_delay_ms']} ms "
                     f"(about {info['filter_delay_m']} m)" + (f"; true delay {info['true_delay_ms']} ms" if 'true_delay_ms' in info else ""))
        lines.append(f"- Bulk delay applied {info['bulk_delay_ms']} ms; echo canceller tail {info['tail_ms']} ms "
                     f"({info['partitions']} partitions of {info['block']} samples)")
        if "dfn_latency_ms" in r:
            lines.append(f"- Speech model: DeepFilterNet3 low-latency, algorithmic latency {r['dfn_latency_ms']} ms")
        lines += ["", "| Stage | Noise removed in gaps (dB) | Level change while talking (dB) | Echo removed (dB) | Speech kept (dB) | SI-SDR (dB) |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for st, m in r["metrics"].items():
            lines.append(f"| {st} | {m.get('noise_reduction_db', '')} | {m.get('talk_level_change_db', '')} | "
                         f"{m.get('echo_removed_db', '')} | {m.get('speech_gain_db', '')} | {m.get('si_sdr_db', '')} |")
        lines.append("")
    lines += ["Noise removed in gaps: level drop where nobody is talking. Level change while talking: includes the "
              "noise removed, so some drop is expected. Echo removed, speech kept (gain applied to the voice itself; "
              "near 0 dB is good) and SI-SDR (overall quality against the clean voice) only exist for synthetic scenes, "
              "where the clean speech is known.", ""]
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(lines))


def run_demo(args):
    from cct.synth import make_scene
    sc = make_scene(dur=args.dur, distance_m=args.distance, rt60=args.rt60, ser_db=args.ser,
                    snr_crowd_db=args.crowd, seed=args.seed)
    fs = sc["fs"]
    dfn = load_dfn(args.no_dfn)
    out = args.out
    os.makedirs(out, exist_ok=True)
    t = time.time()
    stages, info = process_talker(sc["mic"], {"pa": sc["ref"]}, fs, tail_s=args.tail, block=args.block, dfn=dfn)
    elapsed = time.time() - t
    info["true_delay_ms"] = round(1000 * sc["delay_samples"] / fs, 1)
    stages_all = {"0_truth_speech": sc["speech"][: len(stages["1_mic"])], **stages}
    metrics = measure(stages, fs, truth=sc)
    d = os.path.join(out, "caller")
    os.makedirs(d, exist_ok=True)
    for k, v in stages_all.items():
        write_wav(os.path.join(d, f"{k}.wav"), v, fs)
    write_wav(os.path.join(d, "reference_pa.wav"), sc["ref"], fs)
    spectrograms({k: v for k, v in stages_all.items()}, fs, os.path.join(d, "spectrograms.png"),
                 f"Synthetic scene: PA at {args.distance} m, RT60 {args.rt60} s")
    res = {"caller": {"info": info, "metrics": metrics, "processing_s": round(elapsed, 1)}}
    if dfn:
        res["caller"]["dfn_latency_ms"] = dfn.latency_ms
    write_report(out, res, {"when": time.strftime("%Y-%m-%d %H:%M"), "source": "synthetic",
                            "scene": vars(args) | {"func": None}})
    print(open(os.path.join(out, "report.md")).read())


def run_process(args):
    tm = load_trackmap(args.map)
    base = os.path.dirname(os.path.abspath(args.map))
    fs, talkers, refs = load_tracks(tm, base)
    dfn = load_dfn(args.no_dfn)
    os.makedirs(args.out, exist_ok=True)
    res = {}
    for name, mic in talkers.items():
        t = time.time()
        stages, info = process_talker(mic, refs, fs, tail_s=args.tail, block=args.block, dfn=dfn)
        metrics = measure(stages, fs, segments=tm.get("segments"))
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        for k, v in stages.items():
            write_wav(os.path.join(d, f"{k}.wav"), v, fs)
        spectrograms(stages, fs, os.path.join(d, "spectrograms.png"), name)
        res[name] = {"info": info, "metrics": metrics, "processing_s": round(time.time() - t, 1)}
        if dfn:
            res[name]["dfn_latency_ms"] = dfn.latency_ms
    write_report(args.out, res, {"when": time.strftime("%Y-%m-%d %H:%M"), "source": args.map})
    print(open(os.path.join(args.out, "report.md")).read())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("demo", "process"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=f"out_{name}")
        s.add_argument("--tail", type=float, default=0.4, help="echo canceller tail after the bulk delay (s)")
        s.add_argument("--block", type=int, default=256, help="block size in samples")
        s.add_argument("--no-dfn", action="store_true", help="skip the DeepFilterNet stage")
        if name == "demo":
            s.add_argument("--dur", type=float, default=40.0)
            s.add_argument("--distance", type=float, default=70.0, help="PA distance (m)")
            s.add_argument("--rt60", type=float, default=1.6)
            s.add_argument("--ser", type=float, default=0.0, help="speech-to-PA-bleed ratio (dB)")
            s.add_argument("--crowd", type=float, default=10.0, help="speech-to-crowd ratio (dB)")
            s.add_argument("--seed", type=int, default=1)
        else:
            s.add_argument("--map", required=True, help="track map JSON")
    args = p.parse_args()
    run_demo(args) if args.cmd == "demo" else run_process(args)


if __name__ == "__main__":
    main()
