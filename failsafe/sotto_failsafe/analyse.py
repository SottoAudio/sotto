"""Time the Riedel fallback from a bench recording.

  python -m sotto_failsafe.analyse bench_runs/<take>

For each injected fault or disarm, looks at the LISTEN channel (a listener's feed routed back
into the laptop) and reports, in ms after the event:
  * switch: when the listener stopped hearing the laptop path (marker tone gone). For a
    heartbeat fault this is the Riedel's switch time; if the laptop itself went silent it is
    almost immediate;
  * silence: the longest stretch of silence the listener heard in the 3 s after the event
    (the dropout);
  * dry back: when audio without the marker (the Riedel's dry path) arrived.
For the dry path to be audible, latch talker 1's key with a sound source near its mic.
"""
import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

from .signals import goertzel_power

MARKER_HZ = 2930.0


def read_wav24(path):
    import wave
    with wave.open(path, "rb") as w:
        n_ch, sw, fs, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw != 3:
        raise SystemExit("expected a 24-bit recording from the bench tool")
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    i = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
    i = np.where(i >= 1 << 23, i - (1 << 24), i)
    return fs, (i.astype(np.float64) / 8388608.0).reshape(-1, n_ch)


def marker_track(x, fs, hop_s=0.01, win_s=0.04, freq=MARKER_HZ):
    """Per 10 ms: (time, marker present, level dBFS)."""
    hop, win = int(hop_s * fs), int(win_s * fs)
    out = []
    for s in range(0, max(0, len(x) - win), hop):
        seg = x[s:s + win]
        pf = goertzel_power(seg, freq, fs)
        pn = 0.5 * (goertzel_power(seg, freq - 150, fs) + goertzel_power(seg, freq + 150, fs))
        level = 10 * math.log10(float(np.mean(seg ** 2)) + 1e-20)
        tone_db = 10 * math.log10(2 * pf + 1e-20)
        present = tone_db > -60 and pf > 30 * (pn + 1e-20)
        out.append(((s + win / 2) / fs, present, level))
    return out


def analyse(take_dir, listen_ch=None, silence_db=-60.0, window_s=3.0):
    wavs = sorted(glob.glob(os.path.join(take_dir, "*_bench.wav")))
    evs = sorted(glob.glob(os.path.join(take_dir, "*_events.csv")))
    if not wavs or not evs:
        raise SystemExit(f"no bench recording or events in {take_dir}")
    fs, x = read_wav24(wavs[0])
    if listen_ch is None:
        with open(glob.glob(os.path.join(take_dir, "*_channels.txt"))[0]) as f:
            names = f.read().splitlines()
        idx = [int(l.split("\t")[0]) - 1 for l in names if l.endswith("in: LISTEN")]
        if not idx:
            raise SystemExit("this take has no LISTEN channel (start the tool with --listen-in)")
        listen_ch = idx[0]
    # Recording starts when the user pressed Record; events are in audio time since start.
    rec_start = 0.0
    with open(evs[0], newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["event"] == "recording_started":
            rec_start = float(r["audio_time_s"])
            break
    track = marker_track(x[:, listen_ch], fs)
    times = np.array([t for t, _, _ in track]) + rec_start
    present = np.array([p for _, p, _ in track])
    level = np.array([l for _, _, l in track])
    results = []
    marks = []
    last_fault = -1e9
    for r in rows:
        t = float(r["audio_time_s"])
        if r["event"] == "fault_injected":
            marks.append((t, r))
            last_fault = t
        elif r["event"] == "disarmed" and t - last_fault > 1.0:
            marks.append((t, r))             # a disarm on its own, not one a fault caused
    starts = [t for t, _ in marks]
    for k, (t0, r) in enumerate(marks):
        nxt = starts[k + 1] if k + 1 < len(starts) else None
        end = min(t0 + window_s, nxt) if nxt is not None else t0 + window_s
        sel = (times >= t0) & (times < end)
        if not sel.any():
            continue
        tt, pp, ll = times[sel], present[sel], level[sel]
        had_marker = bool(present[(times >= t0 - 0.5) & (times < t0)].any())
        gone = tt[~pp]
        t_gone = float(gone[0]) if len(gone) else None
        quiet = ll < silence_db
        longest, cur, run_end = 0, 0, None
        for k, q in enumerate(quiet):
            cur = cur + 1 if q else 0
            if cur > longest:
                longest = cur
                run_end = k
        # Dry audio counts only after the silence (if the listener heard any), so leftover
        # audio from just before the fault isn't mistaken for the Riedel's dry path.
        start = t_gone
        if longest >= 3 and run_end is not None:
            start = float(tt[run_end])
        dry = tt[(~pp) & (ll > silence_db + 5)]
        if start is not None:
            dry = dry[dry > start]
        t_dry = float(dry[0]) if len(dry) else None
        results.append({
            "event": f"{r['event']}: {r['detail']}", "at_s": round(t0, 3),
            "marker_before": had_marker,
            "switch_ms": None if t_gone is None else round(1000 * (t_gone - t0)),
            "silence_ms": round(10 * longest),
            "dry_back_ms": None if t_dry is None else round(1000 * (t_dry - t0)),
        })
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("take_dir")
    p.add_argument("--listen-ch", type=int, help="channel number of LISTEN in the recording (1-based)")
    a = p.parse_args(argv)
    res = analyse(a.take_dir, None if a.listen_ch is None else a.listen_ch - 1)
    out = os.path.join(a.take_dir, "fallback_timing.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event", "at_s", "marker_before", "switch_ms", "silence_ms", "dry_back_ms"])
        w.writeheader()
        w.writerows(res)
    print(f"{'Event':45s} {'at (s)':>8s} {'switch':>8s} {'silence':>8s} {'dry back':>9s}")
    for r in res:
        fmt = lambda v: "-" if v is None else f"{v} ms"
        print(f"{r['event'][:45]:45s} {r['at_s']:8.2f} {fmt(r['switch_ms']):>8s} {fmt(r['silence_ms']):>8s} {fmt(r['dry_back_ms']):>9s}")
    print("Saved", out)


if __name__ == "__main__":
    sys.exit(main())
