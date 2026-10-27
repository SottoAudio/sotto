#!/usr/bin/env python3
"""Comms voice cleaner: live laptop prototype (shadow mode).

  python rt.py --list                                   # show audio devices and channel counts
  python rt.py --selftest                               # check this laptop is fast enough
  python rt.py --device "Scarlett" --mic 1 --ref 3,4 --out 1 --out-dry 2 --monitor 3,4 --record
  python rt.py --play recordings/<take>/<file>_trackmap.json --mic 1 --ref 1,2   # replay a recording

Channels are numbered from 1, as on the interface. Open http://127.0.0.1:8765 for the controls.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def list_devices():
    import sounddevice as sd
    apis = sd.query_hostapis()
    print("Index  In  Out  Host API            Name")
    for i, d in enumerate(sd.query_devices()):
        print(f"{i:5d} {d['max_input_channels']:3d} {d['max_output_channels']:4d}  "
              f"{apis[d['hostapi']]['name'][:18]:18s}  {d['name']}")
    print("\nPass --device with the index or part of the name. On Windows prefer a "
          "'Windows WASAPI' entry for lower latency.")


def selftest(a):
    import time
    import numpy as np
    from cct.engine import Engine, HOP
    model = None if a.no_dfn else os.path.join(HERE, "models", "DeepFilterNet3_ll_stream")
    n_t = max(1, len((a.mic or "1").split(",")))
    n_r = max(1, len((a.ref or "1,2").split(",")))
    print(f"Self-test: {n_t} talker(s), {n_r} reference(s), speech model {'on' if model else 'off'}")
    eng = Engine([f"t{i}" for i in range(n_t)], n_r, model_dir=model, delay_ms=150, track_delay=None)
    rng = np.random.default_rng(0)
    ts = []
    for b in range(1000):
        m = rng.standard_normal((n_t, HOP)) * 0.05
        r = rng.standard_normal((n_r, HOP)) * 0.1
        t = time.perf_counter()
        out, _ = eng.process(m, r)
        ts.append(time.perf_counter() - t)
        if not np.all(np.isfinite(out)):
            print("FAIL: invalid output")
            return 1
    ts = np.array(ts[100:]) * 1000
    p99, mx = np.percentile(ts, 99), ts.max()
    print(f"Per 10 ms block: mean {ts.mean():.2f} ms, 99th percentile {p99:.2f} ms, worst {mx:.2f} ms")
    if p99 < 6:
        print("OK: plenty of headroom.")
    elif p99 < 9:
        print("OK but tight: close other apps, use mains power, and consider --latency 0.03.")
    else:
        print("TOO SLOW for this many talkers: use fewer talkers, or --no-dfn (echo cancelling only).")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list audio devices and exit")
    p.add_argument("--selftest", action="store_true", help="time the processing on this laptop and exit")
    p.add_argument("--device", help="audio interface: index or part of the name")
    p.add_argument("--mic", help="input channel(s) carrying the talker mic(s), e.g. 1 or 1,2")
    p.add_argument("--names", help="talker names, e.g. caller,foh")
    p.add_argument("--ref", help="input channel(s) carrying the PA reference, e.g. 3,4")
    p.add_argument("--ref-names", help="reference names, e.g. pa_l,pa_r")
    p.add_argument("--out", help="output channel(s) for the processed talker(s), one per mic")
    p.add_argument("--out-dry", help="output channel(s) for the dry talker(s), time-aligned")
    p.add_argument("--monitor", help="headphone pair, e.g. 3,4: Output / Dry / Split, chosen on the page")
    p.add_argument("--delay-ms", type=float, help="fix the PA delay (ms) instead of measuring it")
    p.add_argument("--tail", type=float, default=0.4, help="echo canceller length after the delay (s)")
    p.add_argument("--no-dfn", action="store_true", help="echo cancelling only, no speech model")
    p.add_argument("--latency", default="low", help="interface buffering: low, high, or seconds e.g. 0.03")
    p.add_argument("--record", action="store_true", help="start recording straight away")
    p.add_argument("--record-dir", default=os.path.join(HERE, "recordings"))
    p.add_argument("--play", help="track map JSON: replay a recording through the engine instead of live input")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach the page from another device")
    a = p.parse_args()

    if a.list:
        return list_devices()
    if a.selftest:
        return selftest(a)
    if not a.mic or not a.ref:
        p.error("--mic and --ref are required (see --list for channel counts)")
    from cct.live import LiveApp
    LiveApp(a).run()


if __name__ == "__main__":
    sys.exit(main())
