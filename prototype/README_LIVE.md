# Live laptop prototype (shadow mode)

`rt.py` runs the comms voice cleaner live on a laptop with an audio interface. It takes a
**copy** of a talker's mic and the PA feed, and gives you the cleaned version on headphones and
on interface outputs, with a control page in the browser. It records everything at the same time.

It never sits in the live comms path: nobody on the show hears it unless you deliberately
patch an output somewhere.

## Install (once)

- **Mac:** install Python 3.12 from python.org, then double-click `setup_mac.command`.
  If macOS blocks it, right-click > Open.
- **Windows:** install Python 3.12 from python.org (tick "Add to PATH"), then double-click
  `setup_windows.bat`.

Setup downloads the DeepFilterNet3 model and converts it for frame-by-frame use
(`tools/get_models.py`; needs internet once). It ends with a self-test that times the
processing on your laptop. "OK" means it is fast
enough.

## Try it at home (no interface)

Double-click `try_demo.command` (Mac) or `try_demo_windows.bat`. It plays a 40 s synthetic stadium
scene through the processing to your laptop headphones, and the control page opens. Press
`Space` to flip between dry and processed, and `1` / `2` / `3` for Dry, Echo only and Clean.

## Run at the show

1. Find your interface and its channel count: `.venv/bin/python rt.py --list`
   (on Windows, `.venv\Scripts\python rt.py --list`).
2. Edit `start.command` / `start_windows.bat` to match your patch, then double-click it.

```
python rt.py --device "Scarlett" --mic 1 --names caller --ref 3,4 --ref-names pa_l,pa_r \
             --out 1 --out-dry 2 --monitor 3,4 --record
```

| Option | Meaning |
| --- | --- |
| `--mic 1` or `1,2` | input channels with the talker mic copies (one talker per channel) |
| `--ref 3,4` | input channels with the PA reference (L/R, or one mono channel) |
| `--out 1` | output channel for each talker's processed voice (whatever mode is selected) |
| `--out-dry 2` | output channel for each talker's dry voice, delayed 10 ms to line up |
| `--monitor 3,4` | headphone pair: Output, Dry or Split (dry left / processed right), chosen on the page |
| `--record` | start recording at once (or use the button on the page) |
| `--delay-ms 205` | fix the PA delay by hand instead of measuring it |
| `--no-dfn` | echo cancelling only (if the laptop is too slow for the speech model) |
| `--latency 0.03` | more interface buffering if you see dropouts |
| `--host 0.0.0.0` | reach the control page from a phone on the same network |

Channels are numbered from 1, as printed on the interface. The mic and the PA reference
**must come into the same interface** so they share a clock.

## Control page (http://127.0.0.1:8765)

- **Mode per talker:** Dry / Echo only / Clean. Switching is a 10 ms crossfade.
- **Headphones:** pick the talker, then Output / Dry / Split.
- **PA delay:** measured automatically from the music. It locks after 3 matching readings
  (about 15 s of PA). Re-measure, Fix delay or Reset canceller if needed.
- **Level drop after canceller:** mic level vs echo-cancelled level, averaged over 2 s. It
  includes talking and crowd noise, so expect less than the offline report's "echo removed".
- **Speech model:** shows "Voice" when it thinks someone is talking.
- **CPU:** processing time per 10 ms block. It goes red if the worst case passes 9 ms.
  Dropouts counts interface glitches.
- **Recording and marks:** `M`, or type a note and press Enter, to add a mark to `markers.csv`.
- **Keys:** `Space` Output ⇄ Dry, `S` Split, `1` `2` `3` mode, `M` mark.

## Recordings

`recordings/<date_time>/` holds:

- `*_partNN.wav` files: 24-bit, 48 kHz, a new file every 30 minutes. Channels are: each mic,
  then each reference, then each processed output (see `channels.txt`). Processed tracks run
  10 ms behind the mic tracks.
- `markers.csv`: your marks, with time into the take and into the file.
- `status_log.jsonl`: once a second, the delay, levels, CPU and dropouts.
- `*_trackmap.json` per file, ready for `python run.py process --map <file>`. Fill in its
  `segments` from the markers to get the gap and talk measurements.

Replay a recording through the live engine with
`python rt.py --play <trackmap> --mic 1 --ref <number of refs>`.

## Delay and what it costs

The processing adds 20 ms: a 10 ms block plus 10 ms of overlap in the speech model's
spectrum. The interface adds its own input and output buffering on top. The page shows the
total, typically 30–35 ms on a USB interface. That is over the 20 ms target for the finished
box, but fine for listening tests in shadow mode. The box closes the gap with a faster model
(see the pitch doc).

## How it works

`cct/engine.py` processes each 10 ms block in this order:

1. 80 Hz high-pass.
2. Echo canceller (`cct/aec.py`: partitioned frequency-domain, 400 ms after the bulk delay,
   one filter per reference).
3. Residual echo suppressor.
4. DeepFilterNet3 low-latency, run frame by frame (`cct/stream_dfn.py`).

All three versions (dry, echo-only, clean) are computed every block, so switching modes never
restarts anything. A worker process measures the PA delay without holding up the audio. A
diverging canceller resets itself. Any error in the audio callback outputs dry, never
silence.

The frame-by-frame speech model is generated from the published ONNX export by
`tools/make_streaming.py`. It exposes the GRU states and the causal-convolution history as
inputs and outputs. Its output matches the whole-file model to −126 dB, which is rounding.

## Tested here (no audio hardware in this environment)

- `tools/rt_sim.py` runs the engine block by block on the synthetic scene. Hard case
  (PA bleed as loud as the voice, PA at 70 m):
  - delay found within 0.2 ms, locked after 12 s;
  - 10.8 dB of echo removed;
  - SI-SDR −0.1 → 5.4 dB after echo cancelling, 14.7 dB after the speech model;
  - this matches the offline pipeline (12.4 dB and 14.2 dB).
- `tools/fake_sd/` stands in for the audio driver, to test the whole app in real time:
  - routing, crossfades, recording, markers and the control page all work;
  - no dropouts;
  - about 4 ms of processing per 10 ms block on this machine;
  - in a gap between calls the output fell from −22.6 dB to −78.9 dB.
- It has not yet been run on a real interface or a real stadium mic.

Licences: DeepFilterNet models MIT / Apache-2.0 (Schröter et al.), see
`models/DeepFilterNet3_ll_stream/LICENSE-MIT`.
