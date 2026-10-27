# Sotto

**Sotto: comms voice cleaner.** Sotto removes PA bleed and crowd noise from intercom talkers in
real time, so show callers and crew are understood in the loudest moments. If anything fails, a
Riedel fail-safe puts listeners back on the dry mic.

It's an open-source desktop app for macOS and Windows. It runs on a laptop at the comms
engineer's station and connects to a Riedel Artist frame through Dante Virtual Soundcard or
Merging VAD. Phase 1 handles four key operators: show caller, creative director, technical
manager and stage manager. The design grows to 64 channels in and out.

## How it works

- **Processing per talker**, every 10 ms:
  1. an echo canceller that uses the PA programme (PGM) feed as its reference;
  2. a residual echo suppressor;
  3. a neural speech enhancer (DeepFilterNet3, low-latency).
- **Sidetone never passes through the laptop.** It stays on the Riedel crosspoint.
- **Fail-safe:** the app sends a HEARTBEAT tone only while every check passes. The Riedel
  watches it with a 4-wire VOX: if it stops for longer than the hold time (200–400 ms), logic
  puts listeners back on dry. A PROBE tone looped through the matrix proves both directions of
  the network path.
- **Never back to clean automatically.** After any fallback, the engineer re-arms from the
  console, Companion or a panel key, and only once every check has been green for 5 s.

## In this repository

| Folder | What it is | Status |
| --- | --- | --- |
| [`failsafe/`](failsafe/) | Sotto Bench: the console, the Riedel fail-safe (HEARTBEAT, PROBE, latching arm, fault injection, timing analysis, simulated frame), and the processing with operator profiles and VST3 inserts | Ready for a bench test |
| [`engine/plugin-test/`](engine/plugin-test/) | The VST3 hosting library Sotto uses for inserts, and `sotto-plugin-test`, which checks a plugin (CEDAR, for example) on its own: latency, settings save/restore, CPU | Built and tested on Mac, Windows and Linux |
| [`prototype/`](prototype/) | Python prototype of the processing: offline test pipeline, live laptop app in shadow mode, frame-by-frame and batched speech model | Tested on synthetic stadium audio |
| [`docs/`](docs/) | Design notes | |
| [`tests/`](tests/) | Automated tests, run on macOS, Windows and Linux for every change | |

The production app (a C++ engine with a browser-based console) comes next. It will reuse the
fail-safe rules and Riedel setup proven by the bench tool.

## Quick start

- Try the fail-safe with no hardware: see [`failsafe/README.md`](failsafe/README.md), then run
  `try_simulation.command`.
- Try the processing on the demo scene: see [`prototype/README_LIVE.md`](prototype/README_LIVE.md),
  then run `try_demo.command`.
- Run the tests: `python prototype/tools/get_models.py` once, then
  `python -m unittest discover -s tests -v`.

## Licence

MIT, see [LICENSE](LICENSE). DeepFilterNet models are by Schröter et al., under MIT /
Apache-2.0. They are downloaded by `prototype/tools/get_models.py`, not stored here.
