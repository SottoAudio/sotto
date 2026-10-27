# Fail-safe bench tool

Proves the Riedel fallback before any noise cleaning exists. The tool passes four talkers
straight through the laptop. It sends the HEARTBEAT and PROBE tones that the real app will send,
and it never goes back to clean on its own. Buttons inject faults, and the tool times how long
a listener takes to land on dry.

## Get the app (no Python needed)

1. Download the latest build from the repository's Releases page, under **Sotto Voce Bench (latest
   build)**: `Sotto-Voce-Bench-mac.zip` or `Sotto-Voce-Bench-windows.zip`. Every change to main is built
   automatically on GitHub's Mac and Windows machines and checked before it is published.
2. Unzip it.
   - **Mac:** move **Sotto Voce Bench** to Applications. It isn't signed yet, so the first time,
     right-click it, choose **Open**, then **Open** again. On newer macOS you may need System
     Settings, then Privacy & Security, then **Open Anyway**. Allow microphone access when asked;
     that's how macOS lets it hear the virtual soundcard.
   - **Windows:** open the folder and run **Sotto Voce Bench.exe**. If SmartScreen warns, choose
     **More info**, then **Run anyway**.
3. A setup page opens in your browser. Press **Start with the simulated Riedel** to try it with
   no hardware. For the real test, pick the virtual soundcard, check the channel numbers, and
   press **Start with this soundcard**. Your choices are remembered for next time.
4. The same tab becomes the console. Press **Quit** (twice) to stop.

Recordings and event logs go to `Documents/Sotto Voce Bench Runs`.

## Running from source (developers)

Mac: double-click `setup_mac.command`, then `try_simulation.command`. Anywhere:
`pip install -r requirements.txt`, then `python sotto_bench.py`, which opens the same setup page.
To skip the setup page, use `python -m sotto_failsafe --simulate` or the full options below.

## What it sends and expects

| Signal | Default channel | What it is |
| --- | --- | --- |
| Talker inputs | in 1–4 | the four talkers' sends from the frame (Call to Port, as Design D) |
| Talker returns | out 1–4 | each talker passed straight through, plus a quiet 2.93 kHz marker tone |
| PROBE | out 62 → in 61 | 2 kHz at −20 dBFS, always on; the frame loops it straight back |
| HEARTBEAT | out 61 | 1 kHz at −20 dBFS, only while armed |
| PGM | in 62 | the programme feed, metered (the echo canceller's reference later) |
| Headphones | out 63–64 | the solo bus: whatever you solo on the console, nothing otherwise |
| LISTEN (optional) | in 63 | a listener's feed routed back, so the tool can time the fallback |

These are Sotto Voce's fixed system channels, so set Dante Virtual Soundcard to 64 × 64. Any of them
can be changed on the setup page or with the options below.

The marker tone is on every return from the laptop, so a listener can always hear which path
they're on. The analysis uses it the same way.

## The console

The app opens Sotto Voce's console: **Live** (inputs, channel tiles, macros, outputs, the phones bus)
and **Settings**. It shows the bench tool's real state: arming, checks, faults, levels on every
channel and the event log. Solos go to the headphone outputs. The macros (All to dry, Restore
all) drive the real arming rules. Recording and fault timing are on the classic bench page
(`/bench`, linked from Test without hardware).

## Processing, operator profiles and plugin inserts

Each talker runs through Sotto Voce's processing before it goes back to the frame:

- high-pass, then the echo canceller with PGM (input 62) as its reference;
- the residual suppressor, then the speech model (DeepFilterNet3), run in one batch for every
  talker on Full clean and skipped for the others;
- the operator's **insert plugin** (any VST3, for example CEDAR Voxis), then the trim.

The mode, suppression, speech-model limit, high-pass, echo tail and trim come from Settings ›
Processing, overridden by the operator's profile. Profiles, inserts and their settings are saved
by the app in `project.json` in its settings folder, so they survive a restart. An insert's
controls are shown in Settings › Operators once its profile is on a channel. The plugin's own
window doesn't open in this build.

Timing: the chain needs 20 ms and runs on its own thread with one block (10 ms) of buffering, so
about 30 ms plus the soundcard. The console shows how much of each 10 ms block it uses. If it
falls three blocks behind, Sotto Voce disarms and the frame goes to dry (latched, like any fault); a
slower laptop can use Echo + noise for some talkers, which skips the speech model. One talker's
problem (invalid audio, a diverging echo canceller, a failing plugin) trips only that talker to
dry until it's re-armed. `--no-processing` gives the old straight-through bench.

## Arming

- **Starts disarmed.** The HEARTBEAT is off, so the Riedel holds everyone on dry.
- **Arm clean** is accepted only after every check has been green for 5 s. You can arm from the
  console, from Companion (`POST /api/arm`, optional `{"talker": 1}`), or from a key on the
  engineer's panel through RRCS (`--rrcs-host`, with the panel key setting the ARM_REQ logic
  source).
- **Faults latch.** A missing PROBE (over 200 ms), a disarm, or an injected fault disarms
  everything and stays disarmed until someone re-arms. Tripping one talker leaves the others
  clean.
- Every arm, refusal, trip and fault goes to `bench_runs/<take>/<take>_events.csv`, with the
  audio time, the clock time and where it came from.

## Director setup for the bench (one talker is enough)

1. Route the four talker sends to the virtual soundcard (inputs 1–4). Route its outputs 1–4 back as
   four clean-return 4-wires.
2. **PROBE:** make output 62 a 4-wire input, with a fixed crosspoint back to a 4-wire output that
   feeds the laptop's input 61. Put it in the same stream as the talker sends.
3. **HEARTBEAT:** make output 61 a 4-wire input. Set VOX on about 10 dB under the level the
   heartbeat arrives at on that port's meter (VOX thresholds are set in dBu on the port), VOX
   off a few dB lower, and hold 400 ms. Add a virtual key on that port, type VOX,
   function Logic, which sets the logic source CLEAN_OK.
4. **Design D logic for talker 1:** listeners hear the clean return while CLEAN_OK is on, and
   talker 1's dry mic while it's off.
5. **PGM:** route the programme feed to the laptop's input 62.
6. **LISTEN:** route one listener's feed to a 4-wire output that comes back to the laptop's
   input 63.
7. **Optional:** add logic sources ARMED (set by the app through RRCS) and ARM_REQ (a key on
   your panel). Run with `--rrcs-host <RRCS address>`. With RRCS the console also shows the
   frame's CLEAN_OK and ARMED, and the All to dry macro can drop ARMED straight away.

For the dry path to be audible during the test, latch talker 1's talk key and put a sound source
near its mic.

## Running the test

```
python -m sotto_failsafe --device "Dante Virtual Soundcard" --talker-in 1-4 --talker-out 1-4 \
    --listen-in 63 --record
```

Arm, then use each fault button a few times, re-arming between them:

| Button | What it stands for | What should happen |
| --- | --- | --- |
| Stop heartbeat | the app finding a fault | listener moves to dry after the VOX hold, with no silence |
| Silence all outputs 3 s | laptop, soundcard or network dying | listener hears silence for the hold time, then dry; the app disarms (PROBE lost) and stays disarmed |
| Mute talker returns 3 s | app running but voices gone | listener hears silence; the heartbeat stays on, which shows why the real app needs per-talker watchdogs |
| Freeze audio 400 ms | the app hanging | shows whether the soundcard goes silent or repeats audio (it must go silent) |
| Crash the app | the process dying | listener lands on dry; the restarted tool comes back disarmed |

Then press **Stop recording and time the faults** in the console. From source,
`python -m sotto_failsafe.analyse <take folder>` does the same.

For each fault this shows how long the listener stayed on the laptop path, the longest silence
they heard, and when the dry audio arrived. It also saves them to `fallback_timing.csv`. The
crash test can't be timed by the tool itself, because it stops recording; record the listener
on something else for that one.

## Changing settings while it runs

Everything can be changed from the running console, without quitting:

- **Settings › Audio engine:**
  - the soundcard and its buffer;
  - phones bus dim, and the limiter ceiling;
  - phones on this Mac;
  - Check device: soundcard channels, sample rate, Core Audio, microphone permission, PROBE, PGM and the talker inputs.

  The sample rate is set for Dante Virtual Soundcard in Dante Controller, not here. Sotto Voce shows it and warns if it isn't 48 kHz.
- **Settings › Channels › I/O patch:** which channel does each job:
  - talker inputs and returns;
  - PGM;
  - the PROBE return and send;
  - HEARTBEAT;
  - the headphone outputs;
  - the listener feed.

  It shows the defaults and has a **Back to defaults** button. Applying checks for clashes and that the soundcard has the channels.
- **Setup** (top right): stops the engine and goes back to the setup page, where the Riedel frame's RRCS IP address and port are set. Everything you set is kept.

Changes that move audio restart the engine. HEARTBEAT stops, so the frame puts every listener on dry, and Sotto Voce starts again disarmed.

## When something isn't working: the diagnostics file

Sotto Voce Bench keeps a detailed log while it runs. Each line records the time, the thread, and the function and line of code that wrote it. Errors carry their full traceback. To get help, send the **diagnostics file**. It has everything needed to work out why something isn't working on that computer:

- the build and the computer: model, CPU cores, macOS, Low Power Mode, heat, mains or battery, Rosetta;
- every audio device PortAudio and Core Audio can see, with channel counts;
- the setup choices, options and channel layout;
- the engine, plus processing timings per stage;
- a short speed test: the processing run offline on the efficiency cores, at normal priority, and on the performance cores;
- the fail-safe state and recent events;
- which threads and programs are using the CPU;
- what every thread is running at that moment (file and line);
- the end of the log, and any crash records.

At the top, a **What looks wrong** list sums up the usual problems in plain English.

There are three ways to make it:

- **Setup page:** Troubleshooting › Download diagnostics. Use this when the engine won't start: the page also shows why the last start failed.
- **Console:** the Diagnostics button in the header, or Settings › Audio engine › Download diagnostics. The speed test is skipped while Sotto Voce is armed, so it can't make processing late during a show.
- **Terminal:** `"Sotto Voce Bench.app/Contents/MacOS/Sotto Voce Bench" --diagnostics` (add `--quick` to skip the speed test). The file goes to Downloads. Use this when the app won't open at all.

Files, in the settings folder (`~/Library/Application Support/Sotto Voce Bench/logs` on a Mac, `%APPDATA%\Sotto Voce Bench\logs` on Windows):

| File | What it is |
|---|---|
| `sotto-bench.log` | The log (5 files of 5 MB, oldest dropped). |
| `last-crash.json` | The last error that stopped Sotto Voce Bench. Shown on the setup page until the next good start. |
| `crash-native.log` | Python stacks of every thread if the app was killed by a crash in native code (a plugin, the speech model, the audio driver). |
| `sotto-diagnostics-*.txt` | A copy of every diagnostics file made. |

If the engine can't open the soundcard, Sotto Voce Bench goes back to the setup page and says why, for example "The soundcard doesn't have that many channels". It no longer quits silently.

## What this has been tested against

A simulated frame (VOX hold 400 ms, logic 20 ms, 4 ms network), in the automated tests and a
live run of the console:

- **Stop heartbeat:** the listener was on dry 420–440 ms after the fault, with no silence.
- **Silence all outputs:** about 380 ms of silence, with dry back after about 415 ms. The app
  disarmed about 205 ms after PROBE was lost and refused to re-arm until it had been green for
  another 5 s.

It has not yet been run against a real frame.
