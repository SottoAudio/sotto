# Fail-safe bench tool

Proves the Riedel fallback before any noise cleaning exists. The tool passes four talkers
straight through the laptop. It sends the HEARTBEAT and PROBE tones that the real app will send,
and it never goes back to clean on its own. Buttons inject faults, and the tool times how long
a listener takes to land on dry.

## Get the app (no Python needed)

1. Download the latest build from the repository's Releases page, under **Sotto Bench (latest
   build)**: `Sotto-Bench-mac.zip` or `Sotto-Bench-windows.zip`. Every change to main is built
   automatically on GitHub's Mac and Windows machines and checked before it is published.
2. Unzip it.
   - **Mac:** move **Sotto Bench** to Applications. It isn't signed yet, so the first time,
     right-click it, choose **Open**, then **Open** again. On newer macOS you may need System
     Settings, then Privacy & Security, then **Open Anyway**. Allow microphone access when asked;
     that's how macOS lets it hear the virtual soundcard.
   - **Windows:** open the folder and run **Sotto Bench.exe**. If SmartScreen warns, choose
     **More info**, then **Run anyway**.
3. A setup page opens in your browser. Press **Start with the simulated Riedel** to try it with
   no hardware. For the real test, pick the virtual soundcard, check the channel numbers, and
   press **Start with this soundcard**. Your choices are remembered for next time.
4. The same tab becomes the console. Press **Quit** (twice) to stop.

Recordings and event logs go to `Documents/Sotto Bench Runs`.

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

These are Sotto's fixed system channels, so set Dante Virtual Soundcard to 64 × 64. Any of them
can be changed on the setup page or with the options below.

The marker tone is on every return from the laptop, so a listener can always hear which path
they're on. The analysis uses it the same way.

## The console

The app opens Sotto's console: **Live** (inputs, channel tiles, macros, outputs, the phones bus)
and **Settings**. It shows the bench tool's real state: arming, checks, faults, levels on every
channel and the event log. Solos go to the headphone outputs. The macros (All to dry, Restore
all) drive the real arming rules. Processing, operator profiles and plugin inserts are saved
but have no effect in this build: the talkers pass straight through. Recording and fault timing
are on the classic bench page (`/bench`, linked from Test without hardware).

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

## What this has been tested against

A simulated frame (VOX hold 400 ms, logic 20 ms, 4 ms network), in the automated tests and a
live run of the console:

- **Stop heartbeat:** the listener was on dry 420–440 ms after the fault, with no silence.
- **Silence all outputs:** about 380 ms of silence, with dry back after about 415 ms. The app
  disarmed about 205 ms after PROBE was lost and refused to re-arm until it had been green for
  another 5 s.

It has not yet been run against a real frame.
