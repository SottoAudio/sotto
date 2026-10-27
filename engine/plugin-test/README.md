# sotto-plugin-test

Loads a VST3 plugin the way Sotto's engine will, outside any DAW, and tells you whether it can be
a Sotto insert. Use it on the show laptop when you trial a plugin such as CEDAR Voxis, ScreenVox
or DNS One.

## Get it

Download `sotto-plugin-test-mac.zip` or `sotto-plugin-test-windows.zip` from the
"sotto-plugin-test (latest build)" release on GitHub and unzip it. It isn't signed yet, so on a
Mac run this once in Terminal, in the folder you unzipped to:

```
xattr -d com.apple.quarantine sotto-plugin-test
```

## Use it

Install the plugin and its licence first (CEDAR: the plugin installer plus iLok License Manager,
with a trial or licence activated). Then, in Terminal (Mac) or PowerShell (Windows):

```
./sotto-plugin-test --list
./sotto-plugin-test "/Library/Audio/Plug-Ins/VST3/CEDAR Voxis.vst3" --in caller.wav --out caller_voxis.wav
```

On Windows plugins live in `C:\Program Files\Common Files\VST3`. `--list` shows the exact paths.

It reports:

- **Insert:** whether the plugin runs as a mono insert (Sotto's talkers are mono).
- **Latency:** in samples and ms, at 48 kHz and a 480-sample (10 ms) block. Sotto delays the
  bypassed path by this much so switching an insert never shifts the timing.
- **Audio:** that processing returned valid audio, and where the result was written.
- **Settings:** that the plugin's settings can be saved and loaded back into a fresh copy. This
  is what operator profiles rely on. Adaptive plugins (DNS One's LEARN, for example) may make two
  restored copies sound slightly different; that's expected.
- **CPU:** time per 10 ms block for several copies run together (`--instances`, default 4).
  Try `--instances 16` and `--instances 64` to see how far the laptop goes.
- **Controls:** every control the plugin exposes, with its current value. These are what the
  Sotto console will show and save in an operator profile.

Other options: `--set "Attenuation=12 dB"` (or a 0–1 value), `--save-state FILE`,
`--load-state FILE`, `--compensate` (remove the plugin's latency from `--out`), `--channel N`
(which channel of `--in`), `--block N`, `--json`. `--help` lists them all.

Exit codes: 0 all good; 1 wrong options or files; 2 the plugin wouldn't load; 3 it wouldn't run
or returned invalid audio; 4 its settings couldn't be saved or restored.

If a plugin asks for its licence (an iLok window, for example), that's a sign Sotto must let
plugin windows appear on first load. Note it down.

## Build it

```
cmake -S engine/plugin-test -B build/plugin-test -DCMAKE_BUILD_TYPE=Release
cmake --build build/plugin-test --config Release
```

CMake fetches the Steinberg VST3 SDK 3.8 (MIT licence) without its GUI library. Add
`-DSOTTO_BUILD_SDK_EXAMPLES=ON` to also build the SDK's ADelay example, which CI uses to test the
tool on Mac, Windows and Linux.
