# Sotto Voce prototype: offline test pipeline

Tests the comms voice cleaner's processing on recordings before any hardware is built:
reference delay finding, a partitioned-block frequency-domain echo canceller (the same
structure the real-time box would run), residual echo suppression, and two noise
suppressors: DeepFilterNet3 low-latency (neural) and a classical spectral baseline.

## Run

```
python run.py demo --out out_demo                    # synthetic stadium scene, scored against truth
python run.py process --map show.json --out out_show  # a real show multitrack
```

Options: `--tail` echo canceller tail after the bulk delay in seconds (default 0.4),
`--block` block size (default 256), `--no-dfn` skip the neural stage. Demo scene options:
`--distance` PA distance in metres, `--rt60`, `--ser` speech-to-PA-bleed ratio in dB,
`--crowd` speech-to-crowd ratio in dB, `--seed`.

## Track map for real recordings

```json
{
  "files": ["VENUE_DATE_TAKE.wav"],
  "talkers":    {"caller": 0, "foh_audio": 1, "foh_lx": 2, "backstage": 3, "bolero": 4},
  "references": {"pa_l": 5, "pa_r": 6, "sub": 7},
  "segments":   {"pa_only": [[30.0, 150.0]], "talk": [[612.4, 615.0], [640.2, 642.8]]}
}
```

Channel numbers count across all files in order (0-based). `segments` come from the
capture notes sheet: `pa_only` = PA playing, nobody talking; `talk` = known calls.
They are optional but make the numbers meaningful.

## Outputs (per talker)

- `1_mic.wav` input (high-passed at 80 Hz), `2_aec.wav` after echo cancelling,
  `3_aec_res.wav` plus residual echo suppression, `4_aec_res_dfn.wav` plus DeepFilterNet3,
  `4b_aec_res_spectral.wav` plus the classical suppressor. 32-bit float, 48 kHz.
- `spectrograms.png`, and `report.md` / `report.json` for the whole run: delay to each
  reference (ms and metres), where the echo canceller found the direct PA sound, and
  per-stage measurements.

## Measurements

- **Noise removed in gaps:** level drop where nobody is talking (pa_only segments).
- **Level change while talking:** includes the noise removed, so some drop is expected.
- **Synthetic scenes only** (clean speech known): echo removed, speech kept (gain applied
  to the voice; near 0 dB is good), SI-SDR (overall quality against the clean voice).

## Results on the synthetic scene (40 s, PA at 70 m, RT60 1.6 s, limiter-style PA)

| Case | Stage | Echo removed | Speech kept | SI-SDR |
| --- | --- | --- | --- | --- |
| Hard: PA bleed as loud as the voice | mic | | 0.0 dB | -0.1 dB |
| | echo canceller | 10.8 dB | +0.1 dB | 5.0 dB |
| | + residual suppressor | 12.4 dB | -0.4 dB | 6.2 dB |
| | + DeepFilterNet3 LL | | -1.9 dB | 14.2 dB |
| Typical: PA bleed 10 dB below the voice | mic | | 0.0 dB | 5.3 dB |
| | + residual suppressor | 6.2 dB | -0.1 dB | 7.3 dB |
| | + DeepFilterNet3 LL | | -3.2 dB | 13.6 dB |

Tail sweep (hard case): 0.2 s tail removed 3.6 dB of echo, 0.4 s 10.8 dB, 1.0 s 9.9 dB
(a longer filter converges more slowly inside a 40 s take).

The delay finder found the PA within about 1 ms at 40, 70 and 110 m.

## Caveats

- The demo scene is synthetic: TTS speech (ffmpeg flite, 16 kHz source, so nothing above
  8 kHz), a synthetic band, a synthetic reverb tail. Real stadium recordings are the test
  that counts.
- DeepFilterNet3 LL has 20 ms algorithmic latency (20 ms window, 10 ms hop, no look-ahead).
  Offline, its STFT delay is removed so outputs line up; in real time it would add 20 ms.
- The echo canceller is causal and block-based; offline it uses 256-sample blocks for speed.
  The box would use 128.
- Processing speed here is not representative of the box (Python, unoptimised).

## Requirements

Python 3.10+, numpy, scipy, matplotlib, onnxruntime; ffmpeg built with flite for the demo.
Run `python tools/get_models.py` once: it downloads the DeepFilterNet3 low-latency ONNX
model (github.com/Rikorose/DeepFilterNet, MIT or Apache-2.0), checks its checksum and builds
the frame-by-frame versions. Without it the neural stage is skipped.
