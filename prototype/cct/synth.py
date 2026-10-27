"""Synthetic stadium comms scene for testing the pipeline before real recordings exist.

mic = talker speech + PA bleed (reference through a delayed, reverberant, slightly
non-linear path) + crowd noise. Everything is returned separately so the processing
can be scored against the truth.
"""
import os
import subprocess
import tempfile
import numpy as np
from scipy.signal import resample_poly, lfilter

FS = 48000

CALLS = [
    "Standby lights cue forty five, video cue twelve",
    "Lights forty five go",
    "Standby pyro one, standby automation cue eight",
    "Pyro one, go. Automation eight, go",
    "Follow spots pick up the artist downstage centre",
    "Stand by for the walk out, house lights in ten",
]


def tts(text, voice="slt"):
    """Speech from ffmpeg's flite filter (16 kHz), resampled to 48 kHz."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.wav")
        safe = text.replace("'", "").replace(":", "")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"flite=text='{safe}':voice={voice}", "-y", p], check=True)
        from scipy.io import wavfile
        sr, x = wavfile.read(p)
    x = x.astype(np.float64) / 32768.0
    return resample_poly(x, FS // 16000, 1)


def music(dur, rng):
    """PA programme stand-in: random chords on sawtooth voices, bass line, a sung-vowel lead
    with changing notes, kick, snare and hats. Varied enough that the echo canceller can't
    cheat by exploiting a perfectly repeating signal."""
    from scipy.signal import sawtooth, butter, sosfilt
    n = int(dur * FS)
    y = np.zeros(n)
    beat = 60 / 124
    eighth = beat / 2
    scale = [0, 2, 3, 5, 7, 8, 10]
    root = 110.0
    for k in range(int(dur / eighth) + 1):
        s = int(k * eighth * FS)
        L = min(int(eighth * FS * 1.6), n - s)
        if L <= 0:
            break
        tt = np.arange(L) / FS
        env = np.exp(-tt * rng.uniform(2, 6))
        if k % 4 == 0:                                                   # chord every half bar
            deg = rng.choice(scale, 3, replace=False)
            for dg in deg:
                f = root * 2 ** (dg / 12) * rng.choice([1, 2])
                y[s:s + L] += 0.03 * sawtooth(2 * np.pi * f * tt + rng.uniform(0, 6)) * env
            y[s:s + L] += 0.2 * np.sin(2 * np.pi * root / 2 * 2 ** (deg[0] / 12) * tt) * env   # bass
        f = root * 4 * 2 ** (rng.choice(scale) / 12)                      # lead "vocal"
        lead = sawtooth(2 * np.pi * f * tt + 2 * np.sin(2 * np.pi * 5.5 * tt))
        fc = rng.uniform(500, 1200)
        sos = butter(2, [fc * 0.7, fc * 1.4], btype="band", fs=FS, output="sos")
        y[s:s + L] += 0.12 * sosfilt(sos, lead) * env
        nz = rng.standard_normal(L)
        if k % 2 == 0:                                                   # kick on beats
            y[s:s + L] += 0.6 * np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
        if k % 4 == 2:                                                   # snare on 2 and 4
            y[s:s + L] += 0.25 * nz * np.exp(-tt * 18)
        y[s:s + L] += 0.05 * nz * np.exp(-tt * 70)                       # hats
    return y / (np.max(np.abs(y)) + 1e-9) * 0.5


def room_ir(rt60, fs, rng, length_s=None, early=8):
    length_s = length_s or rt60 * 1.2
    n = int(length_s * fs)
    t = np.arange(n) / fs
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir = lfilter([1.0], [1.0, -0.6], ir)                                # darker tail
    for _ in range(early):
        ir[rng.integers(0, int(0.05 * fs))] += rng.uniform(0.3, 1.0) * rng.choice([-1, 1])
    ir[0] = 1.0
    return ir / np.sqrt(np.sum(ir ** 2))


def pink(n, rng):
    w = rng.standard_normal(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, w)


def make_scene(dur=40.0, distance_m=70.0, rt60=1.6, ser_db=0.0, snr_crowd_db=10.0,
               nonlinear=True, seed=1):
    """Returns dict with fs, mic, ref, speech, echo, crowd, delay_samples, segments."""
    rng = np.random.default_rng(seed)
    n = int(dur * FS)
    ref = music(dur, rng)
    # Speech: calls spread over the take, with gaps
    speech = np.zeros(n)
    talk = []
    tpos = 4.0
    for i, c in enumerate(CALLS):
        s = tts(c)
        start = int(tpos * FS)
        if start + len(s) >= n:
            break
        speech[start:start + len(s)] += s
        talk.append([tpos, tpos + len(s) / FS])
        tpos += len(s) / FS + rng.uniform(2.5, 4.5)
    # PA path: limiter-like soft clip, bulk delay (flight time), reverb
    pa = np.tanh(2.0 * ref) / 2.0 if nonlinear else ref
    delay = int(round(distance_m / 343.0 * FS))
    ir = room_ir(rt60, FS, rng)
    path = np.convolve(pa, ir)[:n]
    echo = np.concatenate([np.zeros(delay), path])[:n]
    crowd = pink(n, rng) * (1 + 0.5 * np.sin(2 * np.pi * 0.07 * np.arange(n) / FS))

    def rms(x, mask=None):
        x = x if mask is None else x[mask]
        return np.sqrt(np.mean(x ** 2) + 1e-20)
    sp_mask = np.abs(speech) > 1e-4
    echo *= rms(speech, sp_mask) / rms(echo) / 10 ** (ser_db / 20)
    crowd *= rms(speech, sp_mask) / rms(crowd) / 10 ** (snr_crowd_db / 20)
    mic = speech + echo + crowd
    peak = np.max(np.abs(mic))
    scale = 0.5 / peak
    pa_only = [[0.5, talk[0][0] - 0.2]] if talk else [[0.5, dur]]
    return dict(fs=FS, mic=mic * scale, ref=ref, speech=speech * scale, echo=echo * scale,
                crowd=crowd * scale, delay_samples=delay,
                segments={"talk": talk, "pa_only": pa_only})
