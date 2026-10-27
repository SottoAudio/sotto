"""Bulk delay finding between a reference (PA feed) and a mic, by GCC-PHAT."""
import numpy as np


def gcc_phat(mic, ref, fs, max_delay_s=0.6):
    """Delay (in samples) of `mic` relative to `ref`, and peak strength (0..1)."""
    n = len(mic) + len(ref)
    nfft = 1 << (n - 1).bit_length()
    M = np.fft.rfft(mic, nfft)
    R = np.fft.rfft(ref, nfft)
    cross = M * np.conj(R)
    cross /= np.abs(cross) + 1e-12
    cc = np.fft.irfft(cross, nfft)
    max_lag = int(max_delay_s * fs)
    # The PA can only arrive after the reference, so search positive lags only.
    a = np.abs(cc[: max_lag + 1])
    imax = int(np.argmax(a))
    peak = float(a[imax])
    # A strong peak shortly before the maximum is the direct sound (the maximum may be a
    # reflection). Look back up to 60 ms, no further, so musical periodicity can't fool it.
    lo = max(0, imax - int(0.06 * fs))
    cand = np.nonzero(a[lo:imax + 1] >= 0.6 * peak)[0]
    lag = lo + int(cand[0]) if len(cand) else imax
    return lag, peak


def track_delay(mic, ref, fs, win_s=4.0, hop_s=2.0, max_delay_s=0.6, min_peak=0.01):
    """Sliding-window delay estimates. Returns list of (time_s, lag_samples, peak)."""
    win, hop = int(win_s * fs), int(hop_s * fs)
    out = []
    for start in range(0, max(1, len(mic) - win), hop):
        m = mic[start:start + win]
        r = ref[start:start + win]
        if np.std(r) < 1e-6:
            continue
        # Skip windows where the mic is mostly gated off (talk-key taps): nothing to match.
        if np.mean(np.abs(m) > 1e-5) < 0.5:
            continue
        lag, peak = gcc_phat(m, r, fs, max_delay_s)
        if peak >= min_peak:
            out.append((start / fs, lag, peak))
    return out


def bulk_delay(mic, ref, fs, max_delay_s=0.6):
    """Robust overall delay: median of confident window estimates (falls back to a global estimate)."""
    est = track_delay(mic, ref, fs, max_delay_s=max_delay_s)
    if est:
        lags = np.array([e[1] for e in est])
        peaks = np.array([e[2] for e in est])
        good = lags[peaks >= np.median(peaks)]
        return int(np.median(good)), est
    lag, _ = gcc_phat(mic, ref, fs, max_delay_s)
    return lag, []
