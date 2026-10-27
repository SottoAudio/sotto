"""The full offline chain for one talker, plus measurements."""
import numpy as np
from .delay import bulk_delay
from .aec import PBFDAF, residual_suppress
from .denoise import spectral_denoise


def hpf(x, fs, fc=80.0):
    from scipy.signal import butter, sosfilt
    sos = butter(2, fc, btype="highpass", fs=fs, output="sos")
    return sosfilt(sos, x)


def process_talker(mic, refs, fs, tail_s=0.4, block=256, mu=0.5, margin_s=0.06,
                   dfn=None, use_spectral=True):
    """mic: (T,), refs: dict name -> (T,). Returns (stages dict, info dict)."""
    mic = hpf(mic, fs)
    names = list(refs)
    delays = {}
    for k in names:
        d, est = bulk_delay(mic, refs[k], fs)
        delays[k] = {"samples": int(d), "ms": round(1000 * d / fs, 1),
                     "metres": round(d / fs * 343.0, 1), "windows": len(est),
                     "spread_ms": round(1000 * float(np.std([e[1] for e in est])) / fs, 1) if est else None}
    bulk = max(0, min(v["samples"] for v in delays.values()) - int(margin_s * fs))

    def run_aec(bulk):
        R = np.stack([np.concatenate([np.zeros(bulk), hpf(refs[k], fs)])[: len(mic)] for k in names])
        a = PBFDAF(len(names), fs, block=block, tail_s=tail_s, mu=mu)
        e, y = a.run(R, mic)
        return a, e, y

    aec, e, y = run_aec(bulk)
    # Refine: the learnt filter shows where the direct PA sound really lands. If it sits at the
    # very start or end of the tail, the bulk delay was off, so correct it and run again.
    ir = aec.impulse_response()
    env = np.max(np.abs(ir), axis=0)
    first = int(np.argmax(env >= 0.5 * env.max()))
    filt_delay = bulk + first
    lo, hi = int(0.01 * fs), len(env) - int(0.1 * fs)
    gcc_ok = all(v["windows"] >= 5 for v in delays.values())
    rerun = (not gcc_ok) and not (lo <= first <= hi)
    if rerun:
        bulk = max(0, filt_delay - int(margin_s * fs))
        aec, e, y = run_aec(bulk)
        ir = aec.impulse_response()
        env = np.max(np.abs(ir), axis=0)
        filt_delay = bulk + int(np.argmax(env >= 0.5 * env.max()))
    n = len(e)
    stages = {"1_mic": mic[:n], "2_aec": e}
    r = residual_suppress(e, y, fs)
    stages["3_aec_res"] = r
    if dfn is not None:
        stages["4_aec_res_dfn"] = dfn.enhance(r)
    if use_spectral:
        stages["4b_aec_res_spectral"] = spectral_denoise(r, fs)
    info = {"delays": delays, "bulk_delay_ms": round(1000 * bulk / fs, 1),
            "filter_delay_ms": round(1000 * filt_delay / fs, 1),
            "filter_delay_m": round(filt_delay / fs * 343.0, 1), "rerun": rerun,
            "tail_ms": round(1000 * aec.P * aec.N / fs), "block": block,
            "partitions": aec.P, "refs": names}
    return stages, info


# ---------------------------------------------------------------- measurements
def _mask_from_segments(n, fs, segs):
    m = np.zeros(n, bool)
    for a, b in segs or []:
        m[int(a * fs):int(b * fs)] = True
    return m


def level_db(x, m):
    return 10 * np.log10(np.mean(x[m] ** 2) + 1e-20) if np.any(m) else None


def measure(stages, fs, segments=None, truth=None, skip_s=5.0):
    """Per-stage levels. With `truth` (synthetic scenes) also echo removal and SI-SDR."""
    n = min(len(v) for v in stages.values())
    out = {}
    base = stages["1_mic"][:n]
    if truth is not None:
        fs_ = fs
        sp = hpf(truth["speech"], fs_)[:n]           # same filter as the chain input
        truth = dict(truth, crowd=hpf(truth["crowd"], fs_))
        quiet = np.abs(sp) < 1e-4
        quiet[: int(skip_s * fs)] = False
        talk = np.abs(sp) > 1e-3
    else:
        quiet = _mask_from_segments(n, fs, (segments or {}).get("pa_only"))
        talk = _mask_from_segments(n, fs, (segments or {}).get("talk"))
    for k, v in stages.items():
        v = v[:n]
        row = {}
        if np.any(quiet):
            row["noise_reduction_db"] = round(level_db(base, quiet) - level_db(v, quiet), 1)
        if np.any(talk):
            row["talk_level_change_db"] = round(level_db(v, talk) - level_db(base, talk), 1)
        if truth is not None:
            a = np.dot(v, sp) / (np.dot(sp, sp) + 1e-20)
            row["speech_gain_db"] = round(20 * np.log10(abs(a) + 1e-12), 1)
            row["si_sdr_db"] = round(10 * np.log10(np.sum((a * sp) ** 2) / (np.sum((v - a * sp) ** 2) + 1e-20)), 1)
            if k in ("2_aec", "3_aec_res"):
                cr = truth["crowd"][:n]
                row["echo_removed_db"] = round(10 * np.log10(
                    np.mean((base[quiet] - cr[quiet]) ** 2) / (np.mean((v[quiet] - cr[quiet]) ** 2) + 1e-20)), 1)
        out[k] = row
    return out
