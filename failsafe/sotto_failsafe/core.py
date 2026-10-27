"""The bench tool's audio core, independent of the audio driver so it can be tested anywhere.

Every block:
  * talker inputs are passed straight to their returns. Every return also carries a quiet
    MARKER tone, so a listener (and analyse.py) can tell whether they are hearing the laptop
    path or the Riedel's dry path, whatever the app's own state;
  * PROBE is always sent, and every PROBE input is checked; a probe missing for longer than
    `probe_timeout_s` is a system fault, which disarms everything (latched);
  * HEARTBEAT is sent only while armed.
Faults can be injected from the console to time the Riedel fallback.

The engineer's headphones (monitor outputs) carry the SOLO BUS: whatever is soloed, summed, with
level, dim and a -3 dBFS limiter. It only reads the inputs and outputs; it never changes them.
Nothing soloed means silence.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from .arming import ArmState, Event
from .signals import Tone, ToneWatch, dbfs


@dataclass
class Config:
    fs: int = 48000
    talker_in: list = field(default_factory=lambda: [0, 1, 2, 3])
    talker_out: list = field(default_factory=lambda: [0, 1, 2, 3])
    talker_names: list = field(default_factory=lambda: ["caller", "creative", "tech mgr", "stage mgr"])
    probe_in: list = field(default_factory=lambda: [4])
    probe_out: int = 4
    heartbeat_out: int = 5
    listen_in: int = None          # optional: a listener's feed routed back, for timing
    pgm_in: int = None             # optional: the PGM reference (metered only, for now)
    monitor_out: list = field(default_factory=list)
    heartbeat_hz: float = 1000.0
    heartbeat_dbfs: float = -20.0
    probe_hz: float = 2000.0
    probe_dbfs: float = -20.0
    marker_hz: float = 2930.0
    marker_dbfs: float = -36.0
    marker_on: bool = True
    probe_timeout_s: float = 0.2
    arm_hold_s: float = 5.0
    startup_grace_s: float = 1.0


class FailsafeCore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        n = len(cfg.talker_in)
        self.arm = ArmState(n, arm_hold_s=cfg.arm_hold_s)
        self.hb = Tone(cfg.heartbeat_hz, cfg.heartbeat_dbfs, cfg.fs)
        self.probe = Tone(cfg.probe_hz, cfg.probe_dbfs, cfg.fs)
        self.markers = [Tone(cfg.marker_hz, cfg.marker_dbfs, cfg.fs) for _ in range(n)]
        self.watch = [ToneWatch(cfg.probe_hz, cfg.fs, cfg.probe_timeout_s) for _ in cfg.probe_in]
        self.samples = 0
        self.xruns = 0
        self.blocks = 0
        self.levels = {"talkers": [-120.0] * n, "listen": -120.0}
        self.monitor_source = None
        self.solo = {"ids": [], "mode": "clicked", "level_db": 0.0, "dim": False, "dim_db": -20.0}
        self._solo_plan = ([], [])     # (left taps, right taps); a tap is ("i" or "o", 0-based channel)
        self._solo_gain = 0.0
        self._limit = 1.0
        self.lv_in = None
        self.lv_out = None
        # injected faults: name -> audio time it ends (None = until cleared)
        self.faults = {}
        self.pending_freeze_ms = 0
        self.health_why = ""
        self.extra_health = None    # optional callable -> (ok, why): the processing engine's check
        self.log = []               # Event list, drained by the app

    @property
    def t(self):
        return self.samples / self.cfg.fs

    # ------------------------------------------------------------ control (any thread)
    def inject(self, name, seconds=None, ms=None):
        t = self.t
        if name == "freeze":
            self.pending_freeze_ms = int(ms or 300)
            self.log.append(Event(t, "fault_injected", f"freeze {self.pending_freeze_ms} ms", "console"))
            return
        if name == "stop_heartbeat":
            self.log.append(Event(t, "fault_injected", "stop heartbeat (app fault)", "console"))
            self.arm.request_disarm(t, "console", "injected fault: heartbeat stopped")
            return
        self.faults[name] = None if seconds is None else t + float(seconds)
        self.log.append(Event(t, "fault_injected", f"{name}" + (f" for {seconds} s" if seconds else ""), "console"))

    def set_solo(self, ids=None, mode=None, level_db=None, dim=None, dim_db=None):
        """ids like "i3" (input 3) or "o3" (output 3), counted from 1. Modes: clicked, dry, clean, split
        (dry left, clean right). Dry/clean swap a talker's input for its return and back."""
        s = dict(self.solo)
        if ids is not None:
            s["ids"] = [str(i) for i in ids][:128]
        if mode in ("clicked", "dry", "clean", "split"):
            s["mode"] = mode
        if level_db is not None:
            s["level_db"] = max(-60.0, min(12.0, float(level_db)))
        if dim is not None:
            s["dim"] = bool(dim)
        if dim_db is not None:
            s["dim_db"] = max(-60.0, min(0.0, float(dim_db)))
        cfg = self.cfg
        tin, tout = list(cfg.talker_in), list(cfg.talker_out)

        def parse(x):
            try:
                k, n = x[0], int(x[1:]) - 1
            except (ValueError, IndexError):
                return None
            if k not in "io" or n < 0 or (k == "o" and n in cfg.monitor_out):
                return None
            return (k, n)

        def dry(t):
            return ("i", tin[tout.index(t[1])]) if t[0] == "o" and t[1] in tout else t

        def clean(t):
            return ("o", tout[tin.index(t[1])]) if t[0] == "i" and t[1] in tin else t

        taps = [t for t in map(parse, s["ids"]) if t]
        m = s["mode"]
        if m == "dry":
            left = right = sorted(set(map(dry, taps)))
        elif m == "clean":
            left = right = sorted(set(map(clean, taps)))
        elif m == "split":
            left, right = sorted(set(map(dry, taps))), sorted(set(map(clean, taps)))
        else:
            left = right = sorted(set(taps))
        self.solo = s
        self._solo_plan = (left, right)
        return {"left": [f"{k}{n + 1}" for k, n in left], "right": [f"{k}{n + 1}" for k, n in right]}

    def _solo_bus(self, x, out, frames, silent):
        cfg = self.cfg
        left, right = self._solo_plan

        def mix(taps):
            y = np.zeros(frames)
            for k, n in taps:
                if k == "i" and n < x.shape[1]:
                    y += x[:, n]
                elif k == "o" and n < out.shape[1]:
                    y += out[:, n]
            return y

        s = self.solo
        target = 0.0 if silent or not (left or right) else 10 ** ((s["level_db"] + (s["dim_db"] if s["dim"] else 0.0)) / 20)
        g = np.linspace(self._solo_gain, target, frames, endpoint=False) + (target - self._solo_gain) / frames
        self._solo_gain = target
        l_sig, r_sig = mix(left) * g, mix(right) * g
        ceiling = 10 ** (-3 / 20)
        peak = max(float(np.max(np.abs(l_sig))) if frames else 0.0, float(np.max(np.abs(r_sig))) if frames else 0.0)
        need = 1.0 if peak <= ceiling else ceiling / peak
        self._limit = min(need, min(1.0, self._limit * 10 ** (6 / 20 * frames / cfg.fs)))   # instant attack, 6 dB/s release
        mon = cfg.monitor_out
        out[:, mon[0]] = np.clip(l_sig * self._limit, -ceiling, ceiling)
        if len(mon) > 1:
            out[:, mon[1]] = np.clip(r_sig * self._limit, -ceiling, ceiling)

    def clear_faults(self):
        self.faults.clear()
        self.log.append(Event(self.t, "faults_cleared", "", "console"))

    # ------------------------------------------------------------ audio
    def process(self, indata, outdata, status=None, returns=None, dry=None):
        """indata: (frames, n_in) float32; outdata: (frames, n_out) float32, written in place.
        returns/dry: optional (frames, n_talkers) processed and delay-matched dry signals from the
        processing engine; without them each talker is passed straight through."""
        cfg = self.cfg
        frames = outdata.shape[0]
        t = self.t
        if status:
            self.xruns += 1
        if self.pending_freeze_ms:
            ms, self.pending_freeze_ms = self.pending_freeze_ms, 0
            time.sleep(ms / 1000.0)          # deliberately miss deadlines: what does the soundcard do?
        for k, end in list(self.faults.items()):
            if end is not None and t >= end:
                del self.faults[k]
                self.log.append(Event(t, "fault_ended", k, "auto"))

        x = np.asarray(indata, dtype=np.float64)
        # --- checks
        for w, ch in zip(self.watch, cfg.probe_in):
            w.update(x[:, ch], t)
        healthy, why = True, ""
        if t >= cfg.startup_grace_s:
            for i, w in enumerate(self.watch):
                if not w.ok(t):
                    healthy = False
                    why = f"PROBE {i + 1} missing for {1000 * min(w.missing_s(t), 99):.0f} ms"
                    break
        else:
            healthy, why = False, "starting up"
        if healthy and self.extra_health is not None:
            ok, w = self.extra_health()
            if not ok:
                healthy, why = False, w
        self.health_why = why
        self.arm.update_health(t, healthy, why)
        for e in self.arm.drain_events():
            self.log.append(e)

        # --- outputs
        outdata.fill(0.0)
        silent_all = "silence_all" in self.faults
        mute_returns = "mute_returns" in self.faults
        for i, (ci, co) in enumerate(zip(cfg.talker_in, cfg.talker_out)):
            self.levels["talkers"][i] = dbfs(np.sqrt(np.mean(x[:, ci] ** 2)) + 1e-12)
            if returns is None:
                sig = x[:, ci].copy()
            elif self.arm.armed and not self.arm.talker_armed[i]:
                sig = np.asarray(dry[:, i], dtype=np.float64)      # tripped: dry inside the app
            else:
                sig = np.asarray(returns[:, i], dtype=np.float64)
            mk = self.markers[i].block(frames)
            if cfg.marker_on:
                sig = sig + mk
            if not (silent_all or mute_returns):
                outdata[:, co] = sig
        if not silent_all:
            outdata[:, cfg.probe_out] = self.probe.block(frames)
        else:
            self.probe.block(frames)
        hb = self.hb.block(frames)
        if self.arm.heartbeat_on and not silent_all:
            outdata[:, cfg.heartbeat_out] = hb
        if cfg.listen_in is not None:
            self.levels["listen"] = dbfs(np.sqrt(np.mean(x[:, cfg.listen_in] ** 2)) + 1e-12)
        if cfg.monitor_out:
            self._solo_bus(x, outdata, frames, silent_all)
        self.lv_in = 20 * np.log10(np.sqrt(np.mean(x * x, axis=0)) + 1e-9)
        o = np.asarray(outdata, dtype=np.float64)
        self.lv_out = 20 * np.log10(np.sqrt(np.mean(o * o, axis=0)) + 1e-9)
        self.samples += frames
        self.blocks += 1

    def state(self):
        t = self.t
        a = self.arm
        return {
            "t": round(t, 2),
            "armed": a.armed, "heartbeat_on": a.heartbeat_on and "silence_all" not in self.faults,
            "reason": a.reason, "health": self.health_why or "all green",
            "green_for_s": round(a.green_for(t), 1), "arm_hold_s": a.arm_hold_s, "can_arm": a.can_arm(t),
            "talkers": [{"name": n, "armed": a.talker_armed[i], "level_db": round(self.levels["talkers"][i], 1)}
                        for i, n in enumerate(self.cfg.talker_names[: len(a.talker_armed)])],
            "probes": [{"ok": w.ok(t), "missing_ms": None if w.last_seen is None else round(1000 * w.missing_s(t))}
                       for w in self.watch],
            "listen_db": round(self.levels["listen"], 1) if self.cfg.listen_in is not None else None,
            "faults": sorted(self.faults), "xruns": self.xruns, "monitor": self.monitor_source,
            "levels_in": [] if self.lv_in is None else [round(max(-120.0, float(v)), 1) for v in self.lv_in],
            "levels_out": [] if self.lv_out is None else [round(max(-120.0, float(v)), 1) for v in self.lv_out],
            "solo": dict(self.solo, left=[f"{k}{n + 1}" for k, n in self._solo_plan[0]],
                         right=[f"{k}{n + 1}" for k, n in self._solo_plan[1]]),
            "channels": {
                "talker_in": [c + 1 for c in self.cfg.talker_in], "talker_out": [c + 1 for c in self.cfg.talker_out],
                "probe_in": [c + 1 for c in self.cfg.probe_in], "probe_out": self.cfg.probe_out + 1,
                "heartbeat_out": self.cfg.heartbeat_out + 1, "monitor_out": [c + 1 for c in self.cfg.monitor_out],
                "listen_in": None if self.cfg.listen_in is None else self.cfg.listen_in + 1,
                "pgm_in": None if self.cfg.pgm_in is None else self.cfg.pgm_in + 1,
            },
        }
