"""The bench tool's audio core, independent of the audio driver so it can be tested anywhere.

Every block:
  * talker inputs are passed straight to their returns. Every return also carries a quiet
    MARKER tone, so a listener (and analyse.py) can tell whether they are hearing the laptop
    path or the Riedel's dry path, whatever the app's own state;
  * PROBE is always sent, and every PROBE input is checked; a probe missing for longer than
    `probe_timeout_s` is a system fault, which disarms everything (latched);
  * HEARTBEAT is sent only while armed.
Faults can be injected from the console to time the Riedel fallback.
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
        self.monitor_source = "listen" if cfg.listen_in is not None else 0
        # injected faults: name -> audio time it ends (None = until cleared)
        self.faults = {}
        self.pending_freeze_ms = 0
        self.health_why = ""
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

    def clear_faults(self):
        self.faults.clear()
        self.log.append(Event(self.t, "faults_cleared", "", "console"))

    # ------------------------------------------------------------ audio
    def process(self, indata, outdata, status=None):
        """indata: (frames, n_in) float32; outdata: (frames, n_out) float32, written in place."""
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
        self.health_why = why
        self.arm.update_health(t, healthy, why)
        for e in self.arm.drain_events():
            self.log.append(e)

        # --- outputs
        outdata.fill(0.0)
        silent_all = "silence_all" in self.faults
        mute_returns = "mute_returns" in self.faults
        for i, (ci, co) in enumerate(zip(cfg.talker_in, cfg.talker_out)):
            sig = x[:, ci].copy()
            self.levels["talkers"][i] = dbfs(np.sqrt(np.mean(sig ** 2)) + 1e-12)
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
            if self.monitor_source == "listen" and cfg.listen_in is not None:
                m = x[:, cfg.listen_in]
            else:
                m = outdata[:, cfg.talker_out[int(self.monitor_source)]]
            for ch in cfg.monitor_out:
                outdata[:, ch] = m
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
        }
