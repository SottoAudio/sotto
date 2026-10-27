"""Sotto fail-safe bench tool.

  python -m sotto_failsafe --list
  python -m sotto_failsafe --device "Dante Virtual Soundcard" --talker-in 1-4 --talker-out 1-4 --record

Channels count from 1 and follow Sotto's fixed system channels unless you say otherwise:
IN 61 PROBE return, IN 62 PGM, OUT 61 HEARTBEAT, OUT 62 PROBE, OUT 63/64 headphones (solo bus).
Set Dante Virtual Soundcard to 64 x 64. Open http://127.0.0.1:8766 for the console.
"""
import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import diagnostics, macos
from .arming import Event
from .core import Config, FailsafeCore

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = diagnostics.LOG


def config_dir():
    if os.environ.get("SOTTO_CONFIG_DIR"):             # tests and builds: never the user's own settings and logs
        return os.environ["SOTTO_CONFIG_DIR"]
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support", "Sotto Bench")
    if os.name == "nt":
        return os.path.join(os.environ.get("APPDATA", home), "Sotto Bench")
    return os.path.join(home, ".config", "sotto-bench")


def chans(s):
    if s in (None, ""):
        return []
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a) - 1, int(b)))
        else:
            out.append(int(part) - 1)
    return out


def _int24(x):
    x = np.clip(np.nan_to_num(x), -1.0, 1.0 - 1.0 / 8388608)
    i = (x * 8388608.0).astype("<i4")
    return i.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()


class Recorder:
    """Records the bench channels to one 24-bit WAV, off the audio thread."""

    def __init__(self, folder, names, fs):
        self.folder, self.names, self.fs = folder, names, fs
        self.q = queue.Queue(maxsize=4000)
        self.on = False
        self.path = None
        self.dropped = 0

    def start(self, take):
        os.makedirs(self.folder, exist_ok=True)
        self.path = os.path.join(self.folder, f"{take}_bench.wav")
        with open(os.path.join(self.folder, f"{take}_channels.txt"), "w") as f:
            for i, n in enumerate(self.names):
                f.write(f"{i + 1}\t{n}\n")
        self.w = wave.open(self.path, "wb")
        self.w.setnchannels(len(self.names))
        self.w.setsampwidth(3)
        self.w.setframerate(self.fs)
        self.on = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def push(self, block):
        if self.on:
            try:
                self.q.put_nowait(block.copy())
            except queue.Full:
                self.dropped += 1

    def _loop(self):
        while True:
            b = self.q.get()
            if b is None:
                break
            self.w.writeframes(_int24(b.reshape(-1)))     # header is patched on every write

    def stop(self):
        if self.on:
            self.on = False
            self.q.put(None)
            self.th.join(timeout=5)
            self.w.close()


def config_from_args(a):
    """The fail-safe Config for parsed command-line options (channels converted to 0-based)."""
    cfg = Config(
        talker_in=chans(a.talker_in), talker_out=chans(a.talker_out),
        probe_in=chans(a.probe_in), probe_out=chans(a.probe_out)[0],
        heartbeat_out=chans(a.heartbeat_out)[0],
        listen_in=(chans(a.listen_in)[0] if a.listen_in else None),
        pgm_in=(chans(a.pgm_in)[0] if a.pgm_in else None),
        monitor_out=chans(a.monitor),
        arm_hold_s=a.arm_hold, probe_timeout_s=a.probe_timeout / 1000.0,
        marker_on=not a.no_marker,
    )
    if a.names:
        cfg.talker_names = [n.strip() for n in a.names.split(",")]
    if len(cfg.talker_in) != len(cfg.talker_out):
        raise SystemExit("--talker-in and --talker-out need the same number of channels")
    cfg.talker_names = (cfg.talker_names + [f"talker {i + 1}" for i in range(len(cfg.talker_in))])[: len(cfg.talker_in)]
    return cfg


class BenchApp:
    def __init__(self, a):
        self.a = a
        cfg = config_from_args(a)
        self.cfg = cfg
        self.core = FailsafeCore(cfg)
        self.take = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(a.out_dir, self.take)
        os.makedirs(self.dir, exist_ok=True)
        self.events_f = open(os.path.join(self.dir, f"{self.take}_events.csv"), "w", newline="")
        self.events = csv.writer(self.events_f)
        self.events.writerow(["audio_time_s", "clock", "event", "detail", "source"])
        self.recent = []
        names = [f"in: {n}" for n in cfg.talker_names] + [f"in: PROBE {i + 1}" for i in range(len(cfg.probe_in))]
        if cfg.listen_in is not None:
            names.append("in: LISTEN")
        names += ["out: HEARTBEAT", f"out: {cfg.talker_names[0]} return"]
        self.rec = Recorder(self.dir, names, cfg.fs)
        self.rrcs = None
        if a.rrcs_host:
            from .rrcs import RRCSLink
            self.rrcs = RRCSLink(a.rrcs_host, a.rrcs_port, armed_name=a.rrcs_armed,
                                 arm_request_name=a.rrcs_arm_request, clean_ok_name=a.rrcs_clean_ok,
                                 on_arm_request=lambda: self.arm(None, "panel"))
        self.lock = threading.Lock()
        self._was_armed = False
        self.quit_ev = threading.Event()
        self.restart_device = None                       # set by /api/engine: the launcher restarts
        self.restart_with = {}
        self.restart_argv = None                         # a whole new command line (console: engine and patch)
        self.to_setup = False                            # console: back to the setup page
        self.srv = None
        self.timing = []
        self.sim = None
        self.proc = None
        self.proc_note = "off (--no-processing)" if a.no_processing else ""
        self.vst3 = None
        self.plugins_cache = None
        self.project_path = a.project or os.path.join(config_dir(), "project.json")
        self.project = None
        try:
            with open(self.project_path) as f:
                self.project = json.load(f)
        except (OSError, ValueError):
            self.project = None
        self._plugin_states_dirty = False
        self._last_state_save = 0.0
        self.callback_errors = 0
        self._cb_exc = None                              # the audio thread's last error, logged by the main loop
        self.cb_ms = 0.0
        self.cb_ms_max = 0.0
        self._cb_last = None                             # when the last block arrived
        self._dev_last = None                            # when the last soundcard callback ran
        self.regroup = None                              # set when the soundcard runs on a small buffer
        from .phones import LocalPhones
        from .testsrc import TestSources
        allowed = {c: f"talker {i + 1}" for i, c in enumerate(cfg.talker_in)}
        if cfg.pgm_in is not None:
            allowed[cfg.pgm_in] = "PGM"
        self.tests = TestSources(os.path.join(config_dir(), "test-audio"), allowed)
        self.local_phones = LocalPhones()
        self._status = None
        self.coreaudio = None
        self.gaps = self._new_gaps()                     # how evenly the soundcard delivers blocks
        self.gaps_last = None
        LOG.info("Settings: device=%s talkers in %s out %s (%s) | PROBE in %s out %s | HEARTBEAT out %s | PGM in %s | "
                 "phones %s | block %s | processing %s | RRCS %s | simulate %s",
                 a.device, a.talker_in, a.talker_out, ",".join(cfg.talker_names), a.probe_in, a.probe_out,
                 a.heartbeat_out, a.pgm_in, a.monitor, a.block, "off" if a.no_processing else "on",
                 a.rrcs_host or "none", a.simulate)

    # ------------------------------------------------------------ events
    def log(self, e: Event):
        LOG.info("event %s: %s (%s, audio time %.3f s)", e.kind, e.detail, e.source, e.t)
        row = [f"{e.t:.4f}", datetime.now().strftime("%H:%M:%S.%f")[:-3], e.kind, e.detail, e.source]
        self.events.writerow(row)
        self.events_f.flush()
        self.recent = (self.recent + [row])[-60:]

    def _drain(self):
        if self.core.arm.armed and self.tests.plan:     # arming switches every test source off
            self.tests.off()
            self.tests.switched_off += 1
            self.log(Event(self.core.t, "test_sources_off", "Sotto armed: every test source switched off", "auto"))
        exc, self._cb_exc = self._cb_exc, None
        if exc is not None:
            LOG.error("Audio callback error (%d so far); outputs were silenced for that block", self.callback_errors,
                      exc_info=(type(exc), exc, exc.__traceback__))
        while self.core.log:
            self.log(self.core.log.pop(0))
        armed = self.core.arm.armed
        if armed != self._was_armed:
            self._was_armed = armed
            if self.sim is not None:          # the simulated frame gets what RRCS would set
                self.sim.set_logic("ARMED", armed)
            if self.rrcs:
                ok = self.rrcs.set_armed_safe(armed)
                self.log(Event(self.core.t, "rrcs_armed_set" if ok else "rrcs_error",
                               f"ARMED={'on' if armed else 'off'}" + ("" if ok else f": {self.rrcs.last_error}"), "auto"))

    # ------------------------------------------------------------ commands
    def arm(self, talker, source):
        with self.lock:
            ok, msg = self.core.arm.request_arm(self.core.t, source, talker)
            for e in self.core.arm.drain_events():
                self.core.log.append(e)
        return {"ok": ok, "message": msg}

    def command(self, c):
        cmd = c.get("cmd")
        src = c.get("source", "console")
        talker = c.get("talker")
        talker = None if talker in (None, "", "all") else int(talker)
        core = self.core
        if cmd == "arm":
            return self.arm(talker, src)
        if cmd == "disarm":
            with self.lock:
                core.arm.request_disarm(core.t, src)
                for e in core.arm.drain_events():
                    core.log.append(e)
            return {"ok": True}
        if cmd == "trip" and talker is not None:
            with self.lock:
                core.arm.trip_talker(core.t, talker, "tripped from console", src)
                for e in core.arm.drain_events():
                    core.log.append(e)
            return {"ok": True}
        if cmd == "fault":
            name = c.get("name")
            if name == "crash":
                self.log(Event(core.t, "fault_injected", "crash (process exits now)", src))
                self.events_f.flush()
                os._exit(3)
            with self.lock:
                core.inject(name, seconds=c.get("seconds"), ms=c.get("ms"))
            return {"ok": True}
        if cmd == "clear_faults":
            with self.lock:
                core.clear_faults()
            return {"ok": True}
        if cmd == "solo":
            with self.lock:
                plan = core.set_solo(c.get("ids"), c.get("mode"), c.get("level_db"), c.get("dim"), c.get("dim_db"),
                                     c.get("limit_db"))
            return {"ok": True, **plan}
        if cmd == "monitor":                  # the classic bench console's listen selector
            sc = c.get("source_ch", "listen")
            if sc == "listen" and self.cfg.listen_in is not None:
                ids = [f"i{self.cfg.listen_in + 1}"]
            elif str(sc).isdigit() and int(sc) < len(self.cfg.talker_out):
                ids = [f"o{self.cfg.talker_out[int(sc)] + 1}"]
            else:
                ids = []
            with self.lock:
                core.monitor_source = sc
                core.set_solo(ids, "clicked")
            return {"ok": True}
        if cmd == "plugin_param":
            return self.plugin_param(int(c.get("talker", -1)), int(c.get("id", 0)), float(c.get("norm", 0)))
        if cmd == "rrcs_set":
            name, on = str(c.get("name", "")), bool(c.get("on"))
            if self.rrcs:
                ok = self.rrcs.set_logic_safe(name, on) if hasattr(self.rrcs, "set_logic_safe") else False
                self.log(Event(core.t, "rrcs_set" if ok else "rrcs_error",
                               f"{name}={'on' if on else 'off'}" + ("" if ok else f": {self.rrcs.last_error}"), src))
                return {"ok": ok, "error": None if ok else f"RRCS: {self.rrcs.last_error}"}
            if self.sim is not None and self.sim.set_logic(name, on):
                self.log(Event(core.t, "rrcs_set", f"{name}={'on' if on else 'off'} (simulated frame)", src))
                return {"ok": True}
            return {"ok": False, "error": "no RRCS link to the frame"}
        if cmd == "analyse":
            if self.rec.on:
                self.rec.stop()
                self.log(Event(core.t, "recording_stopped", self.rec.path, src))
            self.events_f.flush()
            if self.cfg.listen_in is None:
                return {"ok": False, "error": "start with a listener feed input to time the faults"}
            if not self.rec.path:
                return {"ok": False, "error": "nothing recorded yet"}
            from . import analyse
            try:
                res = analyse.analyse(self.dir)
            except SystemExit as e:
                return {"ok": False, "error": str(e)}
            except Exception as e:
                return {"ok": False, "error": repr(e)}
            import csv as _csv
            with open(os.path.join(self.dir, "fallback_timing.csv"), "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=["event", "at_s", "marker_before", "switch_ms", "silence_ms", "dry_back_ms"])
                w.writeheader()
                w.writerows(res)
            self.timing = res
            return {"ok": True, "message": f"timed {len(res)} faults; saved fallback_timing.csv"}
        if cmd == "mark":
            self.log(Event(core.t, "mark", str(c.get("note", ""))[:200], src))
            return {"ok": True}
        if cmd == "rec":
            if c.get("on") and not self.rec.on:
                self.rec.start(self.take)
                self.log(Event(core.t, "recording_started", self.rec.path, src))
            elif not c.get("on") and self.rec.on:
                self.rec.stop()
                self.log(Event(core.t, "recording_stopped", self.rec.path, src))
            return {"ok": True}
        return {"ok": False, "error": "unknown command"}

    def state(self):
        s = self.core.state()
        s.update({"take": self.take, "folder": self.dir, "recording": self.rec.on,
                  "rec_dropped": self.rec.dropped, "events": self.recent[-25:], "timing": self.timing,
                  "rrcs": self.rrcs.status() if self.rrcs else {"configured": False},
                  "latency_note": getattr(self, "latency_note", "")})
        frame = None
        if self.sim is not None:
            frame = self.sim.frame_state()
        elif self.rrcs:
            frame = self.rrcs.frame_state()
        s["frame"] = frame
        s["engine"] = getattr(self, "engine_info", None)
        if s["engine"] is not None:
            s["engine"] = {**s["engine"], "coreaudio": self.coreaudio,
                           "regroup": None if self.regroup is None else {
                               "underruns": self.regroup.underruns, "overruns": self.regroup.overruns}}
        g = self.gaps_last or self.gaps
        s["callback"] = {"ms": round(self.cb_ms, 3), "ms_max": round(self.cb_ms_max, 3), "errors": self.callback_errors,
                         "gaps": {"blocks": g["n"], "min_ms": round(g["min_ms"], 2) if g["n"] else None,
                                  "max_ms": round(g["max_ms"], 2), "bunched": g["bunched"], "late": g["late"],
                                  "device_callbacks": g["dev_n"], "device_gap_max_ms": round(g["dev_max_ms"], 2),
                                  "delay_ms": [round(g["delay_min_ms"], 2), round(g["delay_max_ms"], 2)]
                                  if g["delay_max_ms"] else None}}
        s["power"] = getattr(self, "power", {})
        s["tests"] = self.tests.state()
        s["local_phones"] = self.local_phones.state()
        s["processing"] = self.proc.state() if self.proc is not None else {"on": False, "why": self.proc_note}
        if self.proc is not None:
            s["processing"]["plugins_available"] = self.vst3 is not None
        return s

    # ------------------------------------------------------------ processing, profiles, inserts
    def _start_processing(self):
        if self.a.no_processing:
            return
        if self.a.block != 480:
            self.proc_note = "off: processing needs a 480-sample (10 ms) block"
            return
        try:
            from . import processing
            from . import vst3
        except Exception as e:                       # e.g. no onnxruntime/scipy: stay a pass-through bench
            LOG.exception("Processing could not start")
            self.proc_note = f"off: {e!r}"
            self.log(Event(0.0, "processing_off", self.proc_note, "auto"))
            return
        self.processing = processing
        try:
            self.vst3 = vst3.load_library()
        except OSError as e:
            LOG.exception("The plugin host library could not be loaded")
            self.vst3 = None
            self.log(Event(0.0, "plugins_off", repr(e), "auto"))
        names = list(self.cfg.talker_names)
        settings, labels = processing.settings_from_project(self.project, names)
        model = processing.model_dir()
        LOG.info("Processing: speech model %s, plugin host %s", model, getattr(self.vst3, "path", None))
        workers = self.a.workers
        if str(workers) == "0":                         # the old way: a thread in this process
            self.proc = processing.Processor(labels, settings, model=model, vst3_lib=self.vst3,
                                             track_delay=self._track_delay())
        else:
            # The audio engine stays in this process; the processing runs in worker processes, so
            # the two never compete for a core or for Python's lock, and a crash in one can't
            # freeze the other.
            from . import workers as W
            n = None if workers in (None, "", "auto") else int(workers)
            self.proc = W.Pool(labels, settings, model=model, workers=n, track_delay=self._track_delay(),
                               headroom=self.a.headroom)
        self.core.extra_health = self.proc.health
        self.log(Event(0.0, "processing_on", f"speech model {'loaded' if model else 'not found: echo removal only'}; "
                       f"plugin inserts {'available' if self.vst3 else 'not available'}", "auto"))
        self.apply_project(self.project or {}, save=False)

    def _track_delay(self):
        """PA delay tracking: on with a real frame, off with the simulated one, unless set."""
        t = getattr(self.a, "track_delay", "auto")
        return (t == "on") or (t == "auto" and not self.a.simulate)

    def apply_project(self, project, save=True):
        """Saves the console's project and applies it to the running chains: processing settings,
        operator names, and each operator's insert (loaded here, swapped in by the worker)."""
        self.project = project
        if save:
            try:
                os.makedirs(os.path.dirname(self.project_path), exist_ok=True)
                with open(self.project_path, "w") as f:
                    json.dump(project, f)
            except OSError as e:
                return {"ok": False, "error": f"could not save the project: {e}"}
        if self.proc is None:
            return {"ok": True, "applied": False, "why": self.proc_note}
        P = self.processing
        settings, labels = P.settings_from_project(project, self.cfg.talker_names)
        refused = []
        if self.vst3 is not None:                        # a plugin that crashes must never reach the processing
            from . import vst3
            for st, lab in zip(settings, labels):
                path = (st.insert or {}).get("path")
                if not path:
                    continue
                ok, why = vst3.trial(path)
                if not ok:
                    LOG.error("Insert %s refused for %s: %s", path, lab, why)
                    st.insert = {}
                    refused.append(f"{lab}: {why}")
        res = self.proc.apply(settings, labels)
        res = {**res, "errors": refused + list(res.get("errors", []))}
        for kind, detail in res.get("events", []):
            self.log(Event(self.core.t, kind, detail, "console"))
        errors = list(res.get("errors", []))
        if hasattr(self.proc, "set_headroom"):
            try:
                hr = int((project.get("proc") or {}).get("headroom") or self.a.headroom)
            except (TypeError, ValueError):
                hr = self.a.headroom
            if hr != self.proc.headroom:
                if self.core.arm.armed:              # it moves every talker's delay: never mid-show
                    errors.append("Disarm Sotto before changing the processing headroom: it moves every talker's delay.")
                else:
                    self.proc.set_headroom(hr)
                    self.log(Event(self.core.t, "headroom", f"{10 * hr} ms ({10 * (hr + 1)} ms through Sotto)", "console"))
        return {"ok": True, "applied": True, "errors": errors}

    def insert_info(self, talker):
        if self.proc is None or not 0 <= talker < self.proc.n:
            return {"loaded": False, "error": self.proc_note or "no such talker"}
        return self.proc.insert_info(talker)

    def plugin_param(self, talker, pid, norm):
        if self.proc is None or not 0 <= talker < self.proc.n:
            return {"ok": False, "error": "no such talker"}
        r = self.proc.plugin_param(talker, pid, norm)
        if r.get("ok"):
            self._plugin_states_dirty = True
        return r

    def _save_plugin_states(self, force=False):
        """Copies each running insert's settings into its operator profile, at most every 2 s."""
        if self.proc is None or not self._plugin_states_dirty:
            return
        if not force and time.time() - self._last_state_save < 2.0:
            return
        self._plugin_states_dirty = False
        self._last_state_save = time.time()
        proj = self.project or {}
        ops = {o.get("id"): o for o in proj.get("operators", [])}
        chs = proj.get("ch", [])
        try:
            states = self.proc.plugin_states()
        except Exception:
            LOG.exception("Could not read the inserts' settings")
            return
        for i, (path, data) in states.items():
            if i >= len(chs):
                continue
            op = ops.get(chs[i].get("op"))
            if not op:
                continue
            ins = op.setdefault("ins", {})
            ins.setdefault("vst3", {})[path] = self.processing.b64(data)
        try:
            with open(self.project_path, "w") as f:
                json.dump(proj, f)
        except OSError:
            LOG.exception("Could not save the project")

    def plugins(self, rescan=False):
        if self.plugins_cache is None or rescan:
            from . import vst3
            self.plugins_cache = vst3.scan()
        return self.plugins_cache

    # ------------------------------------------------------------ audio
    @staticmethod
    def _new_gaps():
        return {"n": 0, "min_ms": 1e9, "max_ms": 0.0, "bunched": 0, "late": 0,
                "dev_n": 0, "dev_max_ms": 0.0, "delay_min_ms": 1e9, "delay_max_ms": 0.0}

    def _device_timing(self, tinfo, now):
        """Per soundcard callback: the gap since the last one, and how long after the audio was
        captured the callback ran (PortAudio's time info). If that delay swings by tens of ms, the
        callback is being held up: by the driver, or by something in this process."""
        g = self.gaps
        if self._dev_last is not None:
            gap = 1000 * (now - self._dev_last)
            g["dev_n"] += 1
            if gap > g["dev_max_ms"]:
                g["dev_max_ms"] = gap
        self._dev_last = now
        if tinfo is not None:
            try:
                d = 1000 * (tinfo.currentTime - tinfo.inputBufferAdcTime)
            except Exception:
                return
            if 0 < d < 5000:
                if d < g["delay_min_ms"]:
                    g["delay_min_ms"] = d
                if d > g["delay_max_ms"]:
                    g["delay_max_ms"] = d

    def device_callback(self, indata, outdata, frames, tinfo, status):
        """The soundcard's callback when it runs on a small buffer: regroup into 10 ms blocks."""
        self._device_timing(tinfo, time.perf_counter())
        self._status = status
        try:
            self.regroup.process(indata, outdata, frames, self._regrouped_block)
        except Exception as e:                       # never leave the outputs undefined
            outdata.fill(0.0)
            self.callback_errors += 1
            if self._cb_exc is None:
                self._cb_exc = e

    def _regrouped_block(self, inb, outb):
        self.callback(inb, outb, len(inb), None, self._status)

    def callback(self, indata, outdata, frames, tinfo, status):
        """Sotto's engine for one block (10 ms): straight from the soundcard, or regrouped."""
        t0 = time.perf_counter()
        if self.regroup is None:
            self._device_timing(tinfo, t0)
        if self._cb_last is not None:                # time since the last block (should be 10 ms)
            gap = 1000 * (t0 - self._cb_last)
            g = self.gaps
            g["n"] += 1
            if gap < g["min_ms"]:
                g["min_ms"] = gap
            if gap > g["max_ms"]:
                g["max_ms"] = gap
            if gap < 3.0:
                g["bunched"] += 1
            elif gap > 17.0:
                g["late"] += 1
        self._cb_last = t0
        try:
            if self.tests.plan and not self.core.arm.armed:    # test sources: never while armed
                indata = self.tests.apply(indata, frames)
            else:
                self.tests.clock += frames
            if self.proc is not None and frames == 480:
                cfg = self.cfg
                mics = np.asarray(indata[:, cfg.talker_in], dtype=np.float64).T
                ref = indata[:, cfg.pgm_in] if cfg.pgm_in is not None else np.zeros(frames)
                out, dry = self.proc.exchange(mics, ref)
                with self.lock:
                    self.core.process(indata, outdata, status, returns=out.T, dry=dry.T)
                    for i, why in self.proc.drain_faults():
                        self.core.arm.trip_talker(self.core.t, i, why, "auto")
                        for e in self.core.arm.drain_events():
                            self.core.log.append(e)
            else:
                with self.lock:
                    self.core.process(indata, outdata, status)
            if self.local_phones.stream is not None and self.cfg.monitor_out:
                mon = self.cfg.monitor_out
                self.local_phones.push(outdata[:, mon[0]], outdata[:, mon[-1]])
            if self.rec.on:
                cfg = self.cfg
                cols = [indata[:, c] for c in cfg.talker_in + cfg.probe_in]
                if cfg.listen_in is not None:
                    cols.append(indata[:, cfg.listen_in])
                cols += [outdata[:, cfg.heartbeat_out], outdata[:, cfg.talker_out[0]]]
                self.rec.push(np.stack(cols, axis=1))
        except Exception as e:                       # never leave the outputs undefined
            outdata.fill(0.0)
            self.callback_errors += 1
            if self._cb_exc is None:
                self._cb_exc = e                         # logged with its traceback by the main loop
                self.core.log.append(Event(self.core.t, "callback_error", repr(e), "auto"))
        dt = 1000 * (time.perf_counter() - t0)
        self.cb_ms = 0.98 * self.cb_ms + 0.02 * dt
        self.cb_ms_max = max(self.cb_ms_max * 0.999, dt)

    def run(self):
        a, cfg = self.a, self.cfg
        clash = channel_clashes(cfg)
        if clash:
            raise SystemExit(clash)
        if not a.simulate:
            import sounddevice as sd
        self.app_nap = macos.keep_responsive()[1]
        n_in = max(cfg.talker_in + cfg.probe_in + [c for c in (cfg.listen_in, cfg.pgm_in) if c is not None]) + 1
        n_out = max(cfg.talker_out + [cfg.probe_out, cfg.heartbeat_out] + cfg.monitor_out) + 1
        dev = int(a.device) if a.device and str(a.device).isdigit() else a.device
        n_out = max(n_out, 2)
        lat = a.latency
        try:
            lat = float(lat)
        except ValueError:
            pass
        if a.io_buffer and a.io_buffer != a.block and a.latency == "low":
            # Ask for a Core Audio buffer of exactly one chunk: with 'low', PortAudio may pick a
            # multiple of it (320 for 160), and the chunks then arrive in pairs.
            lat = a.io_buffer / cfg.fs
        # The audio callback is Python: when another thread holds Python's lock, the callback
        # waits up to the switch interval (5 ms by default) each time it needs the lock back.
        sys.setswitchinterval(a.switch_interval / 1000.0)
        LOG.info("Python switch interval %.2f ms", a.switch_interval)
        LOG.info("App Nap: %s", self.app_nap)
        try:
            self._start_processing()
            self._start_web()
            if self.rrcs:
                self.rrcs.start()
            if a.record:
                self.command({"cmd": "rec", "on": True, "source": "startup"})
            self.log(Event(0.0, "started", f"take {self.take}; starts DISARMED", "auto"))
            if a.io_buffer and a.io_buffer != a.block:
                from .regroup import Regrouper
                self.regroup = Regrouper(n_in, n_out, a.block, a.io_buffer)
                LOG.info("Soundcard buffer %d samples, regrouped into %d-sample blocks (adds %.1f ms)",
                         a.io_buffer, a.block, self.regroup.added_ms)
            io = a.io_buffer if self.regroup else a.block
            cb = self.device_callback if self.regroup else self.callback
            if a.simulate:
                from .sim import SimRiedel, SimStream
                self.sim = SimRiedel(cfg, vox_hold_ms=a.sim_hold, model_armed=True)
                stream = SimStream(self.sim, n_in, n_out, io, cb)
                print(f"SIMULATION: virtual Riedel frame, VOX hold {a.sim_hold:.0f} ms, no audio hardware used")
            else:
                try:
                    LOG.info("Opening soundcard %r: %d in / %d out at %d Hz, block %d, latency %r. Device reports: %s",
                             dev, n_in, n_out, cfg.fs, a.block, lat, sd.query_devices(dev))
                except Exception:
                    LOG.exception("Could not read the soundcard's details before opening it")
                stream = sd.Stream(device=dev, samplerate=cfg.fs, blocksize=io, channels=(n_in, n_out),
                                   dtype="float32", latency=lat, callback=cb)
            with stream as st:
                il, ol = st.latency if isinstance(st.latency, tuple) else (0.0, st.latency)
                self.latency_note = f"interface in {1000 * il:.1f} ms + out {1000 * ol:.1f} ms"
                name = "simulated"
                if not a.simulate:
                    try:
                        name = sd.query_devices(st.device[0] if isinstance(st.device, (tuple, list)) else st.device)["name"]
                    except Exception:
                        name = str(a.device)
                self.engine_info = {"device": "simulated" if a.simulate else str(a.device), "device_name": name,
                                    "app_nap": self.app_nap, "simulate": bool(a.simulate),
                                    "fs": cfg.fs, "block": a.block, "n_in": n_in, "n_out": n_out,
                                    "latency_note": self.latency_note,
                                    "io_buffer": io, "regroup_ms": round(self.regroup.added_ms, 2) if self.regroup else 0.0,
                                    "pa_latency_ms": {"in": round(1000 * il, 2), "out": round(1000 * ol, 2)}}
                if not a.simulate:
                    self._read_coreaudio(name)
                    self._reopen_local_phones()
                LOG.info("Soundcard open: %s (%s)", name, self.latency_note)
                diagnostics.last_crash(clear=True)       # started fine: the setup page stops showing it
                print(f"Running ({self.latency_note}). Console: http://{a.host}:{a.port}   Ctrl+C to stop")
                print(f"Events and recordings: {self.dir}")
                try:
                    next_power = next_stats = 0.0
                    while not self.quit_ev.is_set():
                        time.sleep(0.05)
                        self._drain()
                        self._save_plugin_states()
                        now = time.monotonic()
                        if now >= next_power:              # Low Power Mode, heat, battery
                            self.power = macos.power_state()
                            next_power = now + 10
                        if now >= next_stats:
                            self._log_stats()
                            next_stats = now + 10
                except KeyboardInterrupt:
                    print("\nStopping ...")
        except Exception:
            LOG.exception("The engine stopped with an error")
            raise
        finally:
            self._shutdown()

    def _read_coreaudio(self, name):
        """What Core Audio says the device is doing now (buffer, latency, safety offsets)."""
        try:
            from . import coreaudio
            self.coreaudio = coreaudio.find(name)
            if self.coreaudio:
                LOG.info("Core Audio, %s: %s", name, self.coreaudio)
        except Exception:
            LOG.exception("Could not read Core Audio's view of %s", name)

    def _log_stats(self):
        """One line every 10 s: enough to see from the log when and how things went wrong."""
        g, self.gaps = self.gaps, self._new_gaps()
        if g["n"]:
            self.gaps_last = g
            LOG.info("block timing (last 10 s): %d blocks, gap %.2f to %.2f ms, %d bunched (<3 ms), %d late (>17 ms) | "
                     "soundcard callbacks %d, longest gap %.2f ms, run %.2f to %.2f ms after capture",
                     g["n"], g["min_ms"], g["max_ms"], g["bunched"], g["late"], g["dev_n"], g["dev_max_ms"],
                     g["delay_min_ms"] if g["delay_max_ms"] else 0, g["delay_max_ms"])
        try:
            st = self.core.state()
            pr = self.proc.state() if self.proc is not None else {}
            lv_in = st.get("levels_in") or [-120]
            LOG.info("stats: armed=%s health=%r | callback %.2f ms (max %.2f), errors %d, xruns %s | processing %s ms "
                     "(cpu %s, stages %s, max %s) misses %s ok=%s %s | loudest input %.0f dB | power %s",
                     st.get("armed"), st.get("health"), self.cb_ms, self.cb_ms_max, self.callback_errors, st.get("xruns"),
                     pr.get("proc_ms"), pr.get("cpu_ms"), pr.get("stage_ms"), pr.get("proc_ms_max"), pr.get("misses"),
                     pr.get("ok"), pr.get("why", ""), max(lv_in), getattr(self, "power", {}))
        except Exception:
            LOG.exception("Could not write the stats line")

    def _shutdown(self):
        """Stops everything that run() started, whether it ended normally or with an error."""
        try:
            self._drain()
            self.log(Event(self.core.t, "stopped", "", "auto"))
        except Exception:
            LOG.exception("Could not write the last events")
        for what, fn in (("recorder", self.rec.stop),
                         ("plugin settings", lambda: self.proc is not None and self._save_plugin_states(force=True)),
                         ("processing", lambda: self.proc is not None and self.proc.stop()),
                         ("RRCS", lambda: self.rrcs and (self.rrcs.set_armed_safe(False), self.rrcs.stop())),
                         ("phones on this computer", self.local_phones.close),
                         ("event log", self.events_f.close)):
            try:
                fn()
            except Exception:
                LOG.exception("Could not stop the %s cleanly", what)
        if self.srv:
            self.srv.shutdown()
            self.srv.server_close()
            self.srv = None
        LOG.info("Engine stopped")

    IO_BUFFERS = (0, 64, 96, 128, 160, 192, 240, 256, 320, 480, 512)

    def switch_device(self, device=None, io_buffer=None):
        """Restarts on another soundcard and/or soundcard buffer. The new engine starts disarmed,
        like any start, so the frame falls back to dry until someone re-arms."""
        if self.a.simulate:
            return {"ok": False, "error": "The simulated frame has no soundcard. Quit and choose a soundcard on the setup page."}
        changes, saved_changes, what = {}, {}, []
        if device is not None and str(device) != str(self.a.device):
            devs = device_list()
            if not devs["ok"]:
                return {"ok": False, "error": "Could not list audio devices: " + devs["error"]}
            d = next((x for x in devs["devices"] if str(x["index"]) == str(device)), None)
            if d is None:
                return {"ok": False, "error": "That device is no longer there."}
            need_in, need_out = self.engine_info["n_in"], self.engine_info["n_out"]
            if d["inputs"] < need_in or d["outputs"] < need_out:
                return {"ok": False, "error": f"{d['name']} has {d['inputs']} inputs and {d['outputs']} outputs. "
                                              f"These channel settings need {need_in} in and {need_out} out."}
            changes["--device"] = saved_changes["device"] = str(d["index"])
            what.append(d["name"])
        if io_buffer is not None:
            try:
                io = int(io_buffer)
            except (TypeError, ValueError):
                return {"ok": False, "error": "The soundcard buffer must be a number of samples."}
            if io not in self.IO_BUFFERS:
                return {"ok": False, "error": f"Choose one of {', '.join(str(x) for x in self.IO_BUFFERS if x)} samples."}
            rng = (self.coreaudio or {}).get("buffer_range")
            if io and rng and not rng[0] <= io <= rng[1]:
                return {"ok": False, "error": f"This soundcard allows buffers of {rng[0]} to {rng[1]} samples."}
            if io != (self.a.io_buffer or 0):
                changes["--io-buffer"] = saved_changes["io_buffer"] = str(io)
                what.append(f"a {io}-sample buffer" if io else "10 ms blocks from the soundcard")
        if not changes:
            return {"ok": False, "error": "That's what Sotto is already using."}
        self.log(Event(self.core.t, "engine_change", "restarting on " + " with ".join(what), "console"))
        try:                                             # remembered for the next launch too
            path = os.path.join(config_dir(), "config.json")
            with open(path) as f:
                saved = json.load(f)
            saved.update(saved_changes)
            with open(path, "w") as f:
                json.dump(saved, f, indent=2)
        except (OSError, ValueError):
            LOG.exception("Could not save the soundcard choice")
        self.restart_with = changes
        self.restart_device = changes.get("--device")
        self.quit_ev.set()
        return {"ok": True, "device": " with ".join(what)}

    # ------------------------------------------------------------ engine settings and the I/O patch
    PATCH_KEYS = ("talker_in", "talker_out", "names", "probe_in", "probe_out", "heartbeat_out", "pgm_in", "monitor",
                  "listen_in")
    ENGINE_KEYS = ("device", "io_buffer", "rrcs_host", "rrcs_port")

    def config(self):
        from . import launcher as L
        saved = L.load_config()
        return {"saved": saved, "defaults": {k: L.DEFAULTS[k] for k in self.PATCH_KEYS + self.ENGINE_KEYS if k in L.DEFAULTS},
                "running": {"device": self.a.device, "io_buffer": self.a.io_buffer or 0, "simulate": self.a.simulate,
                            **{k: getattr(self.a, k) for k in self.PATCH_KEYS}},
                "io_buffers": list(self.IO_BUFFERS)}

    def reconfigure(self, changes):
        """Checks a change to the I/O patch or the engine settings, saves it and restarts the engine
        with it. Sotto starts again disarmed, so the frame falls back to dry until someone re-arms."""
        from . import launcher as L
        allowed = set(self.PATCH_KEYS + self.ENGINE_KEYS)
        bad = [k for k in changes if k not in allowed]
        if bad:
            return {"ok": False, "error": f"Unknown settings: {', '.join(bad)}"}
        cfg = L.load_config()
        cfg.update({k: ("" if v is None else str(v).strip()) for k, v in changes.items()})
        # Start from this run's own options (port, folder, processing, ...) and change only what
        # the console changed, using the saved setup choices for the patch and the engine.
        cur = argv_of(self.a)
        new = {"--talker-in": cfg["talker_in"], "--talker-out": cfg["talker_out"], "--names": cfg.get("names", ""),
               "--probe-in": cfg["probe_in"], "--probe-out": cfg["probe_out"], "--heartbeat-out": cfg["heartbeat_out"],
               "--pgm-in": cfg.get("pgm_in", ""), "--monitor": cfg.get("monitor", ""), "--listen-in": cfg.get("listen_in", "")}
        if not self.a.simulate:
            new.update({"--device": cfg.get("device", ""), "--io-buffer": str(cfg.get("io_buffer") or 0),
                        "--rrcs-host": cfg.get("rrcs_host", ""), "--rrcs-port": str(cfg.get("rrcs_port") or 8193)})
        argv = []
        skip = False
        for k, x in enumerate(cur):
            if skip:
                skip = False
                continue
            if x in new:
                skip = True
                continue
            argv.append(x)
        for opt, val in new.items():
            if val not in (None, ""):
                argv += [opt, str(val)]
        try:
            a2 = parse_args(argv)
            c2 = config_from_args(a2)
        except SystemExit as e:
            return {"ok": False, "error": str(e) or "Those channel settings don't work together."}
        except (ValueError, IndexError):
            return {"ok": False, "error": "Channels are numbers, ranges like 1-4, or lists like 63,64."}
        clash = channel_clashes(c2)
        if clash:
            return {"ok": False, "error": clash}
        n_in = max(c2.talker_in + c2.probe_in + [c for c in (c2.listen_in, c2.pgm_in) if c is not None]) + 1
        n_out = max(c2.talker_out + [c2.probe_out, c2.heartbeat_out] + c2.monitor_out) + 1
        if not a2.simulate:
            devs = device_list()
            d = next((x for x in devs.get("devices", []) if str(x["index"]) == str(a2.device)), None)
            if d is None:
                return {"ok": False, "error": "That soundcard isn't there. Press Look again."}
            if d["inputs"] < n_in or d["outputs"] < n_out:
                return {"ok": False, "error": f"{d['name']} has {d['inputs']} inputs and {d['outputs']} outputs; "
                                              f"this patch needs {n_in} in and {n_out} out."}
        L.save_config(cfg)
        self.log(Event(self.core.t, "engine_change", "restarting with new settings: " +
                       ", ".join(f"{k}={v}" for k, v in changes.items()), "console"))
        self.restart_argv = argv
        self.quit_ev.set()
        return {"ok": True}

    def back_to_setup(self):
        self.log(Event(self.core.t, "to_setup", "engine stopped: back to the setup page", "console"))
        self.to_setup = True
        self.quit_ev.set()
        return {"ok": True}

    def check_device(self):
        """A real check of the soundcard and the patch, for the console's Check device button."""
        out = []
        st = self.core.state()
        e = getattr(self, "engine_info", None) or {}
        if e.get("simulate"):
            out.append(["ok", "Running on the simulated frame (no soundcard)."])
        else:
            d = next((x for x in device_list().get("devices", []) if str(x["index"]) == str(self.a.device)), None)
            if d is None:
                out.append(["err", "The soundcard isn't in the device list any more."])
            else:
                ok = d["inputs"] >= e.get("n_in", 0) and d["outputs"] >= e.get("n_out", 0)
                out.append(["ok" if ok else "err", f"{d['name']}: {d['inputs']} inputs and {d['outputs']} outputs "
                            f"({e.get('n_in')} in and {e.get('n_out')} out in use)."])
            ca = self.coreaudio or {}
            if ca:
                rate = ca.get("rate")
                out.append(["ok" if rate == 48000 else "err",
                            f"Sample rate {rate / 1000 if rate else '?'} kHz" + ("." if rate == 48000 else
                            ": set Dante Virtual Soundcard to 48 kHz in Dante Controller (the speech model needs it).")])
                out.append(["ok", f"Core Audio buffer {ca.get('buffer_frames')} samples (allowed {ca.get('buffer_range')}); "
                            f"Core Audio adds {ca.get('total_ms', {}).get('in')} ms in and {ca.get('total_ms', {}).get('out')} ms out."])
                if ca.get("running_elsewhere"):
                    out.append(["warn", "Another app is also using this soundcard; it may change its buffer."])
            if sys.platform == "darwin":
                perm = macos.mic_permission()
                out.append(["ok" if perm == "allowed" else "err",
                            "Microphone permission: " + {"allowed": "allowed.", "denied": "denied. Allow Sotto Bench in System Settings, Privacy & Security, Microphone.",
                                                         "not asked": "not asked yet (macOS asks when the soundcard opens).",
                                                         "restricted": "restricted by this Mac's management profile."}.get(perm, perm)])
        g = (self.gaps_last or self.gaps)
        out.append(["ok" if not st.get("xruns") else "warn", f"{st.get('xruns', 0)} dropouts reported by the soundcard; "
                    f"blocks arrive {round(g['min_ms'], 1) if g['n'] else '?'} to {round(g['max_ms'], 1)} ms apart"
                    f"{', ' + str(g['bunched']) + ' bunched' if g['bunched'] else ''}."])
        pr = st.get("probes") or []
        pok = pr and all(p["ok"] for p in pr)
        out.append(["ok" if pok else "err", "PROBE comes back on IN " + ", ".join(str(c + 1) for c in self.cfg.probe_in) +
                    (" in time." if pok else f": missing. Patch OUT {self.cfg.probe_out + 1} to IN {self.cfg.probe_in[0] + 1} "
                     "through the frame (or loop it in Dante Controller for a bench test).")])
        lv = st.get("levels_in") or []
        if self.cfg.pgm_in is not None and len(lv) > self.cfg.pgm_in:
            p = lv[self.cfg.pgm_in]
            out.append(["ok" if p > -60 else "warn", f"PGM on IN {self.cfg.pgm_in + 1} at {p:.0f} dBFS" +
                        ("." if p > -60 else ": nothing there. Route the programme feed to it.")])
        talk = [lv[c] for c in self.cfg.talker_in if c < len(lv)]
        if talk:
            live = sum(1 for v in talk if v > -70)
            out.append(["ok" if live else "warn", f"{live} of {len(talk)} talker inputs have signal" +
                        ("." if live else ": route mics to them, or play a test source.")])
        out.append(["ok" if not getattr(self, "callback_errors", 0) else "err", f"{self.callback_errors} audio errors."])
        return {"ok": True, "checks": out}

    # ------------------------------------------------------------ test sources, phones on this computer
    def test_source(self, b):
        if self.core.arm.armed:
            return {"ok": False, "error": "Disarm Sotto first: test sources only play while disarmed."}
        try:
            if b.get("remove"):
                r = self.tests.remove(b["remove"])
                self.log(Event(self.core.t, "test_file", f"removed {r['removed']}", "console"))
                return {"ok": True, **r, **self.tests.state()}
            if b.get("all_off"):
                self.tests.off()
                self.log(Event(self.core.t, "test_sources_off", "all", "console"))
                return {"ok": True, **self.tests.state()}
            ch = int(b.get("channel")) - 1
            kind = b.get("kind")
            if kind in (None, "", "off"):
                self.tests.off(ch)
                self.log(Event(self.core.t, "test_source", f"input {ch + 1}: off", "console"))
                return {"ok": True, **self.tests.state()}
            d = self.tests.set(ch, kind, float(b.get("level_db", -20)), b.get("file") or None,
                               float(b.get("bleed_db", -6)), b.get("programme_file") or None)
            self.log(Event(self.core.t, "test_source", f"input {ch + 1}: {d['what']}", "console"))
            return {"ok": True, **self.tests.state()}
        except Exception as e:
            LOG.exception("Test source %s failed", b)
            return {"ok": False, "error": str(e)}

    def test_upload(self, name, data):
        try:
            r = self.tests.load(name, data)
            self.log(Event(self.core.t, "test_file", f"{r['name']} ({r['seconds']} s)", "console"))
            return {"ok": True, **r, **self.tests.state()}
        except Exception as e:
            LOG.exception("Could not load %s", name)
            return {"ok": False, "error": f"Couldn't read that file: {e}. Use a WAV file."}

    def phones_local(self, device):
        try:
            r = self.local_phones.open(None if device in (None, "", "off") else int(device))
        except Exception as e:
            LOG.exception("Phones on this computer: could not open %s", device)
            return {"ok": False, "error": f"Couldn't play on that device: {e}"}
        try:                                              # remembered by name for the next start
            path = os.path.join(config_dir(), "config.json")
            with open(path) as f:
                saved = json.load(f)
            saved["local_phones"] = r.get("name") or ""
            with open(path, "w") as f:
                json.dump(saved, f, indent=2)
        except (OSError, ValueError):
            pass
        return {**r, "state": self.local_phones.state()}

    def _reopen_local_phones(self):
        try:
            with open(os.path.join(config_dir(), "config.json")) as f:
                name = json.load(f).get("local_phones")
        except (OSError, ValueError):
            return
        if not name or self.a.simulate:
            return
        d = next((x for x in device_list().get("devices", []) if x["name"] == name and x["outputs"] > 0), None)
        if d:
            try:
                self.local_phones.open(d["index"])
            except Exception:
                LOG.exception("Phones on this computer: could not reopen %s", name)

    def _start_web(self):
        app = self
        with open(os.path.join(HERE, "console.html"), "rb") as f:
            page = f.read()
        with open(os.path.join(HERE, "console_bench.html"), "rb") as f:
            bench_page = f.read()

        class H(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                try:
                    self._get()
                except Exception as e:
                    self._fail(e)

            def do_POST(self):
                try:
                    self._post()
                except Exception as e:
                    self._fail(e)

            def _fail(self, e):
                LOG.exception("HTTP %s %s failed", self.command, self.path)
                try:
                    self._send(500, json.dumps({"ok": False, "error": f"Sotto Bench hit an error: {e!r} (details in the log)"}).encode(),
                               "application/json")
                except Exception:
                    pass

            def _get(self):
                if self.path in ("/", "/index.html"):
                    self._send(200, page, "text/html; charset=utf-8")
                elif self.path == "/bench":                  # classic bench console: recording and fault timing
                    self._send(200, bench_page, "text/html; charset=utf-8")
                elif self.path in ("/state", "/api/state"):
                    self._send(200, _json(app.state()).encode(), "application/json")
                elif self.path == "/api/project":
                    self._send(200, json.dumps({"project": app.project, "path": app.project_path}).encode(), "application/json")
                elif self.path.startswith("/api/plugins"):
                    self._send(200, json.dumps(app.plugins(rescan="rescan" in self.path)).encode(), "application/json")
                elif self.path == "/api/devices":
                    self._send(200, json.dumps(device_list()).encode(), "application/json")
                elif self.path == "/api/testsrc":
                    self._send(200, _json(app.tests.state()).encode(), "application/json")
                elif self.path == "/api/config":
                    self._send(200, _json(app.config()).encode(), "application/json")
                elif self.path == "/api/check":
                    self._send(200, _json(app.check_device()).encode(), "application/json")
                elif self.path.startswith("/api/diagnostics"):
                    # The speed test loads the computer for ~15 s: never while armed (it could make
                    # processing late and disarm Sotto mid-show).
                    full = "quick" not in self.path and not app.core.arm.armed
                    text = diagnostics.build_report(app, run_speed_test=full)
                    diagnostics.save_report(text, app.dir)
                    name = diagnostics.report_filename()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(text.encode("utf-8"))
                elif self.path.startswith("/api/insert"):
                    from urllib.parse import urlparse, parse_qs
                    q = parse_qs(urlparse(self.path).query)
                    self._send(200, _json(app.insert_info(int(q.get("talker", ["0"])[0]))).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def _post(self):
                n = int(self.headers.get("Content-Length", 0))
                if self.path.startswith("/api/testsrc/upload"):      # a WAV file, as the body
                    from urllib.parse import parse_qs, urlparse
                    if n > 60_000_000:
                        return self._send(200, b'{"ok": false, "error": "That file is too big (60 MB at most)."}', "application/json")
                    name = parse_qs(urlparse(self.path).query).get("name", ["recording.wav"])[0]
                    return self._send(200, _json(app.test_upload(name, self.rfile.read(n))).encode(), "application/json")
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    body = {}
                if self.path == "/cmd":
                    res = app.command(body)
                elif self.path == "/api/arm":            # Companion: {"talker": 1-4 or omitted}
                    t = body.get("talker")
                    res = app.arm(None if t in (None, "", "all") else int(t) - 1, "companion")
                elif self.path == "/api/disarm":
                    res = app.command({"cmd": "disarm", "source": "companion"})
                elif self.path == "/api/project":
                    res = app.apply_project(body if isinstance(body, dict) else {})
                elif self.path == "/quit":
                    app.quit_ev.set()
                    res = {"ok": True}
                elif self.path == "/api/testsrc":
                    res = app.test_source(body if isinstance(body, dict) else {})
                elif self.path == "/api/config":
                    res = app.reconfigure(body if isinstance(body, dict) else {})
                elif self.path == "/api/setup":
                    res = app.back_to_setup()
                elif self.path == "/api/phones":
                    res = app.phones_local((body or {}).get("device") if isinstance(body, dict) else None)
                elif self.path == "/api/engine":
                    b = body if isinstance(body, dict) else {}
                    res = app.switch_device(b.get("device"), b.get("io_buffer"))
                else:
                    return self._send(404, b"not found", "text/plain")
                self._send(200, _json(res).encode(), "application/json")

        self.srv = ThreadingHTTPServer((self.a.host, self.a.port), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


def _json(obj):
    """JSON for the console; numpy numbers become plain numbers."""
    return json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def argv_of(a):
    """The command line that gives these parsed options (to restart with a few of them changed)."""
    out = []
    for act in build_parser()._actions:
        if not act.option_strings or act.dest == "help":
            continue
        v = getattr(a, act.dest, None)
        opt = act.option_strings[-1]
        if act.nargs == 0:                               # a flag
            if v:
                out.append(opt)
        elif v not in (None, ""):
            out += [opt, str(v)]
    return out


def channel_clashes(cfg):
    """Plain-English message if one soundcard channel is given two jobs, else None."""
    used = {}
    ins = [(c, f"talker {i + 1} input") for i, c in enumerate(cfg.talker_in)]
    ins += [(c, "PROBE return") for c in cfg.probe_in]
    ins += [(c, n) for c, n in ((cfg.pgm_in, "PGM"), (cfg.listen_in, "listen input")) if c is not None]
    outs = [(c, f"talker {i + 1} output") for i, c in enumerate(cfg.talker_out)]
    outs += [(cfg.probe_out, "PROBE send"), (cfg.heartbeat_out, "HEARTBEAT")]
    outs += [(c, "headphones") for c in cfg.monitor_out]
    for side, items in (("Input", ins), ("Output", outs)):
        used = {}
        for c, what in items:
            if c in used and not (what == "headphones" and used[c] == "headphones"):
                return f"{side} {c + 1} is set as both {used[c]} and {what}. Give each job its own channel."
            used[c] = what
    return None


def device_list():
    """Every audio device, for the setup page and the console."""
    try:
        import sounddevice as sd
        apis = sd.query_hostapis()
        return {"ok": True, "error": None, "devices": [
            {"index": i, "name": d["name"], "inputs": d["max_input_channels"], "outputs": d["max_output_channels"],
             "api": apis[d["hostapi"]]["name"]} for i, d in enumerate(sd.query_devices())]}
    except Exception as e:                       # no PortAudio, no permission, ...
        return {"ok": False, "error": repr(e), "devices": []}


def list_devices():
    import sounddevice as sd
    apis = sd.query_hostapis()
    print("Index  In  Out  Host API            Name")
    for i, d in enumerate(sd.query_devices()):
        print(f"{i:5d} {d['max_input_channels']:3d} {d['max_output_channels']:4d}  "
              f"{apis[d['hostapi']]['name'][:18]:18s}  {d['name']}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list audio devices and exit")
    p.add_argument("--device", help="virtual soundcard: index or part of its name")
    p.add_argument("--talker-in", default="1-4", help="inputs carrying the talkers (default 1-4)")
    p.add_argument("--talker-out", default="1-4", help="outputs for the talker returns (default 1-4)")
    p.add_argument("--names", default="caller,creative,tech mgr,stage mgr")
    p.add_argument("--probe-in", default="61", help="PROBE return input(s), one per stream (default 61)")
    p.add_argument("--probe-out", default="62", help="PROBE output (default 62)")
    p.add_argument("--heartbeat-out", default="61", help="HEARTBEAT output (default 61)")
    p.add_argument("--pgm-in", default="62", help="PGM reference input, metered (default 62; empty for none)")
    p.add_argument("--listen-in", help="input carrying a listener's feed, for timing the fallback (e.g. 63)")
    p.add_argument("--monitor", default="63,64", help="headphone outputs carrying the solo bus (default 63,64)")
    p.add_argument("--arm-hold", type=float, default=5.0, help="seconds of green before arming is allowed")
    p.add_argument("--probe-timeout", type=float, default=200.0, help="ms of missing PROBE before disarming")
    p.add_argument("--no-marker", action="store_true", help="don't add the marker tone to the talker returns")
    p.add_argument("--block", type=int, default=480, help="audio block in samples (480 = 10 ms; processing needs 480)")
    p.add_argument("--no-processing", action="store_true", help="pass the talkers straight through (no echo removal, model or inserts)")
    p.add_argument("--workers", default="auto",
                   help="processing worker processes: auto (up to 3 talkers each), a number, or 0 for a thread in this process")
    p.add_argument("--io-buffer", type=int, default=0,
                   help="soundcard buffer in samples (e.g. 128, 160, 256); Sotto regroups into 10 ms blocks. "
                        "0 = ask the soundcard for 10 ms blocks directly (as before)")
    p.add_argument("--switch-interval", type=float, default=0.5,
                   help="ms another Python thread may hold the lock before the audio callback gets it back (Python's default is 5)")
    p.add_argument("--headroom", type=int, default=1, choices=[1, 2, 3],
                   help="blocks of time the processing workers get per block (1 = 20 ms through Sotto, 2 = 30 ms)")
    p.add_argument("--track-delay", choices=["auto", "on", "off"], default="auto",
                   help="measure the PA delay into each talker's mic (auto: on unless --simulate)")
    p.add_argument("--project", help="project file shared with the console (default: in the app's settings folder)")
    p.add_argument("--latency", default="low")
    p.add_argument("--record", action="store_true", help="record from the start")
    p.add_argument("--out-dir", default=os.path.join(os.getcwd(), "bench_runs"))
    p.add_argument("--rrcs-host", help="RRCS gateway address, to read the panel ARM key and set ARMED")
    p.add_argument("--rrcs-port", type=int, default=8193)
    p.add_argument("--rrcs-armed", default="ARMED", help="logic source the app sets while armed")
    p.add_argument("--rrcs-arm-request", default="ARM_REQ", help="logic source the panel ARM key sets")
    p.add_argument("--rrcs-clean-ok", default="CLEAN_OK", help="logic source the HEARTBEAT VOX sets (read only)")
    p.add_argument("--simulate", action="store_true", help="no hardware: run against a simulated Riedel frame")
    p.add_argument("--sim-hold", type=float, default=400.0, help="simulated VOX hold time in ms")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open the console from another device")
    p.add_argument("--port", type=int, default=8766)
    return p


def parse_args(argv=None):
    a = build_parser().parse_args(argv)
    if a.simulate and not a.listen_in:
        a.listen_in = "63"
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.list:
        return list_devices()
    BenchApp(a).run()


if __name__ == "__main__":
    sys.exit(main())
