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

from .arming import Event
from .core import Config, FailsafeCore

HERE = os.path.dirname(os.path.abspath(__file__))


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


class BenchApp:
    def __init__(self, a):
        self.a = a
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
        self.srv = None
        self.timing = []
        self.sim = None

    # ------------------------------------------------------------ events
    def log(self, e: Event):
        row = [f"{e.t:.4f}", datetime.now().strftime("%H:%M:%S.%f")[:-3], e.kind, e.detail, e.source]
        self.events.writerow(row)
        self.events_f.flush()
        self.recent = (self.recent + [row])[-60:]

    def _drain(self):
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
                plan = core.set_solo(c.get("ids"), c.get("mode"), c.get("level_db"), c.get("dim"), c.get("dim_db"))
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
        return s

    # ------------------------------------------------------------ audio
    def callback(self, indata, outdata, frames, tinfo, status):
        try:
            with self.lock:
                self.core.process(indata, outdata, status)
            if self.rec.on:
                cfg = self.cfg
                cols = [indata[:, c] for c in cfg.talker_in + cfg.probe_in]
                if cfg.listen_in is not None:
                    cols.append(indata[:, cfg.listen_in])
                cols += [outdata[:, cfg.heartbeat_out], outdata[:, cfg.talker_out[0]]]
                self.rec.push(np.stack(cols, axis=1))
        except Exception as e:                       # never leave the outputs undefined
            outdata.fill(0.0)
            self.core.log.append(Event(self.core.t, "callback_error", repr(e), "auto"))

    def run(self):
        a, cfg = self.a, self.cfg
        if not a.simulate:
            import sounddevice as sd
        n_in = max(cfg.talker_in + cfg.probe_in + [c for c in (cfg.listen_in, cfg.pgm_in) if c is not None]) + 1
        n_out = max(cfg.talker_out + [cfg.probe_out, cfg.heartbeat_out] + cfg.monitor_out) + 1
        dev = int(a.device) if a.device and str(a.device).isdigit() else a.device
        n_out = max(n_out, 2)
        lat = a.latency
        try:
            lat = float(lat)
        except ValueError:
            pass
        self._start_web()
        if self.rrcs:
            self.rrcs.start()
        if a.record:
            self.command({"cmd": "rec", "on": True, "source": "startup"})
        self.log(Event(0.0, "started", f"take {self.take}; starts DISARMED", "auto"))
        if a.simulate:
            from .sim import SimRiedel, SimStream
            self.sim = SimRiedel(cfg, vox_hold_ms=a.sim_hold, model_armed=True)
            stream = SimStream(self.sim, n_in, n_out, a.block, self.callback)
            print(f"SIMULATION: virtual Riedel frame, VOX hold {a.sim_hold:.0f} ms, no audio hardware used")
        else:
            stream = sd.Stream(device=dev, samplerate=cfg.fs, blocksize=a.block, channels=(n_in, n_out),
                               dtype="float32", latency=lat, callback=self.callback)
        with stream as st:
            il, ol = st.latency if isinstance(st.latency, tuple) else (0.0, st.latency)
            self.latency_note = f"interface in {1000 * il:.1f} ms + out {1000 * ol:.1f} ms"
            self.engine_info = {"device": "simulated" if a.simulate else str(a.device), "simulate": bool(a.simulate),
                                "fs": cfg.fs, "block": a.block, "n_in": n_in, "n_out": n_out,
                                "latency_note": self.latency_note}
            print(f"Running ({self.latency_note}). Console: http://{a.host}:{a.port}   Ctrl+C to stop")
            print(f"Events and recordings: {self.dir}")
            try:
                while not self.quit_ev.is_set():
                    time.sleep(0.05)
                    self._drain()
            except KeyboardInterrupt:
                print("\nStopping ...")
        self._drain()
        self.log(Event(self.core.t, "stopped", "", "auto"))
        self.rec.stop()
        if self.rrcs:
            self.rrcs.set_armed_safe(False)
            self.rrcs.stop()
        self.events_f.close()
        if self.srv:
            self.srv.shutdown()
            self.srv.server_close()

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
                if self.path in ("/", "/index.html"):
                    self._send(200, page, "text/html; charset=utf-8")
                elif self.path == "/bench":                  # classic bench console: recording and fault timing
                    self._send(200, bench_page, "text/html; charset=utf-8")
                elif self.path in ("/state", "/api/state"):
                    self._send(200, json.dumps(app.state()).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
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
                elif self.path == "/quit":
                    app.quit_ev.set()
                    res = {"ok": True}
                else:
                    return self._send(404, b"not found", "text/plain")
                self._send(200, json.dumps(res).encode(), "application/json")

        self.srv = ThreadingHTTPServer((self.a.host, self.a.port), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


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
    p.add_argument("--block", type=int, default=240, help="audio block in samples (240 = 5 ms)")
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
