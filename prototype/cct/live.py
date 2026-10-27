"""Live prototype: audio interface in/out, engine, recorder and a control page in the browser.

Shadow mode by design: it takes copies of the mic(s) and the PA reference and produces cleaned
copies for monitoring and recording. It never sits in the live comms path.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .engine import Engine, HOP, FS, MODES
from .recorder import Recorder

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_chans(s):
    """'1,2' or '3-4' (1-based) -> [0, 1] / [2, 3]."""
    if s is None or s == "":
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


class LiveApp:
    def __init__(self, a):
        self.a = a
        self.mic_idx = parse_chans(a.mic)
        self.ref_idx = parse_chans(a.ref)
        self.out_idx = parse_chans(a.out)
        self.dry_idx = parse_chans(a.out_dry)
        self.mon_idx = parse_chans(a.monitor)
        names = [n.strip() for n in a.names.split(",")] if a.names else []
        names += [f"talker{i + 1}" for i in range(len(names), len(self.mic_idx))]
        self.names = names[: len(self.mic_idx)]
        self.ref_names = [n.strip() for n in a.ref_names.split(",")] if a.ref_names else \
            [f"pa{i + 1}" for i in range(len(self.ref_idx))]
        if self.out_idx and len(self.out_idx) != len(self.mic_idx):
            raise SystemExit("--out needs one output channel per --mic channel")
        if self.dry_idx and len(self.dry_idx) != len(self.mic_idx):
            raise SystemExit("--out-dry needs one output channel per --mic channel")
        if self.mon_idx and len(self.mon_idx) not in (1, 2):
            raise SystemExit("--monitor takes one or two output channels (e.g. 5,6)")
        model = None if a.no_dfn else os.path.join(os.path.dirname(HERE), "models", "DeepFilterNet3_ll_stream")
        if model and not os.path.isfile(os.path.join(model, "enc.onnx")):
            print("Speech model not found, running echo cancelling only:", model)
            model = None
        self.engine = Engine(self.names, len(self.ref_idx), model_dir=model, tail_s=a.tail,
                             delay_ms=a.delay_ms, track_delay="process")
        n_m, n_r = len(self.mic_idx), len(self.ref_idx)
        ch_names = [f"{n} mic" for n in self.names] + [f"{n} (reference)" for n in self.ref_names] + \
                   [f"{n} processed" for n in self.names]
        self.rec = Recorder(a.record_dir, ch_names,
                            talkers={n: i for i, n in enumerate(self.names)},
                            refs={n: n_m + i for i, n in enumerate(self.ref_names)})
        self.mon_talker = 0
        self.mon_source = "out"          # out | dry | split
        self.xruns = 0
        self.cb_errors = 0
        self.last_error = None
        self.started = time.time()
        self.mode_label = "file playback" if a.play else "live"
        self.play = None
        if a.play:
            self._load_play(a.play)

    # ---------------------------------------------------------------- file playback
    def _load_play(self, path):
        from .audio_io import load_trackmap, load_tracks
        tm = load_trackmap(path)
        fs, talkers, refs = load_tracks(tm, os.path.dirname(os.path.abspath(path)))
        tk = list(talkers)[: len(self.mic_idx) or len(talkers)]
        if not self.mic_idx:
            raise SystemExit("--play still needs --mic to say how many talkers (e.g. --mic 1)")
        mics = np.stack([talkers[k] for k in tk])
        rf = np.stack([refs[k] for k in list(refs)[: len(self.ref_idx)]])
        if rf.shape[0] != len(self.ref_idx):
            raise SystemExit(f"track map has {rf.shape[0]} references, --ref lists {len(self.ref_idx)}")
        self.play = {"mics": mics, "refs": rf, "pos": 0, "n": mics.shape[1]}
        print(f"Playing {path}: talkers {tk}, {self.play['n'] / FS:.0f} s, looping")

    def _next_play_block(self):
        p = self.play
        idx = (np.arange(p["pos"], p["pos"] + HOP)) % p["n"]
        p["pos"] = (p["pos"] + HOP) % p["n"]
        return p["mics"][:, idx], p["refs"][:, idx]

    # ---------------------------------------------------------------- audio callback
    def _route(self, outdata, mics, outs, allv):
        outdata.fill(0)
        for i, ch in enumerate(self.out_idx):
            outdata[:, ch] = outs[i]
        for i, ch in enumerate(self.dry_idx):
            outdata[:, ch] = allv[i]["dry"]
        if self.mon_idx:
            t = min(self.mon_talker, len(outs) - 1)
            o, d = outs[t], allv[t]["dry"]
            if self.mon_source == "out":
                L = R = o
            elif self.mon_source == "dry":
                L = R = d
            else:
                L, R = d, o
            outdata[:, self.mon_idx[0]] += L
            if len(self.mon_idx) > 1:
                outdata[:, self.mon_idx[1]] += R

    def callback(self, indata, outdata, frames, tinfo, status):
        if status:
            self.xruns += 1
        try:
            if self.play is not None:
                mics, refs = self._next_play_block()
            else:
                x = indata.astype(np.float64)
                mics = x[:, self.mic_idx].T
                refs = x[:, self.ref_idx].T
            outs, allv = self.engine.process(mics, refs)
            self._route(outdata, mics, outs, allv)
            self.rec.push(np.concatenate([mics.T, refs.T, outs.T], axis=1))
        except Exception as e:                                   # fail to dry, never to silence
            self.cb_errors += 1
            self.last_error = repr(e)
            try:
                outdata.fill(0)
                for i, ch in enumerate(self.out_idx):
                    outdata[:, ch] = mics[i]
                if self.mon_idx:
                    for ch in self.mon_idx:
                        outdata[:, ch] = mics[min(self.mon_talker, len(mics) - 1)]
            except Exception:
                outdata.fill(0)

    def play_callback(self, outdata, frames, tinfo, status):
        self.callback(None, outdata, frames, tinfo, status)

    # ---------------------------------------------------------------- control
    def state(self):
        s = self.engine.state()
        s.update({
            "running_s": round(time.time() - self.started), "xruns": self.xruns,
            "callback_errors": self.cb_errors, "last_error": self.last_error,
            "monitor": {"talker": self.mon_talker, "source": self.mon_source, "available": bool(self.mon_idx)},
            "rec": self.rec.status(), "source": self.mode_label, "block_ms": 1000 * HOP / FS,
            "latency_note": self.latency_note,
        })
        return s

    def command(self, c):
        cmd = c.get("cmd")
        t = int(c.get("talker", 0))
        talker = self.engine.talkers[t] if 0 <= t < len(self.engine.talkers) else None
        if cmd == "mode" and talker and c.get("mode") in MODES:
            if c["mode"] == "clean" and talker.dfn is None:
                return {"ok": False, "error": "speech model not loaded"}
            talker.mode = c["mode"]
        elif cmd == "monitor":
            if c.get("source") in ("out", "dry", "split"):
                self.mon_source = c["source"]
            if "talker" in c:
                self.mon_talker = t
        elif cmd == "delay" and talker:
            ms = c.get("ms")
            if ms in (None, "", "auto"):
                talker.clear_manual_delay()
                if self.engine.tracker:
                    self.engine.tracker.hist.pop(t, None)
            else:
                talker.set_delay(int(float(ms) * FS / 1000), source="manual")
        elif cmd == "remeasure" and talker:
            talker.clear_manual_delay()
            if self.engine.tracker:
                self.engine.tracker.hist.pop(t, None)
        elif cmd == "reset" and talker:
            talker._reset_req = True
        elif cmd == "rec":
            self.rec.start() if c.get("on") else self.rec.stop()
        elif cmd == "mark":
            tt = self.rec.mark(str(c.get("note", ""))[:200])
            return {"ok": tt is not None, "t": tt}
        else:
            return {"ok": False, "error": "unknown command"}
        return {"ok": True}

    # ---------------------------------------------------------------- run
    def run(self):
        import sounddevice as sd
        a = self.a
        n_in = max(self.mic_idx + self.ref_idx) + 1 if not self.play else 0
        outs_all = self.out_idx + self.dry_idx + self.mon_idx
        n_out = max(outs_all) + 1 if outs_all else 2
        if not outs_all:
            self.mon_idx = [0, 1]
        dev = a.device
        dev = int(dev) if dev is not None and str(dev).isdigit() else dev
        lat = a.latency
        try:
            lat = float(lat)
        except ValueError:
            pass
        self._start_web()
        status_th = threading.Thread(target=self._status_loop, daemon=True)
        status_th.start()
        if a.record:
            self.rec.start()
        if self.play:
            stream = sd.OutputStream(device=dev, samplerate=FS, blocksize=HOP, channels=n_out,
                                     dtype="float32", latency=lat, callback=self.play_callback)
        else:
            stream = sd.Stream(device=dev, samplerate=FS, blocksize=HOP, channels=(n_in, n_out),
                               dtype="float32", latency=lat, callback=self.callback)
        with stream:
            il, ol = (stream.latency if isinstance(stream.latency, tuple) else (0.0, stream.latency))
            self.latency_note = (f"interface in {1000 * il:.1f} ms + out {1000 * ol:.1f} ms + "
                                 f"processing 20 ms = about {1000 * (il + ol) + 20:.0f} ms total")
            print(f"Running. {self.latency_note}")
            print(f"Control page: http://127.0.0.1:{a.port}   (Ctrl+C to stop)")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nStopping...")
        self.rec.stop()
        self.engine.stop()

    latency_note = "starting"

    def _status_loop(self):
        while True:
            time.sleep(1.0)
            try:
                s = self.engine.state()
                s["xruns"] = self.xruns
                self.rec.log(s)
            except Exception:
                pass

    def _start_web(self):
        app = self
        page = open(os.path.join(HERE, "ui.html"), "rb").read()

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
                elif self.path == "/state":
                    self._send(200, json.dumps(app.state()).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path != "/cmd":
                    return self._send(404, b"not found", "text/plain")
                n = int(self.headers.get("Content-Length", 0))
                try:
                    res = app.command(json.loads(self.rfile.read(n) or b"{}"))
                except Exception as e:
                    res = {"ok": False, "error": repr(e)}
                self._send(200, json.dumps(res).encode(), "application/json")

        srv = ThreadingHTTPServer((self.a.host, self.a.port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
