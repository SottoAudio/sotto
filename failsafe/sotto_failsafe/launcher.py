"""Double-click entry point for the packaged Sotto Bench app.

Opens a setup page in the browser (pick the virtual soundcard and channels, or the simulated
Riedel frame), remembers the choices, then starts the bench tool and hands the same browser tab
over to its console. With command-line options it skips the setup page and starts directly.
"""
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .app import BenchApp, parse_args

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8766

DEFAULTS = {
    "device": "", "talker_in": "1-4", "talker_out": "1-4", "names": "Show caller,Creative director,Technical manager,Stage manager",
    "probe_in": "61", "probe_out": "62", "heartbeat_out": "61", "pgm_in": "62", "listen_in": "", "monitor": "63,64",
    "rrcs_host": "", "record": True,
}


def config_dir():
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support", "Sotto Bench")
    if os.name == "nt":
        return os.path.join(os.environ.get("APPDATA", home), "Sotto Bench")
    return os.path.join(home, ".config", "sotto-bench")


def runs_dir():
    docs = os.path.join(os.path.expanduser("~"), "Documents")
    return os.path.join(docs if os.path.isdir(docs) else os.path.expanduser("~"), "Sotto Bench Runs")


def load_config():
    try:
        with open(os.path.join(config_dir(), "config.json")) as f:
            return {**DEFAULTS, **json.load(f)}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_config(cfg):
    try:
        os.makedirs(config_dir(), exist_ok=True)
        with open(os.path.join(config_dir(), "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass


def list_devices():
    try:
        import sounddevice as sd
        apis = sd.query_hostapis()
        return {"ok": True, "devices": [
            {"index": i, "name": d["name"], "inputs": d["max_input_channels"], "outputs": d["max_output_channels"],
             "api": apis[d["hostapi"]]["name"]} for i, d in enumerate(sd.query_devices())]}
    except Exception as e:                       # no PortAudio, no permission, ...
        return {"ok": False, "error": repr(e), "devices": []}


def argv_from(cfg, simulate):
    """Turns the setup page's choices into bench tool options."""
    argv = ["--out-dir", runs_dir(), "--port", str(PORT)]
    if simulate:
        argv.append("--simulate")
    else:
        argv += ["--device", str(cfg["device"]), "--talker-in", cfg["talker_in"], "--talker-out", cfg["talker_out"],
                 "--probe-in", cfg["probe_in"], "--probe-out", cfg["probe_out"], "--heartbeat-out", cfg["heartbeat_out"],
                 "--pgm-in", cfg.get("pgm_in", "")]
        if cfg.get("listen_in"):
            argv += ["--listen-in", cfg["listen_in"]]
        if cfg.get("monitor"):
            argv += ["--monitor", cfg["monitor"]]
        if cfg.get("rrcs_host"):
            argv += ["--rrcs-host", cfg["rrcs_host"]]
    if cfg.get("names"):
        argv += ["--names", cfg["names"]]
    if cfg.get("record", True):
        argv.append("--record")
    return argv


def run_setup(open_browser=True):
    """Serves the setup page until the user starts or quits. Returns bench argv, or None to quit."""
    with open(os.path.join(HERE, "setup.html"), "rb") as f:
        page = f.read()
    done = threading.Event()
    result = {"argv": None}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/setup/devices":
                self._send(200, json.dumps(list_devices()).encode())
            elif self.path == "/setup/config":
                self._send(200, json.dumps({"config": load_config(), "runs_dir": runs_dir()}).encode())
            elif self.path == "/api/state":
                self._send(503, b'{"setup": true}')           # console not running yet
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            if self.path == "/setup/start":
                cfg = {**DEFAULTS, **{k: body.get(k, v) for k, v in DEFAULTS.items()}}
                simulate = bool(body.get("simulate"))
                if not simulate and str(cfg["device"]) == "":
                    return self._send(200, b'{"ok": false, "error": "Pick a soundcard first"}')
                save_config(cfg)
                result["argv"] = argv_from(cfg, simulate)
                self._send(200, b'{"ok": true}')
                done.set()
            elif self.path == "/quit":
                self._send(200, b'{"ok": true}')
                done.set()
            else:
                self._send(404, b"not found", "text/plain")

    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{PORT}/")
    print(f"Sotto Bench setup: http://127.0.0.1:{PORT}/")
    done.wait()
    srv.shutdown()
    srv.server_close()
    return result["argv"]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = [x for x in argv if not x.startswith("-psn_")]          # old macOS Finder launch argument
    open_browser = "--no-browser" not in argv
    argv = [x for x in argv if x != "--no-browser"]
    if not argv:
        argv = run_setup(open_browser)
        if argv is None:
            return 0
    else:
        if "--out-dir" not in argv:
            argv += ["--out-dir", runs_dir()]
        if open_browser:
            port = argv[argv.index("--port") + 1] if "--port" in argv else str(PORT)
            threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    a = parse_args(argv)
    if a.list:
        from .app import list_devices as print_devices
        return print_devices()
    BenchApp(a).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
