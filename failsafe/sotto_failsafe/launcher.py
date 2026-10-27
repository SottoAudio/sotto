"""Double-click entry point for the packaged Sotto Bench app.

Opens a setup page in the browser (pick the virtual soundcard and channels, or the simulated
Riedel frame), remembers the choices, then starts the bench tool and hands the same browser tab
over to its console. With command-line options it skips the setup page and starts directly.
"""
import json
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .app import BenchApp, channel_clashes, config_from_args, parse_args

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8766

DEFAULTS = {
    "device": "", "talker_in": "1-4", "talker_out": "1-4", "names": "Show caller,Creative director,Technical manager,Stage manager",
    "probe_in": "61", "probe_out": "62", "heartbeat_out": "61", "pgm_in": "62", "listen_in": "", "monitor": "63,64",
    "rrcs_host": "", "record": True, "layout": 2,
}
# Channels with a fixed job since layout 2 (IN 61 PROBE return, IN 62 PGM, OUT 61 HEARTBEAT,
# OUT 62 PROBE, OUT 63/64 headphones). Settings saved by older versions are moved onto them.
FIXED = ("probe_in", "probe_out", "heartbeat_out", "pgm_in", "listen_in", "monitor")


from .app import config_dir  # noqa: E402,F811  (one settings folder for the launcher and the app)


def runs_dir():
    docs = os.path.join(os.path.expanduser("~"), "Documents")
    return os.path.join(docs if os.path.isdir(docs) else os.path.expanduser("~"), "Sotto Bench Runs")


def load_config():
    try:
        with open(os.path.join(config_dir(), "config.json")) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return dict(DEFAULTS)
    if saved.get("layout") != DEFAULTS["layout"]:
        saved.update({k: DEFAULTS[k] for k in FIXED})
        saved["layout"] = DEFAULTS["layout"]
    return {**DEFAULTS, **saved}


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
                argv = argv_from(cfg, simulate)
                try:
                    clash = channel_clashes(config_from_args(parse_args(argv)))
                except (SystemExit, ValueError) as e:
                    clash = f"Check the channel numbers ({e})"
                if clash:
                    return self._send(200, json.dumps({"ok": False, "error": clash}).encode())
                save_config(cfg)
                result["argv"] = argv
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


def port_free(port):
    """True if nothing is listening on 127.0.0.1:port. Tries to connect first: on Windows a bind
    with address reuse succeeds even while another program is listening."""
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.settimeout(0.5)
    try:
        if c.connect_ex(("127.0.0.1", port)) == 0:
            return False
    finally:
        c.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sys.platform != "win32":
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)    # same as the HTTP server
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def sotto_on(port):
    """True if the program on this port is Sotto Bench (its setup page or its console)."""
    try:
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=2).read()
    except urllib.error.HTTPError as e:                           # the setup page answers 503
        body = e.read()
    except Exception:
        return False
    try:
        j = json.loads(body)
    except ValueError:
        return False
    return isinstance(j, dict) and ("setup" in j or "armed" in j)


def pick_port():
    """Returns (port, already_running). Sotto Bench normally uses 8766. If Sotto Bench is already
    running there, the caller just opens it. If something else has the port, use the next free one."""
    if port_free(PORT):
        return PORT, False
    if sotto_on(PORT):
        return PORT, True
    for p in range(PORT + 1, PORT + 34):
        if port_free(p):
            return p, False
    return PORT, False                                            # let the error say what's wrong


def main(argv=None):
    global PORT
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = [x for x in argv if not x.startswith("-psn_")]          # old macOS Finder launch argument
    open_browser = "--no-browser" not in argv
    argv = [x for x in argv if x != "--no-browser"]
    if "--port" not in argv and not any(x in argv for x in ("--list", "-h", "--help")):
        port, running = pick_port()
        if running:
            # Opened a second time (Sotto Bench has no window of its own): show the one that's running.
            print(f"Sotto Bench is already running: http://127.0.0.1:{port}/")
            if open_browser:
                webbrowser.open(f"http://127.0.0.1:{port}/")
            return 0
        if port != PORT:
            print(f"Port {PORT} is in use by another program; using {port}")
            PORT = port
            if argv:
                argv += ["--port", str(port)]
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
