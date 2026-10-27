"""Double-click entry point for the packaged Sotto Voce app.

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

from . import diagnostics
from .app import BenchApp, channel_clashes, config_from_args, device_list, parse_args
LOG = diagnostics.LOG

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8766

DEFAULTS = {
    "device": "", "talker_in": "1-4", "talker_out": "1-4", "names": "Show caller,Creative director,Technical manager,Stage manager",
    "probe_in": "61", "probe_out": "62", "heartbeat_out": "61", "pgm_in": "62", "listen_in": "", "monitor": "63,64",
    "rrcs_host": "", "rrcs_port": "8193", "record": True, "layout": 2, "io_buffer": "0",
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
    return device_list()


def argv_from(cfg, simulate):
    """Turns the setup page's choices into bench tool options."""
    argv = ["--out-dir", runs_dir(), "--port", str(PORT)]
    if simulate:
        argv.append("--simulate")
    else:
        argv += ["--device", str(cfg["device"])]
        if cfg.get("rrcs_host"):
            argv += ["--rrcs-host", cfg["rrcs_host"], "--rrcs-port", str(cfg.get("rrcs_port") or 8193)]
        if str(cfg.get("io_buffer") or "0") not in ("", "0"):
            argv += ["--io-buffer", str(cfg["io_buffer"])]
    # The I/O patch (the simulated frame follows it too)
    argv += ["--talker-in", cfg["talker_in"], "--talker-out", cfg["talker_out"],
             "--probe-in", cfg["probe_in"], "--probe-out", cfg["probe_out"], "--heartbeat-out", cfg["heartbeat_out"],
             "--pgm-in", cfg.get("pgm_in", "")]
    if cfg.get("listen_in"):
        argv += ["--listen-in", cfg["listen_in"]]
    if cfg.get("monitor"):
        argv += ["--monitor", cfg["monitor"]]
    if cfg.get("names"):
        argv += ["--names", cfg["names"]]
    if cfg.get("record", True):
        argv.append("--record")
    return argv


def run_setup(open_browser=True, start_error=None):
    """Serves the setup page until the user starts or quits. Returns bench argv, or None to quit.
    start_error: why the last start failed, shown on the page."""
    LOG.info("Setup page on http://127.0.0.1:%s/%s", PORT, f" (last start failed: {start_error})" if start_error else "")
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
            try:
                self._get()
            except Exception as e:
                LOG.exception("Setup page request %s failed", self.path)
                self._send(500, json.dumps({"ok": False, "error": repr(e)}).encode())

        def do_POST(self):
            try:
                self._post()
            except Exception as e:
                LOG.exception("Setup page request %s failed", self.path)
                self._send(500, json.dumps({"ok": False, "error": repr(e)}).encode())

        def _get(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/setup/devices":
                d = list_devices()
                LOG.info("Devices listed for the setup page: %s", d if not d.get("ok") else
                         "; ".join(f"{x['index']}: {x['name']} ({x['inputs']}/{x['outputs']})" for x in d["devices"]))
                self._send(200, json.dumps(d).encode())
            elif self.path == "/setup/config":
                self._send(200, json.dumps({"config": load_config(), "defaults": DEFAULTS, "runs_dir": runs_dir(), "start_error": start_error,
                                            "last_crash": diagnostics.last_crash(),
                                            "log": diagnostics.log_path()}).encode())
            elif self.path.startswith("/setup/diagnostics"):
                text = diagnostics.build_report(None, setup_config=load_config(), run_speed_test="quick" not in self.path)
                diagnostics.save_report(text)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{diagnostics.report_filename()}"')
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(text.encode("utf-8"))
            elif self.path == "/api/state":
                self._send(503, b'{"setup": true}')           # console not running yet
            else:
                self._send(404, b"not found", "text/plain")

        def _post(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            if self.path == "/setup/start":
                # Settings the page doesn't show (the soundcard buffer, set from the console) keep
                # their saved value.
                cfg = {**load_config(), **{k: body[k] for k in DEFAULTS if k in body}}
                simulate = bool(body.get("simulate"))
                if not simulate and str(cfg["device"]) == "":
                    return self._send(200, b'{"ok": false, "error": "Pick a soundcard first"}')
                argv = argv_from(cfg, simulate)
                try:
                    clash = channel_clashes(config_from_args(parse_args(argv)))
                except (SystemExit, ValueError) as e:
                    clash = f"Check the channel numbers ({e})"
                if clash:
                    LOG.warning("Setup refused: %s", clash)
                    return self._send(200, json.dumps({"ok": False, "error": clash}).encode())
                save_config(cfg)
                LOG.info("Setup: start %s with %s", "the simulation" if simulate else f"device {cfg['device']}", argv)
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
    print(f"Sotto Voce setup: http://127.0.0.1:{PORT}/")
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
    """True if the program on this port is Sotto Voce (its setup page or its console)."""
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
    """Returns (port, already_running). Sotto Voce normally uses 8766. If Sotto Voce is already
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
    diagnostics.setup_logging()
    try:
        return _main(argv)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as e:
        diagnostics.record_crash(e, "start-up")
        raise
    finally:
        diagnostics.stop_logging()


def write_diagnostics(argv):
    """Sotto Voce --diagnostics [--quick]: writes the diagnostics file to Downloads and exits."""
    text = diagnostics.build_report(None, setup_config=load_config(), run_speed_test="--quick" not in argv)
    # Builds and tests set SOTTO_CONFIG_DIR: keep their files out of the computer's Downloads.
    paths = diagnostics.save_report(text, None if os.environ.get("SOTTO_CONFIG_DIR") else diagnostics.downloads_dir())
    print("Diagnostics written to:\n  " + "\n  ".join(paths))
    return 0


def _main(argv=None):
    global PORT
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = [x for x in argv if not x.startswith("-psn_")]          # old macOS Finder launch argument
    LOG.info("Launched with %s (executable %s)", argv, sys.executable)
    if "--diagnostics" in argv:
        return write_diagnostics(argv)
    open_browser = "--no-browser" not in argv
    argv = [x for x in argv if x != "--no-browser"]
    if "--port" not in argv and not any(x in argv for x in ("--list", "-h", "--help")):
        port, running = pick_port()
        if running:
            # Opened a second time (Sotto Voce has no window of its own): show the one that's running.
            LOG.info("Already running on port %s: opening it", port)
            print(f"Sotto Voce is already running: http://127.0.0.1:{port}/")
            if open_browser:
                webbrowser.open(f"http://127.0.0.1:{port}/")
            return 0
        if port != PORT:
            LOG.warning("Port %s is in use by another program; using %s", PORT, port)
            print(f"Port {PORT} is in use by another program; using {port}")
            PORT = port
            if argv:
                argv += ["--port", str(port)]
    if argv:                                           # started with options (Terminal, a restart)
        if "--out-dir" not in argv:
            argv += ["--out-dir", runs_dir()]
        if open_browser:
            port = argv[argv.index("--port") + 1] if "--port" in argv else str(PORT)
            threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
        a = parse_args(argv)
        if a.list:
            from .app import list_devices as print_devices
            return print_devices()
        if run_engine(argv, a) != "setup":
            return 0
        PORT = a.port                                  # the setup page where the console was
        open_browser = False
    # Started like an app: the setup page, then the engine. If the engine can't start, back to the
    # setup page with the reason, instead of quitting silently.
    error = None
    while True:
        argv = run_setup(open_browser, start_error=error)
        if argv is None:
            LOG.info("Quit from the setup page")
            return 0
        open_browser = False                           # the page is already open
        try:
            if run_engine(argv, parse_args(argv)) == "setup":
                error = None
                continue
            return 0
        except SystemExit as e:
            if not str(e) or str(e) == "0":
                raise
            error = str(e)
            LOG.warning("Engine didn't start: %s", error)
        except Exception as e:
            diagnostics.record_crash(e, "starting the engine")
            error = diagnostics.friendly_error(e)


def run_engine(argv, a):
    """Runs the engine. Returns "setup" when the console asked to go back to the setup page."""
    app = BenchApp(a)
    app.run()
    if app.restart_argv:                               # the console changed the engine or the patch
        LOG.info("Restarting the engine with %s", app.restart_argv)
        diagnostics.stop_logging()
        restart(app.restart_argv, {})
    if app.restart_with:                               # the console switched soundcard or buffer
        LOG.info("Restarting the engine with %s", app.restart_with)
        diagnostics.stop_logging()
        restart(argv, app.restart_with)
    if app.to_setup:
        LOG.info("Back to the setup page")
        return "setup"
    return 0


def restart(argv, changes):
    """Starts again with changed options (e.g. {"--device": "2", "--io-buffer": "160"}), in the same
    process slot (the browser stays open). A plain string means a new --device."""
    if isinstance(changes, str):
        changes = {"--device": changes}
    argv = list(argv)
    for opt, val in changes.items():
        if opt in argv:
            argv[argv.index(opt) + 1] = str(val)
        else:
            argv += [opt, str(val)]
    argv.append("--no-browser")
    cmd = [sys.executable] + ([] if getattr(sys, "frozen", False) else [os.path.abspath(sys.argv[0])]) + argv
    sys.stdout.flush()
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    sys.exit(main())
