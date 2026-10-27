"""Packaged-app launcher: setup choices become the right options, and the app quits cleanly."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "failsafe"))

from sotto_failsafe import launcher  # noqa: E402
from sotto_failsafe.app import BenchApp, parse_args  # noqa: E402


class LauncherTests(unittest.TestCase):
    def test_simulation_options(self):
        argv = launcher.argv_from(dict(launcher.DEFAULTS), simulate=True)
        a = parse_args(argv)
        self.assertTrue(a.simulate)
        self.assertEqual(a.listen_in, "63")
        self.assertEqual((a.probe_in, a.pgm_in, a.heartbeat_out, a.probe_out, a.monitor), ("61", "62", "61", "62", "63,64"))
        self.assertTrue(a.record)

    def test_device_options(self):
        cfg = dict(launcher.DEFAULTS, device="3", monitor="7,8", rrcs_host="10.0.0.10", record=False)
        a = parse_args(launcher.argv_from(cfg, simulate=False))
        self.assertFalse(a.simulate)
        self.assertEqual((a.device, a.talker_in, a.heartbeat_out, a.monitor, a.rrcs_host), ("3", "1-4", "61", "7,8", "10.0.0.10"))
        self.assertFalse(a.record)

    def test_runs_and_quits_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            a = parse_args(["--simulate", "--port", "8797", "--out-dir", d])
            app = BenchApp(a)
            th = threading.Thread(target=app.run, daemon=True)
            th.start()
            state = None
            for _ in range(50):
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8797/api/state", timeout=1) as r:
                        state = json.load(r)
                    break
                except OSError:
                    time.sleep(0.1)
            self.assertIsNotNone(state)
            self.assertFalse(state["armed"])                      # always starts disarmed
            self.assertEqual(state["solo"]["ids"], [])             # and with nothing soloed
            for _ in range(30):                                   # first blocks arrive
                with urllib.request.urlopen("http://127.0.0.1:8797/api/state", timeout=1) as r:
                    state = json.load(r)
                if state["levels_in"]:
                    break
                time.sleep(0.1)
            self.assertEqual(len(state["levels_in"]), 63)          # talkers, PROBE 61, PGM 62, listener 63
            self.assertGreater(state["levels_in"][61], -60)        # the simulated PGM is metered
            self.assertEqual(state["channels"]["monitor_out"], [63, 64])
            self.assertEqual(state["frame"]["source"], "simulated")
            self.assertTrue(state["engine"]["simulate"])
            with urllib.request.urlopen("http://127.0.0.1:8797/", timeout=2) as r:
                self.assertIn(b"<title>Sotto Console</title>", r.read())
            with urllib.request.urlopen("http://127.0.0.1:8797/bench", timeout=2) as r:
                self.assertEqual(r.status, 200)

            def cmd(body):
                req = urllib.request.Request("http://127.0.0.1:8797/cmd", data=json.dumps(body).encode(), method="POST")
                with urllib.request.urlopen(req, timeout=2) as r:
                    return json.load(r)
            res = cmd({"cmd": "solo", "ids": ["o1"], "mode": "split"})
            self.assertEqual((res["left"], res["right"]), (["i1"], ["o1"]))
            self.assertTrue(cmd({"cmd": "rrcs_set", "name": "ARMED", "on": False})["ok"])
            req = urllib.request.Request("http://127.0.0.1:8797/quit", data=b"{}", method="POST")
            with urllib.request.urlopen(req, timeout=2) as r:
                self.assertTrue(json.load(r)["ok"])
            th.join(timeout=5)
            self.assertFalse(th.is_alive())
            with open(os.path.join(app.dir, f"{app.take}_events.csv")) as f:
                self.assertIn("stopped", f.read())


if __name__ == "__main__":
    unittest.main()


class SecondLaunchTests(unittest.TestCase):
    """Opening Sotto Bench while it's already running must not fail silently."""

    def setUp(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.saved = launcher.PORT
        launcher.PORT = self.port

    def tearDown(self):
        launcher.PORT = self.saved

    def _serve(self, code, body):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(code)
                self.end_headers()
                self.wfile.write(body)

        srv = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)

    def test_free_port_is_used(self):
        self.assertEqual(launcher.pick_port(), (self.port, False))

    def test_sotto_already_running_is_reused(self):
        self._serve(503, b'{"setup": true}')
        self.assertEqual(launcher.pick_port(), (self.port, True))

    def test_other_program_on_the_port_moves_sotto(self):
        self._serve(200, b"<html>someone else</html>")
        port, running = launcher.pick_port()
        self.assertFalse(running)
        self.assertNotEqual(port, self.port)
        self.assertTrue(launcher.port_free(port))


class ChannelLayoutTests(unittest.TestCase):
    """Settings from older versions move to the fixed channels; a channel can't have two jobs."""

    def test_old_saved_settings_move_to_the_fixed_channels(self):
        from unittest import mock
        old = {"device": "2", "talker_in": "1-5", "talker_out": "1-5", "probe_in": "5", "probe_out": "5",
               "heartbeat_out": "6", "listen_in": "6", "monitor": "7,8", "names": "a,b,c,d,e"}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(launcher, "config_dir", return_value=d):
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(old, f)
            cfg = launcher.load_config()
        self.assertEqual((cfg["probe_in"], cfg["probe_out"], cfg["heartbeat_out"], cfg["pgm_in"], cfg["monitor"]),
                         ("61", "62", "61", "62", "63,64"))
        self.assertEqual(cfg["listen_in"], "")
        self.assertEqual((cfg["device"], cfg["talker_in"], cfg["names"]), ("2", "1-5", "a,b,c,d,e"))   # kept

    def test_new_saved_settings_are_kept(self):
        from unittest import mock
        new = dict(launcher.DEFAULTS, monitor="7,8")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(launcher, "config_dir", return_value=d):
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(new, f)
            self.assertEqual(launcher.load_config()["monitor"], "7,8")

    def test_clashes(self):
        from sotto_failsafe.app import channel_clashes, config_from_args
        ok = config_from_args(parse_args(launcher.argv_from(dict(launcher.DEFAULTS, device="0"), simulate=False)))
        self.assertIsNone(channel_clashes(ok))
        bad = dict(launcher.DEFAULTS, device="0", talker_in="1-5", talker_out="1-5", probe_in="5")
        msg = channel_clashes(config_from_args(parse_args(launcher.argv_from(bad, simulate=False))))
        self.assertEqual(msg, "Input 5 is set as both talker 5 input and PROBE return. Give each job its own channel.")
        bad = dict(launcher.DEFAULTS, device="0", monitor="61,62")
        self.assertIn("Output 61", channel_clashes(config_from_args(parse_args(launcher.argv_from(bad, simulate=False)))))
        mono = dict(launcher.DEFAULTS, device="0", monitor="63,63")                 # mono phones: fine
        self.assertIsNone(channel_clashes(config_from_args(parse_args(launcher.argv_from(mono, simulate=False)))))
        sim = config_from_args(parse_args(launcher.argv_from(dict(launcher.DEFAULTS), simulate=True)))
        self.assertIsNone(channel_clashes(sim))


class MacOSTests(unittest.TestCase):
    def test_calls_never_raise(self):
        from sotto_failsafe import macos
        for f in (macos.keep_responsive, macos.fast_thread):
            ok, msg = f()
            self.assertIsInstance(msg, str)
            if sys.platform == "darwin":
                self.assertTrue(ok, msg)


class SwitchDeviceTests(unittest.TestCase):
    def test_restart_replaces_the_device_and_keeps_the_rest(self):
        from unittest import mock
        argv = ["--out-dir", "/x", "--port", "8766", "--device", "0", "--talker-in", "1-4"]
        with mock.patch.object(launcher.os, "execv") as ex:
            launcher.restart(argv, "3")
        exe, cmd = ex.call_args[0]
        self.assertEqual(exe, sys.executable)
        tail = cmd[cmd.index("--out-dir"):]
        self.assertEqual(tail, ["--out-dir", "/x", "--port", "8766", "--device", "3", "--talker-in", "1-4", "--no-browser"])

    def test_simulated_app_refuses_to_switch(self):
        with tempfile.TemporaryDirectory() as d:
            app = BenchApp(parse_args(["--simulate", "--port", "8798", "--out-dir", d]))
            r = app.switch_device("0")
            app.events_f.close()                           # Windows can't delete open files
            self.assertFalse(r["ok"])
            self.assertIsNone(app.restart_device)
            self.assertFalse(app.quit_ev.is_set())

    def test_switch_checks_channels_saves_and_stops(self):
        from unittest import mock
        from sotto_failsafe import app as appmod
        devs = {"ok": True, "error": None, "devices": [
            {"index": 0, "name": "Dante Virtual Soundcard", "inputs": 64, "outputs": 64, "api": "Core Audio"},
            {"index": 1, "name": "MacBook Pro Microphone", "inputs": 1, "outputs": 0, "api": "Core Audio"},
            {"index": 2, "name": "Merging VAD", "inputs": 64, "outputs": 64, "api": "Core Audio"}]}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(appmod, "device_list", return_value=devs), \
                mock.patch.object(appmod, "config_dir", return_value=d):
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"device": "0", "layout": 2}, f)
            app = BenchApp(parse_args(["--device", "0", "--port", "8799", "--out-dir", d]))
            app.engine_info = {"n_in": 62, "n_out": 64}
            try:
                r = app.switch_device("1")
                self.assertFalse(r["ok"])
                self.assertIn("1 inputs and 0 outputs", r["error"])
                r = app.switch_device("2")
            finally:
                app.events_f.close()                       # Windows can't delete open files
            self.assertTrue(r["ok"])
            self.assertEqual(app.restart_device, "2")
            self.assertTrue(app.quit_ev.is_set())
            with open(os.path.join(d, "config.json")) as f:
                self.assertEqual(json.load(f)["device"], "2")


class SoundcardBufferTests(unittest.TestCase):
    def test_buffer_choice_restarts_the_engine_and_is_remembered(self):
        from unittest import mock
        from sotto_failsafe import app as appmod
        with tempfile.TemporaryDirectory() as d, mock.patch.object(appmod, "config_dir", return_value=d):
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"device": "0", "layout": 2}, f)
            app = BenchApp(parse_args(["--device", "0", "--port", "8801", "--out-dir", d]))
            app.engine_info = {"n_in": 62, "n_out": 64}
            app.coreaudio = {"buffer_range": [64, 4096]}
            try:
                self.assertFalse(app.switch_device(io_buffer=100)["ok"])            # not a choice
                self.assertFalse(app.switch_device(io_buffer=0)["ok"])              # already in use
                r = app.switch_device(io_buffer=160)
            finally:
                app.events_f.close()
            self.assertTrue(r["ok"], r)
            self.assertEqual(app.restart_with, {"--io-buffer": "160"})
            with open(os.path.join(d, "config.json")) as f:
                self.assertEqual(json.load(f)["io_buffer"], "160")
        argv = launcher.argv_from(dict(launcher.DEFAULTS, device="0", io_buffer="160"), simulate=False)
        self.assertEqual(parse_args(argv).io_buffer, 160)
        self.assertNotIn("--io-buffer", launcher.argv_from(dict(launcher.DEFAULTS, device="0"), simulate=False))

    def test_restart_with_several_changes(self):
        from unittest import mock
        with mock.patch.object(launcher.os, "execv") as ex:
            launcher.restart(["--device", "0", "--port", "8766"], {"--device": "2", "--io-buffer": "128"})
        cmd = ex.call_args[0][1]
        tail = cmd[cmd.index("--device"):]
        self.assertEqual(tail, ["--device", "2", "--port", "8766", "--io-buffer", "128", "--no-browser"])


class ConsoleConfigTests(unittest.TestCase):
    def _app(self, d, *extra):
        app = BenchApp(parse_args(["--simulate", "--no-processing", "--port", "8802", "--out-dir", d, *extra]))
        app.events_f.close()
        app.log = lambda e: None
        return app

    def test_patch_change_restarts_with_this_runs_options(self):
        from unittest import mock
        from sotto_failsafe import app as appmod
        with tempfile.TemporaryDirectory() as d, mock.patch.object(appmod, "config_dir", return_value=d), \
                mock.patch.object(launcher, "config_dir", return_value=d):
            app = self._app(d)
            r = app.reconfigure({"pgm_in": "2"})                          # clashes with talker 2
            self.assertFalse(r["ok"])
            self.assertIn("Input 2", r["error"])
            self.assertIsNone(app.restart_argv)
            self.assertFalse(app.reconfigure({"bogus": "1"})["ok"])
            r = app.reconfigure({"talker_in": "1-2", "talker_out": "1-2", "names": "a,b"})
            self.assertTrue(r["ok"], r)
            a2 = parse_args(app.restart_argv)
            self.assertEqual((a2.talker_in, a2.talker_out, a2.names), ("1-2", "1-2", "a,b"))
            self.assertEqual((a2.port, a2.out_dir, a2.no_processing, a2.simulate), (8802, d, True, True))
            self.assertTrue(app.quit_ev.is_set())
            self.assertEqual(launcher.load_config()["talker_in"], "1-2")    # remembered

    def test_back_to_setup(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            app = self._app(d)
            self.assertTrue(app.back_to_setup()["ok"])
            self.assertTrue(app.to_setup and app.quit_ev.is_set())
        calls = []

        def fake_setup(open_browser=True, start_error=None):
            calls.append(1)
            return None
        with mock.patch.object(launcher, "run_setup", side_effect=fake_setup), \
                mock.patch.object(launcher, "run_engine", return_value="setup"):
            self.assertEqual(launcher._main(["--simulate", "--port", "8803", "--no-browser"]), 0)
        self.assertEqual(calls, [1])                                        # the console went back to setup

    def test_phones_limiter_ceiling(self):
        import numpy as np
        from sotto_failsafe.core import FailsafeCore
        from sotto_failsafe.app import config_from_args
        core = FailsafeCore(config_from_args(parse_args(["--simulate"])))
        core.set_solo(["i1", "i2"], "clicked", level_db=12, limit_db=-10)
        x = np.full((480, 63), 0.9, np.float32)
        out = np.zeros((480, 64), np.float32)
        for _ in range(3):
            core.process(x, out)
        self.assertLessEqual(float(np.max(np.abs(out[:, 62]))), 10 ** (-10 / 20) + 1e-6)
