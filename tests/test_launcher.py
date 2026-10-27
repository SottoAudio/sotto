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
