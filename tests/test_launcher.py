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
        self.assertEqual(a.listen_in, "6")
        self.assertTrue(a.record)

    def test_device_options(self):
        cfg = dict(launcher.DEFAULTS, device="3", monitor="7,8", rrcs_host="10.0.0.10", record=False)
        a = parse_args(launcher.argv_from(cfg, simulate=False))
        self.assertFalse(a.simulate)
        self.assertEqual((a.device, a.talker_in, a.heartbeat_out, a.monitor, a.rrcs_host), ("3", "1-4", "6", "7,8", "10.0.0.10"))
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
            req = urllib.request.Request("http://127.0.0.1:8797/quit", data=b"{}", method="POST")
            with urllib.request.urlopen(req, timeout=2) as r:
                self.assertTrue(json.load(r)["ok"])
            th.join(timeout=5)
            self.assertFalse(th.is_alive())
            with open(os.path.join(app.dir, f"{app.take}_events.csv")) as f:
                self.assertIn("stopped", f.read())


if __name__ == "__main__":
    unittest.main()
