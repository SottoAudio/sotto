"""The log, crash records and the diagnostics file."""
import json
import logging
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "failsafe"))
sys.path.insert(0, os.path.join(ROOT, "prototype"))

from sotto_failsafe import app as appmod, diagnostics, launcher  # noqa: E402


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.cfg = mock.patch.object(appmod, "config_dir", return_value=self.tmp.name)
        self.cfg.start()
        diagnostics.stop_logging()
        diagnostics.setup_logging(echo=False)

    def tearDown(self):
        diagnostics.stop_logging()
        self.cfg.stop()
        self.tmp.cleanup()

    def read_log(self):
        diagnostics.stop_logging()                     # flushes the queue to the file
        with open(diagnostics.log_path(), encoding="utf-8") as f:
            return f.read()

    def test_log_lines_say_where_they_came_from(self):
        diagnostics.LOG.info("hello from the test")
        try:
            1 / 0
        except ZeroDivisionError:
            diagnostics.LOG.exception("it broke")
        text = self.read_log()
        self.assertIn("test_diagnostics.test_log_lines_say_where_they_came_from:", text)
        self.assertIn("hello from the test", text)
        self.assertIn("ZeroDivisionError", text)
        self.assertIn('test_diagnostics.py", line', text)       # the traceback gives file and line

    def test_crash_is_kept_for_the_setup_page(self):
        try:
            raise OSError(48, "Address already in use")
        except OSError as e:
            diagnostics.record_crash(e, "start-up")
        c = diagnostics.last_crash()
        self.assertIn("port 8766", c["error"])
        self.assertIn("Traceback", c["traceback"])
        diagnostics.last_crash(clear=True)
        self.assertIsNone(diagnostics.last_crash())

    def test_failed_start_goes_back_to_the_setup_page_with_the_reason(self):
        calls = []

        def fake_setup(open_browser=True, start_error=None):
            calls.append(start_error)
            return None if len(calls) > 1 else ["--device", "0", "--out-dir", self.tmp.name, "--port", "8796"]

        def boom(self):
            raise RuntimeError("Error opening Stream: Invalid number of channels [PaErrorCode -9998]")

        with mock.patch.object(launcher, "run_setup", side_effect=fake_setup), \
                mock.patch.object(appmod.BenchApp, "run", boom):
            self.assertEqual(launcher._main([]), 0)
        self.assertEqual(calls[0], None)
        self.assertIn("64 x 64", calls[1])                           # plain reason on the page
        self.assertIn("Invalid number of channels", diagnostics.last_crash()["detail"])
        self.assertIn("Sotto Bench stopped during starting the engine", self.read_log())

    def test_report_has_every_section(self):
        text = diagnostics.build_report(None, setup_config={"device": "0"}, run_speed_test=False)
        for title in ("What looks wrong", "Build", "Computer", "Audio devices (PortAudio", "Saved setup choices",
                      "Setup page now", "Busy threads and programs", "Threads right now", "Log (last"):
            self.assertIn(f"==== {title}", text)
        self.assertIn("MainThread", text)

    def test_findings_name_the_usual_problems(self):
        state = {"processing": {"on": True, "proc_ms": 18.0, "cpu_ms": 14.5, "talkers": [{}] * 5},
                 "levels_in": [-120] * 62, "health": "PROBE 1 missing", "xruns": 0}
        sysinfo = {"power": {"low_power": True, "thermal": "normal", "power": "mains"}, "rosetta": "no"}
        devs = {"ok": True, "devices": [{"inputs": 2, "outputs": 2}]}
        f = "\n".join(diagnostics._findings(state, sysinfo, devs, None, None))
        for words in ("Low Power Mode", "64 inputs and 64 outputs", "can't keep up", "Every input is silent", "not green"):
            self.assertIn(words, f)

    @unittest.skipUnless(os.path.isdir(os.path.join(ROOT, "prototype", "models", "DeepFilterNet3_ll_stream_batch")),
                         "speech model not downloaded")
    def test_speed_test_measures_each_stage(self):
        r = diagnostics.speed_test(n_talkers=2, blocks=10)
        row = r["user_interactive"] if sys.platform == "darwin" else r["default"]
        for k in ("front", "model", "back", "cpu_total", "python_loop_ms"):
            self.assertGreater(row[k], 0)


if __name__ == "__main__":
    unittest.main()
