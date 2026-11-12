import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "failsafe"))
from sotto_failsafe import loadtest as L  # noqa: E402


def row(n, same, h1, h2, cost=2.8):
    mk = lambda pct, run=0: {"blocks": 1000, "late": int(pct * 10), "late_pct": pct, "max_run": run,   # noqa: E731
                             "fits": pct <= 0.5 and run < 3}
    return {"talkers": n, "workers": min(n, 9), "cpu_ms_per_block": cost * n, "cpu_ms_per_talker": cost,
            "cores_busy": cost * n / 10, "slowest_worker_ms": 3.0, "slowest_worker_max_ms": 7.0,
            "same": mk(same), "h1": mk(h1), "h2": mk(h2)}


class LateTests(unittest.TestCase):
    def test_rate_and_runs(self):
        t = L._Late()
        for ok in [True] * 995 + [False] * 2 + [True] + [False] * 2:
            t.add(ok)
        r = t.report()
        self.assertEqual((r["blocks"], r["late"], r["max_run"]), (1000, 4, 2))
        self.assertTrue(r["fits"])                       # 0.4 %, never 3 in a row
        for ok in (False, False, False):
            t.add(ok)
        self.assertFalse(t.report()["fits"])             # 3 in a row drops a talker to dry: never fits

    def test_nothing_counted_does_not_fit(self):
        self.assertFalse(L._Late().report()["fits"])


class SummaryTests(unittest.TestCase):
    def test_the_most_that_fit_at_each_delay(self):
        rows = [row(1, 0, 0, 0), row(2, 0, 0, 0), row(4, 0.1, 0, 0), row(6, 2.0, 0.2, 0), row(8, 9.0, 1.5, 0.3),
                row(10, 40.0, 12.0, 4.0, cost=3.1)]
        s = L.summarise(rows)
        self.assertEqual((s["same"], s["h1"], s["h2"]), (4, 6, 8))
        self.assertEqual(s["cpu_ms_per_talker"], 2.8)    # from the steps that kept up
        self.assertIn("4 talkers at 10 ms through (same block)", s["headline"])
        self.assertIn("8 talkers at 30 ms through", s["headline"])

    def test_a_late_step_stops_the_count_even_if_a_bigger_one_fits(self):
        s = L.summarise([row(1, 0, 0, 0), row(2, 3.0, 0, 0), row(4, 0.1, 0, 0)])
        self.assertEqual(s["same"], 1)

    def test_kept_up_with_everything_tried(self):
        s = L.summarise([row(1, 0, 0, 0), row(2, 0, 0, 0)])
        self.assertIn("still kept up", s["headline"])

    def test_report_text(self):
        res = {"state": "done", "rows": [row(1, 0, 0, 0), {"talkers": 2, "workers": 1, "error": "no workers"}],
               "summary": L.summarise([row(1, 0, 0, 0)]), "insert": "", "seconds_per_step": 10, "cores": 12,
               "started": time.time()}
        text = L.report(res)
        self.assertIn("per talker", text)
        self.assertIn("no workers", text)
        self.assertEqual(L.report(None), "(not run)")


class InsertTests(unittest.TestCase):
    def test_first_insert_in_the_project(self):
        p = {"operators": [{"id": "a"}], "ch": [{}, {"insert": {"path": "/Library/Audio/Plug-Ins/VST3/CEDAR DNS.vst3",
                                                               "class": "x", "on": False, "state": "abc"}}]}
        ins = L.first_insert(p)
        self.assertEqual(ins["path"], "/Library/Audio/Plug-Ins/VST3/CEDAR DNS.vst3")
        self.assertTrue(ins["on"])                       # always on for the test
        self.assertEqual(L.insert_name(ins), "CEDAR DNS")
        self.assertIsNone(L.first_insert({}))


class RunTests(unittest.TestCase):
    """A short real run: the workers start, get a block every 10 ms, and the row is filled in.
    Echo removal only, so it runs without the speech model and on slow test machines."""

    def test_one_step(self):
        r = L.run_step(2, None, mode="aec", seconds=1.0, track_delay=False, workers=1)
        self.assertNotIn("error", r, r)
        self.assertEqual((r["talkers"], r["workers"]), (2, 1))
        self.assertGreater(r["cpu_ms_per_block"], 0.0)
        for k, _ in L.FITS:
            self.assertEqual(r[k]["blocks"], 100)

    def test_runs_in_the_background_and_stops(self):
        """The steps in order, up to the most asked for, stopping at the first that doesn't fit at 30 ms;
        saved for the diagnostics; one at a time; Stop works mid-step."""
        def fake(n, model, cancel=None, seconds=0, **k):
            t0 = time.monotonic()
            while time.monotonic() - t0 < seconds:
                if cancel.is_set():
                    return {"talkers": n, "cancelled": True}
                time.sleep(0.01)
            return row(n, 0, 0, 0 if n < 4 else 5.0)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(L, "run_step", side_effect=fake) as rs:
            path = os.path.join(d, "loadtest.json")
            t = L.LoadTest(path)
            self.assertFalse(t.start(None)[0])           # Full clean needs the speech model
            ok, _ = t.start("model", steps=(1, 2, 4, 6, 64), max_talkers=48, seconds=0.05)
            self.assertTrue(ok)
            self.assertFalse(t.start("model")[0])        # one at a time
            t.thread.join(30)
            st = t.state()
            self.assertEqual(st["state"], "done", st)
            self.assertEqual(st["plan"], [1, 2, 4, 6])   # 64 is over the most asked for
            self.assertEqual([r["talkers"] for r in st["rows"]], [1, 2, 4])   # 4 didn't keep up: stop
            self.assertEqual(st["summary"]["h2"], 2)
            self.assertEqual(L.load(path)["state"], "done")
            self.assertEqual(L.LoadTest(path).state()["rows"], st["rows"])   # read back after a restart
            ok, _ = t.start("model", steps=(1, 2), seconds=30)
            time.sleep(0.3)
            t.cancel()
            t.thread.join(10)
            self.assertEqual(t.state()["state"], "cancelled")
            self.assertEqual(rs.call_count, 4)


class SetupPageTests(unittest.TestCase):
    def test_endpoints_and_no_start_while_running(self):
        from sotto_failsafe import launcher
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"SOTTO_CONFIG_DIR": d}):
            with open(os.path.join(d, "project.json"), "w") as f:
                json.dump({"ch": [{"insert": {"path": "/x/Denoise.vst3"}}]}, f)
            port = 18000 + os.getpid() % 1000
            res = {}
            with mock.patch.object(launcher, "PORT", port), mock.patch.object(launcher, "_LOADTEST", None):
                th = threading.Thread(target=lambda: res.update(argv=launcher.run_setup(False)), daemon=True)
                th.start()
                base = f"http://127.0.0.1:{port}"
                for _ in range(50):
                    try:
                        st = json.loads(urllib.request.urlopen(base + "/setup/loadtest", timeout=2).read())
                        break
                    except OSError:
                        time.sleep(0.1)
                self.assertEqual(st["insert"], "Denoise")
                self.assertIsNone(st["result"])
                busy = mock.Mock()
                busy.running.return_value = True
                launcher._LOADTEST = busy

                def post(path, body):
                    rq = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
                    return json.loads(urllib.request.urlopen(rq, timeout=5).read())
                r = post("/setup/start", {"simulate": True})
                self.assertFalse(r["ok"])
                self.assertIn("load test", r["error"])
                self.assertTrue(post("/quit", {})["ok"])
                busy.cancel.assert_called_once()
                th.join(10)
                self.assertIsNone(res.get("argv"))


if __name__ == "__main__":
    unittest.main()
