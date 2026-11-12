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


class PerWorkerTests(unittest.TestCase):
    """The load test plans workers exactly as Sotto Voce does (workers.plan_groups)."""

    def test_plan(self):
        from sotto_failsafe import processing as P, workers as W
        full = [P.Settings(mode="full")] * 8
        self.assertEqual(W.plan_groups(full, 9), [[i] for i in range(8)])            # a worker each
        self.assertEqual(W.plan_groups(full, 9, 3), [[0, 1, 2], [3, 4, 5], [6, 7]])
        self.assertLessEqual(len(W.plan_groups([P.Settings(mode="full")] * 64, 9, 1)), 9)  # then the groups grow
        self.assertEqual(W.plan_groups(full[:1], 9, 4), [[0]])
        mixed = [P.Settings(mode="aec")] * 3 + [P.Settings(mode="full")] * 2
        self.assertEqual(W.plan_groups(mixed, 9, 2), [[0, 1, 2, 3], [4]])           # weight 3 + 3 = 6 a group


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

    def test_talkers_sharing_a_worker(self):
        r = L.run_step(4, None, mode="full", seconds=0.5, track_delay=False, per_worker=2)
        self.assertNotIn("error", r, r)
        self.assertEqual(r["groups"], [2, 2] if (os.cpu_count() or 2) >= 5 else [4])
        self.assertEqual(sum(r["groups"]), 4)

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
            self.assertEqual(rs.call_args.kwargs["per_worker"], 1)
            ok, _ = t.start("model", steps=(1, 2), seconds=0.05, per_worker=3, model_on="coreml")   # another setting
            t.thread.join(30)
            self.assertEqual(rs.call_args.kwargs["per_worker"], 3)
            self.assertEqual(rs.call_args.kwargs["model_on"], "coreml")
            cmp = L.compare(t.state())
            self.assertEqual([(h["per_worker"], h["model_on"]) for h in cmp], [(1, "CPU"), (3, "Neural Engine (Core ML)")])
            self.assertEqual((cmp[0]["h2"], cmp[1]["h2"]), (2, 2))
            self.assertEqual(len(L.compare(L.LoadTest(path).state())), 2)       # kept after a restart
            self.assertIn("Every setting tried", L.report(t.state()))


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
                busy.start.return_value = (False, "The load test is already running.")
                busy.state.return_value = None
                launcher._LOADTEST = busy

                def post(path, body):
                    rq = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
                    return json.loads(urllib.request.urlopen(rq, timeout=5).read())
                r = post("/setup/loadtest", {"per_worker": 3, "max_talkers": 16, "insert": True})
                self.assertFalse(r["ok"])
                kw = busy.start.call_args.kwargs
                self.assertEqual((kw["per_worker"], kw["max_talkers"], L.insert_name(kw["insert"])), (3, 16, "Denoise"))
                post("/setup/loadtest", {"per_worker": 5})
                self.assertEqual(busy.start.call_args.kwargs["per_worker"], 1)   # not one of the choices
                self.assertEqual(busy.start.call_args.kwargs["model_on"], "cpu")
                post("/setup/loadtest", {"model_on": "coreml"})
                self.assertEqual(busy.start.call_args.kwargs["model_on"], "coreml")
                self.assertIn("coreml", json.loads(urllib.request.urlopen(base + "/setup/loadtest", timeout=2).read()))
                r = post("/setup/start", {"simulate": True})
                self.assertFalse(r["ok"])
                self.assertIn("load test", r["error"])
                self.assertTrue(post("/quit", {})["ok"])
                busy.cancel.assert_called_once()
                th.join(10)
                self.assertIsNone(res.get("argv"))


if __name__ == "__main__":
    unittest.main()


class PerWorkerSettingTests(unittest.TestCase):
    """The choice the load test informs, as a Sotto Voce setting: saved, passed on, checked."""

    def test_setup_choice_becomes_an_option(self):
        from sotto_failsafe import launcher, app as A
        cfg = {**launcher.DEFAULTS, "device": "3"}
        self.assertNotIn("--per-worker", launcher.argv_from(cfg, False))              # 1: the default
        argv = launcher.argv_from({**cfg, "per_worker": "3"}, False)
        self.assertEqual(argv[argv.index("--per-worker") + 1], "3")
        self.assertEqual(A.parse_args(argv).per_worker, 3)
        import contextlib
        import io
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            A.parse_args(["--simulate", "--per-worker", "5"])

    def test_console_change_is_checked_and_restarts(self):
        from sotto_failsafe import app as A
        # ignore_cleanup_errors: Windows won't delete the event log the app still has open
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d, mock.patch.dict(os.environ, {"SOTTO_CONFIG_DIR": d}):
            a = A.parse_args(["--simulate", "--no-processing", "--port", "0", "--out-dir", d])
            b = A.BenchApp(a)
            self.assertFalse(b.reconfigure({"per_worker": 5})["ok"])
            self.assertTrue(b.reconfigure({"per_worker": 3})["ok"])
            self.assertEqual(A.parse_args(b.restart_argv).per_worker, 3)
            self.assertTrue(b.quit_ev.is_set())


class ModelOnTests(unittest.TestCase):
    """The speech model on the Neural Engine (Core ML): an experiment in the load test only."""

    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))

    def test_providers(self):
        from cct import stream_dfn as D
        ort = mock.Mock()
        ort.get_available_providers.return_value = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        with mock.patch.dict(os.environ, {"SOTTO_MODEL_ON": "coreml"}):
            self.assertEqual(D.model_providers(ort)[0][0], "CoreMLExecutionProvider")
            ort.get_available_providers.return_value = ["CPUExecutionProvider"]          # not on this computer
            self.assertEqual(D.model_providers(ort), ["CPUExecutionProvider"])
        with mock.patch.dict(os.environ, {"SOTTO_MODEL_ON": "cpu"}):
            ort.get_available_providers.return_value = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            self.assertEqual(D.model_providers(ort), ["CPUExecutionProvider"])        # never unless asked

    def test_core_ml_refusing_the_model_falls_back_to_the_cpu(self):
        from sotto_failsafe import processing as P
        from cct import stream_dfn as D
        model = P.model_dir()
        if model is None:
            self.skipTest("speech model not downloaded")
        import onnxruntime as real

        class Sess:
            def __init__(self, path, so, providers):
                if any(isinstance(p, tuple) for p in providers):
                    raise RuntimeError("Core ML can't run this model")
                self.s = real.InferenceSession(path, so, providers=["CPUExecutionProvider"])

            def __getattr__(self, k):
                return getattr(self.s, k)
        fake = mock.Mock(SessionOptions=real.SessionOptions, InferenceSession=Sess,
                         get_available_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"])
        with mock.patch.dict(os.environ, {"SOTTO_MODEL_ON": "coreml"}), mock.patch.dict(sys.modules, {"onnxruntime": fake}):
            d = D.BatchedDFN(model, 2)
        self.assertEqual(d.model_on, "CPU")
        self.assertIn("Core ML can't run", d.model_on_error)

    def test_workers_start_with_the_choice_and_it_is_put_back(self):
        from sotto_failsafe import workers as W
        seen = []

        class FakePool:
            def __init__(self, *a, **k):
                seen.append(os.environ.get("SOTTO_MODEL_ON"))
                self.workers, self.ok, self.why = [], False, "test"

            def stop(self):
                pass
        with mock.patch.object(W, "Pool", FakePool), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SOTTO_MODEL_ON", None)
            r = L.run_step(2, "m", model_on="coreml", seconds=0.1)
            self.assertEqual(seen, ["coreml"])
            self.assertNotIn("SOTTO_MODEL_ON", os.environ)
            self.assertEqual(r["error"], "test")
