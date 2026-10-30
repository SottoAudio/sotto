"""Sotto Voce's log and the diagnostics file.

The log
  Everything Sotto Voce does that matters for troubleshooting goes to a rotating log file in the
  settings folder (logs/sotto-bench.log, 5 x 5 MB). Each line says when, which thread, and which
  function and line of code wrote it. Errors carry their full traceback (file and line of every
  call). Records go through a queue to a writer thread, so logging never makes a caller wait on
  the disk. The audio callback doesn't log at all: it keeps its last error, and the main loop
  logs it.

Crashes
  Uncaught errors in any thread are logged with their traceback. A crash that ends the app
  (including a failed start) is also written to logs/last-crash.json, which the setup page shows
  the next time. Hard crashes inside native code (a plugin, the speech model, the audio driver)
  can't be caught; faulthandler writes every thread's Python stack to logs/crash-native.log as
  the process dies.

The diagnostics file
  build_report() writes one plain-text file with everything needed to work out why something
  isn't working on a particular computer: the build, the computer (model, CPU cores, macOS,
  power, Rosetta), every audio device PortAudio and Core Audio can see, the settings, the
  engine, processing timings, a short speed test, the fail-safe state, recent events, what every
  thread is running right now (file and line), the end of the log, and any crash records.
  The console, the setup page and `Sotto Voce --diagnostics` all produce it.
"""
import faulthandler
import json
import logging
import logging.handlers
import os
import platform
import queue
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

LOG = logging.getLogger("sotto")
STARTED = time.time()
_listener = None
_native_file = None


def logs_dir():
    from .app import config_dir
    return os.path.join(config_dir(), "logs")


def log_path():
    return os.path.join(logs_dir(), "sotto-bench.log")


def build_info():
    """Which build this is: written by the build workflow, or read from git in a source checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, "build_info.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        pass
    info = {"source": "source checkout"}
    try:
        r = subprocess.run(["git", "-C", here, "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            info["commit"] = r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return info


# ------------------------------------------------------------------ logging
def setup_logging(level=logging.DEBUG, echo=True):
    """Starts the log (once per process). Safe to call again."""
    global _listener, _native_file
    if _listener is not None:
        return log_path()
    d = logs_dir()
    os.makedirs(d, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] "
                            "%(module)s.%(funcName)s:%(lineno)d  %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(log_path(), maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    handlers = [fh]
    if echo and sys.stderr is not None:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        handlers.append(sh)
    q = queue.SimpleQueue()                                   # unbounded: put() never waits
    LOG.setLevel(level)
    LOG.propagate = False
    for h in list(LOG.handlers):
        LOG.removeHandler(h)
    LOG.addHandler(logging.handlers.QueueHandler(q))
    _listener = logging.handlers.QueueListener(q, *handlers, respect_handler_level=True)
    _listener.start()
    try:
        _native_file = open(os.path.join(d, "crash-native.log"), "a", encoding="utf-8")
        _native_file.write(f"\n--- process {os.getpid()} started {datetime.now().isoformat(timespec='seconds')}\n")
        _native_file.flush()
        faulthandler.enable(_native_file, all_threads=True)
    except OSError:
        pass
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    b = build_info()
    LOG.info("==== Sotto Voce starting (pid %s) build %s | Python %s %s | %s %s", os.getpid(),
             b.get("commit", "?"), platform.python_version(), platform.machine(), platform.system(), platform.release())
    return log_path()


def stop_logging():
    """Writes out anything queued and closes the log files (before exiting or restarting)."""
    global _listener, _native_file
    if _listener is not None:
        _listener.stop()
        for h in _listener.handlers:
            h.close()
        _listener = None
    for h in list(LOG.handlers):
        LOG.removeHandler(h)
    if _native_file is not None:
        try:
            faulthandler.disable()
            _native_file.close()
        except Exception:
            pass
        _native_file = None


def _excepthook(kind, value, tb):
    LOG.critical("Uncaught error", exc_info=(kind, value, tb))
    sys.__excepthook__(kind, value, tb)


def _thread_excepthook(args):
    if args.exc_type is SystemExit:
        return
    LOG.critical("Uncaught error in thread %s", getattr(args.thread, "name", "?"),
                 exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


def record_crash(exc, where="start-up"):
    """Logs a crash that ends the app and keeps it for the setup page to show next time."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    LOG.critical("Sotto Voce stopped during %s: %r", where, exc, exc_info=(type(exc), exc, exc.__traceback__))
    try:
        os.makedirs(logs_dir(), exist_ok=True)
        with open(os.path.join(logs_dir(), "last-crash.json"), "w", encoding="utf-8") as f:
            json.dump({"time": datetime.now().isoformat(timespec="seconds"), "where": where,
                       "error": friendly_error(exc), "detail": repr(exc), "traceback": tb}, f, indent=2)
    except OSError:
        pass


def last_crash(clear=False):
    p = os.path.join(logs_dir(), "last-crash.json")
    try:
        with open(p, encoding="utf-8") as f:
            c = json.load(f)
    except (OSError, ValueError):
        return None
    if clear:
        try:
            os.replace(p, os.path.join(logs_dir(), "last-crash.seen.json"))
        except OSError:
            pass
    return c


def friendly_error(exc):
    """A plain sentence for the errors people actually hit."""
    s = str(exc)
    low = s.lower()
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (48, 98, 10048):
        return "Another program (probably another Sotto Voce) is using Sotto Voce's address (port 8766)."
    if "invalid number of channels" in low or "paInvalidChannelCount".lower() in low:
        return "The soundcard doesn't have that many channels. Set Dante Virtual Soundcard to 64 x 64."
    if "invalid sample rate" in low or "paInvalidSampleRate".lower() in low:
        return "The soundcard won't run at 48 kHz. Check the sample rate in Dante Controller or the VAD settings."
    if "error querying device" in low or "no such device" in low or "device unavailable" in low:
        return "The soundcard isn't there any more. Is Dante Virtual Soundcard (or Merging VAD) running?"
    if "unanticipated host error" in low:
        return "Core Audio refused to open the soundcard. Check microphone permission for Sotto Voce (System Settings, Privacy & Security, Microphone) and that no other app has it in exclusive use."
    return s or repr(exc)


# ------------------------------------------------------------------ the computer
def _run(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"({' '.join(cmd)}: {e!r})"


def system_info():
    info = {"python": sys.version.split()[0], "python_arch": platform.machine(), "executable": sys.executable,
            "frozen": bool(getattr(sys, "frozen", False)), "os": f"{platform.system()} {platform.release()}",
            "platform": platform.platform(), "cpu_count": os.cpu_count(), "pid": os.getpid(),
            "uptime_s": round(time.time() - STARTED, 1)}
    if sys.platform == "darwin":
        sc = lambda k: _run(["sysctl", "-n", k], 3)   # noqa: E731
        info.update({
            "macos": _run(["sw_vers", "-productVersion"], 3), "macos_build": _run(["sw_vers", "-buildVersion"], 3),
            "model": sc("hw.model"), "cpu": sc("machdep.cpu.brand_string"),
            "performance_cores": sc("hw.perflevel0.logicalcpu"), "efficiency_cores": sc("hw.perflevel1.logicalcpu"),
            "memory_gb": round(int(sc("hw.memsize") or 0) / 2 ** 30, 1) if sc("hw.memsize").isdigit() else "?",
            "rosetta": {"1": "yes (running translated: slow)", "0": "no"}.get(sc("sysctl.proc_translated"), "no"),
        })
        from . import macos
        info["power"] = macos.power_state()
        info["low_power_setting"] = next((ln.strip() for ln in _run(["pmset", "-g"], 3).splitlines() if "lowpowermode" in ln), "")
    elif sys.platform == "win32":
        info["cpu"] = platform.processor()
    else:
        try:
            with open("/proc/cpuinfo") as f:
                info["cpu"] = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), "?")
        except OSError:
            pass
    return info


def package_versions():
    out = {}
    for mod in ("numpy", "scipy", "onnxruntime", "sounddevice"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:
            out[mod] = f"not available ({type(e).__name__})"
    try:
        import sounddevice as sd
        out["portaudio"] = sd.get_portaudio_version()[1]
    except Exception:
        pass
    return out


def audio_devices():
    """Every device PortAudio sees, with the details that decide whether Sotto Voce can open it."""
    try:
        import sounddevice as sd
        apis = sd.query_hostapis()
        devs = []
        for i, d in enumerate(sd.query_devices()):
            devs.append({"index": i, "name": d["name"], "api": apis[d["hostapi"]]["name"],
                         "inputs": d["max_input_channels"], "outputs": d["max_output_channels"],
                         "default_rate": d["default_samplerate"],
                         "low_latency_ms": [round(1000 * d["default_low_input_latency"], 1),
                                            round(1000 * d["default_low_output_latency"], 1)]})
        try:
            default = list(sd.default.device)
        except Exception:
            default = None
        return {"ok": True, "devices": devs, "default": default}
    except Exception as e:
        return {"ok": False, "error": repr(e), "traceback": traceback.format_exc(), "devices": []}


def core_audio_report():
    """macOS's own view of the audio devices (catches devices PortAudio can't open)."""
    if sys.platform != "darwin":
        return ""
    return _run(["system_profiler", "SPAudioDataType"], 15)


def busy_snapshot():
    """Which of Sotto Voce's threads use CPU, and which programs are busiest (macOS and Linux)."""
    if sys.platform == "win32":
        return _run(["tasklist", "/v", "/fo", "csv"], 10)[:6000]
    me = str(os.getpid())
    out = ["== Sotto Voce threads (ps -M): one line per thread, CPU % in the CPU column"]
    out.append(_run(["ps", "-M", "-p", me], 5) if sys.platform == "darwin" else _run(["ps", "-L", "-o", "tid,pcpu,stat,comm", "-p", me], 5))
    out.append("\n== busiest programs")
    if sys.platform == "darwin":
        out.append("\n".join(_run(["top", "-l", "2", "-s", "1", "-n", "15", "-o", "cpu", "-stats", "pid,command,cpu,threads,state"], 15).splitlines()[-18:]))
    else:
        out.append("\n".join(_run(["ps", "-eo", "pid,pcpu,nlwp,comm", "--sort=-pcpu"], 5).splitlines()[:16]))
    return "\n".join(out)


def thread_cpu():
    """CPU seconds used so far by each of Sotto Voce's Python threads (Linux and macOS via the thread
    id where available), so a runaway thread shows up by name."""
    out = {}
    for t in threading.enumerate():
        out[t.name] = {"alive": t.is_alive(), "daemon": t.daemon, "native_id": getattr(t, "native_id", None)}
    return out


def thread_stacks():
    """What every thread is running right now, innermost call last."""
    names = {t.ident: t.name for t in threading.enumerate()}
    out = []
    for ident, frame in sys._current_frames().items():
        out.append(f"--- thread {names.get(ident, '?')} ({ident})")
        out.append("".join(traceback.format_stack(frame)).rstrip())
    return "\n".join(out)


def _tail(path, lines):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except OSError:
        return "(none)"


# ------------------------------------------------------------------ speed test
def speed_test(n_talkers=4, blocks=120):
    """Runs the processing chain offline (noise in, Full clean) on a fresh thread at three macOS
    priorities, measuring thread CPU time per 10 ms block for each stage. 'background' is pinned to
    the efficiency cores, so it shows what slow cores look like on this computer. If the live
    processing time is close to 'background', the live thread is on the slow cores."""
    try:
        from . import processing as P
        from . import macos
    except Exception as e:
        return {"error": f"processing isn't available in this build: {e!r}"}
    model = P.model_dir()
    if model is None:
        return {"error": "speech model not found"}
    import numpy as np
    from cct.stream_dfn import BatchedDFN
    dfn = BatchedDFN(model, n_talkers)
    results = {}

    def one(qos):
        note = macos.set_thread_qos(qos)[1] if qos else "not set"
        chains = [P.TalkerChain(f"t{i}", P.Settings()) for i in range(n_talkers)]
        ring = P.Ring(1, 12 * P.FS)
        rng = np.random.default_rng(0)
        t = {"front": 0.0, "model": 0.0, "back": 0.0, "wall": 0.0}
        x = np.zeros(1)
        loop_t0 = time.thread_time()
        for _ in range(300_000):                            # plain Python speed
            x[0] += 1.0
        py_ms = 1000 * (time.thread_time() - loop_t0)
        for b in range(blocks + 20):
            mics = rng.standard_normal((n_talkers, P.HOP)) * 0.05
            ref = rng.standard_normal(P.HOP) * 0.1
            ring.write(ref[None])
            w0, c0 = time.perf_counter(), time.thread_time()
            fr = [c.front(mics[i], ring) for i, c in enumerate(chains)]
            c1 = time.thread_time()
            Zc = dfn.process_specs(np.stack([f[0] for f in fr]))
            c2 = time.thread_time()
            for i, c in enumerate(chains):
                c.back(fr[i][1], Zc[i], fr[i][0])
            c3, w1 = time.thread_time(), time.perf_counter()
            if b >= 20:
                t["front"] += c1 - c0
                t["model"] += c2 - c1
                t["back"] += c3 - c2
                t["wall"] += w1 - w0
        r = {k: round(1000 * v / blocks, 2) for k, v in t.items()}
        r["cpu_total"] = round(r["front"] + r["model"] + r["back"], 2)
        r["python_loop_ms"] = round(py_ms, 1)
        r["qos"] = note
        results[qos or "default"] = r

    for qos in (["background", None, "user_interactive"] if sys.platform == "darwin" else [None]):
        th = threading.Thread(target=one, args=(qos,), name=f"speed-test-{qos or 'default'}")
        th.start()
        th.join(120)
    results["talkers"] = n_talkers
    results["blocks"] = blocks
    return results


# ------------------------------------------------------------------ the report
def _findings(app_state, sysinfo, devs, crash, speed):
    """Plain-English list of anything that looks wrong."""
    f = []
    if crash:
        f.append(f"Sotto Voce stopped unexpectedly on {crash.get('time')} during {crash.get('where')}: {crash.get('error')}")
    if sysinfo.get("rosetta", "").startswith("yes"):
        f.append("Sotto Voce is running translated by Rosetta. Download the Apple-silicon build.")
    pw = sysinfo.get("power") or {}
    if pw.get("low_power"):
        f.append("Low Power Mode is on: processing will be slow. Turn it off for live work.")
    if pw.get("thermal") and not str(pw["thermal"]).startswith("normal"):
        f.append(f"The Mac is {pw['thermal']}.")
    if pw.get("power") == "battery":
        f.append("Running on battery.")
    if devs and not devs.get("ok"):
        f.append(f"Could not list audio devices: {devs.get('error')}")
    elif devs:
        big = [d for d in devs["devices"] if d["inputs"] >= 64 and d["outputs"] >= 64]
        if not big:
            f.append("No audio device has 64 inputs and 64 outputs. Set Dante Virtual Soundcard to 64 x 64.")
    if app_state:
        pr = app_state.get("processing") or {}
        if pr.get("on") and pr.get("proc_ms", 0) > 9:
            f.append(f"Processing can't keep up: {pr.get('proc_ms')} ms per 10 ms block "
                     f"({pr.get('cpu_ms')} ms of it computing). Talkers are getting dry audio.")
        if pr.get("on") is False and pr.get("why"):
            f.append(f"Processing is off: {pr.get('why')}")
        for w in pr.get("worker_detail") or []:
            an = w.get("answer") or {}
            n = an.get("blocks") or 0
            if n < 500:
                continue
            slow = (an.get("over_10_ms") or 0) + (an.get("skipped") or 0)
            if slow / n > 0.002:
                f.append(f"Processing worker {w.get('worker')} answered more than 10 ms after the hand-over on "
                         f"{100 * slow / n:.1f} % of blocks (woke {an.get('wake_ms_avg')} ms after on average, worst "
                         f"{an.get('wake_ms_max')} ms; answered {an.get('answer_ms_avg')} ms after, worst "
                         f"{an.get('answer_ms_max')} ms). See 'Processing', worker_detail, answer.")
        if speed and not speed.get("error") and pr.get("cpu_ms"):
            bg, ui = speed.get("background", {}), speed.get("user_interactive", {})
            if bg and ui and ui.get("cpu_total"):
                wd = [w for w in (pr.get("worker_detail") or []) if w.get("cpu_ms")]
                live_per_talker = (max(w["cpu_ms"] / max(1, len(w["talkers"])) for w in wd) if wd else
                                   pr["cpu_ms"] / max(1, len(pr.get("talkers", [])) or 1))
                ui_per_talker = ui["cpu_total"] / speed["talkers"]
                bg_per_talker = bg["cpu_total"] / speed["talkers"]
                if bg_per_talker > 1.3 * ui_per_talker and abs(live_per_talker - bg_per_talker) < abs(live_per_talker - ui_per_talker):
                    f.append("The live processing runs at efficiency-core speed although the speed test reaches "
                             "performance-core speed: macOS is holding the processing thread on the slow cores.")
        eng = app_state.get("engine") or {}
        gaps = (app_state.get("callback") or {}).get("gaps") or {}
        if gaps.get("bunched"):
            f.append(f"The soundcard delivered {gaps['bunched']} of the last {gaps.get('blocks')} blocks bunched together "
                     f"(gaps {gaps.get('min_ms')} to {gaps.get('max_ms')} ms). Set a small soundcard buffer (Settings, "
                     "Audio engine), or Processing headroom: 30 ms through Sotto Voce.")
        d = gaps.get("delay_ms")
        if d and d[1] - d[0] > 15:
            f.append(f"The audio callback ran between {d[0]} and {d[1]} ms after the audio was captured: it is being held "
                     f"up by up to {d[1] - d[0]:.0f} ms (the longest gap between soundcard callbacks was "
                     f"{gaps.get('device_gap_max_ms')} ms). Check what else uses the soundcard, and the busy threads below.")
        pa = eng.get("pa_latency_ms") or {}
        if pa.get("in") is not None and pa["in"] + pa["out"] > 30:
            f.append(f"The soundcard adds {pa['in']} ms in and {pa['out']} ms out. A smaller soundcard buffer should "
                     "bring that down.")
        rg = eng.get("regroup") or {}
        if rg.get("underruns"):
            f.append(f"The soundcard ran out of audio {rg['underruns']} times with a {eng.get('io_buffer')}-sample buffer: "
                     "go one size up.")
        if app_state.get("callback_errors"):
            f.append(f"The audio callback raised {app_state['callback_errors']} errors (see the log).")
        if app_state.get("xruns"):
            f.append(f"The soundcard reported {app_state['xruns']} dropouts (over/underruns).")
        if app_state.get("health") and app_state.get("health") != "all green":
            f.append(f"Fail-safe checks not green: {app_state.get('health')}")
        lv = app_state.get("levels_in") or []
        if lv and max(lv) <= -100:
            f.append("Every input is silent. Check the Dante routing to Dante Virtual Soundcard in Dante Controller.")
    return f or ["Nothing obviously wrong."]


def build_report(app=None, setup_config=None, run_speed_test=True):
    """The diagnostics file as text. app: the running BenchApp, or None on the setup page."""
    t0 = time.time()
    LOG.info("Building diagnostics (speed test: %s)", run_speed_test)
    sections = []

    def sec(title, body):
        if not isinstance(body, str):
            body = json.dumps(body, indent=2, default=str)
        sections.append(f"\n==== {title} ====\n{body.rstrip()}\n")

    def safe(fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception:
            return {"error": traceback.format_exc()}

    sysinfo = safe(system_info)
    devs = safe(audio_devices)
    crash = last_crash()
    state = None
    if app is not None:
        state = safe(app.state)
        if isinstance(state, dict):
            state["callback_errors"] = getattr(app, "callback_errors", 0)
    speed = None
    if run_speed_test:
        n = len(getattr(getattr(app, "cfg", None), "talker_in", []) or []) or 4
        speed = safe(speed_test, n_talkers=n)

    found = _findings(state, sysinfo, devs, crash, speed)
    tr = getattr(app, "trace", None) if app is not None else None
    if tr is not None and tr.skipped:
        found.insert(0, f"Core Audio skipped {tr.skipped} soundcard cycles in {tr.skip_events} places since the engine "
                        f"started ({1000 * tr.skipped * tr.io / tr.fs:.0f} ms of audio lost): heard as clicks and "
                        f"dropouts. See 'Audio timing in detail'.")
    if tr is not None and getattr(tr, "outages", 0):
        found.insert(0, f"The soundcard stopped {tr.outages} time(s) ({tr.outage_s:.1f} s without audio in all): "
                        "Dante Virtual Soundcard restarted or lost its network. See 'Where the problem is'.")
    ff = getattr(app, "finder", None) if app is not None else None
    if ff is not None and ff.root:
        st = ff.state()
        found.insert(0, st["headline"] + (f" Look at: {st['look']}" if st["look"] else ""))
    sh = getattr(app, "shadow", None) if app is not None else None
    if sh is not None and sh.blocks and sh.matched < sh.blocks:
        found.insert(0, f"The C core differed from the Python core on {sh.blocks - sh.matched} of {sh.blocks} blocks. "
                        "See 'C core (shadow)'.")
    sec("What looks wrong", "\n".join(f"- {x}" for x in found))
    if ff is not None:
        sec("Where the problem is (each link in the chain, upstream first)", safe(ff.report))
    ce = getattr(app, "cengine", None) if app is not None else None
    if ce is not None and ce.h:
        sec("Engine in C (stage 3): the 10 ms engine on its own real-time thread",
            safe(lambda: {"thread": ce.thread_note(), **ce.stats()}))
    if sh is not None:
        sec("C core (shadow): the fail-safe core in C, compared with the Python one on every block", safe(sh.report))
    sec("Build", {**build_info(), "packages": safe(package_versions)})
    sec("Computer", sysinfo)
    sec("Audio devices (PortAudio, what Sotto Voce can open)", devs)
    if sys.platform == "darwin":
        from . import coreaudio
        sec("Core Audio buffers and latency (frames at the device's rate)", safe(coreaudio.devices))
        sec("Audio devices (Core Audio, system_profiler)", safe(core_audio_report))
    try:
        from .app import config_dir
        with open(os.path.join(config_dir(), "config.json")) as f:
            saved = json.load(f)
    except Exception as e:
        saved = f"(no saved settings: {e!r})"
    sec("Saved setup choices", saved)
    if setup_config is not None:
        sec("Setup page now", setup_config)
    if app is not None:
        sec("Command line", " ".join(sys.argv))
        a = getattr(app, "a", None)
        sec("Options", vars(a) if a is not None else {})
        if isinstance(state, dict):
            sec("Engine", {"engine": state.get("engine"), "latency": state.get("latency_note"), "xruns": state.get("xruns"),
                           "callback_errors": state.get("callback_errors"), "power": state.get("power")})
            sec("Processing", state.get("processing"))
            tr = getattr(app, "trace", None)
            if tr is not None:
                sec("Audio timing in detail (every soundcard cycle and block, last ~40 s)", safe(tr.summary))
                sec("Engine stalls (blocks over 8 ms since start, and what else was running)", safe(tr.stall_report))
                sec("Audio timing, last 10 s (CSV: one row per soundcard cycle and per block; times in seconds)",
                    safe(lambda: tr.csv(10)))
    if speed is not None:
        sec("Speed test (thread CPU ms per 10 ms block; 'background' = efficiency cores)", speed)
    if isinstance(state, dict):
        sec("Fail-safe", {k: state.get(k) for k in ("armed", "heartbeat_on", "reason", "health", "green_for_s",
                                                    "can_arm", "probes", "faults", "talkers", "channels", "rrcs", "frame", "solo")})
        sec("Recent events", "\n".join("  ".join(str(c) for c in r) for r in state.get("events", [])) or "(none)")
        sec("Network ports (ifconfig -a)", _run(["/sbin/ifconfig", "-a"]) if sys.platform == "darwin" else "(macOS only)")
    sec("Busy threads and programs", safe(busy_snapshot))
    sec("Python threads", safe(thread_cpu))
    sec("Threads right now (innermost call last)", safe(thread_stacks))
    sec("Log (last 1500 lines)", _tail(log_path(), 1500))
    sec("Native crash log (last 200 lines)", _tail(os.path.join(logs_dir(), "crash-native.log"), 200))
    if crash:
        sec("Last crash", crash)
    if isinstance(state, dict):
        sec("Full state (JSON)", state)
    head = (f"SOTTO VOCE DIAGNOSTICS\nCreated {datetime.now().isoformat(timespec='seconds')} in {time.time() - t0:.1f} s"
            f" | build {build_info().get('commit', '?')} | {sysinfo.get('model', '') if isinstance(sysinfo, dict) else ''}\n"
            "Send this file to whoever is helping you troubleshoot. It contains no passwords; it does contain"
            " your device names, channel settings, operator names and file paths.\n")
    LOG.info("Diagnostics built in %.1f s", time.time() - t0)
    return head + "".join(sections)


def report_filename():
    return f"sotto-diagnostics-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"


def save_report(text, folder=None):
    """Keeps a copy in the logs folder (and folder, if given). Returns the paths written."""
    name = report_filename()
    paths = []
    for d in filter(None, [logs_dir(), folder]):
        try:
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, name)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            paths.append(p)
        except OSError:
            pass
    return paths


def downloads_dir():
    d = os.path.join(os.path.expanduser("~"), "Downloads")
    return d if os.path.isdir(d) else os.path.expanduser("~")
