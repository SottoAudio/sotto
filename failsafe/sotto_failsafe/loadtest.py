"""Processing load test: how many talkers this computer can clean, and what each one costs.

Run from the setup page, while the engine is stopped, so the test has the computer to itself.
It starts the processing workers exactly as Sotto Voce does (one worker per Full clean talker
while there are cores for it, then groups), hands them a block every 10 ms, and checks every
block's result at the three points the engine would take it:

  * same block (10 ms through Sotto Voce): within 8 ms of the hand-over, the longest the engine
    waits for its own block at the 30 ms engine safety margin;
  * 20 ms through: before the next block is handed over;
  * 30 ms through: before the one after that.

A setting fits when no more than 0.5 % of blocks are late and never three in a row (three in a
row is what drops a talker to dry in the show). It steps up the talker count until even 30 ms
through doesn't fit, or it reaches the most asked for.

The input is speech-like noise on every talker with programme (PGM) under it, so the echo
canceller and the speech model both have work to do. The workers' own CPU time gives the cost
per talker. Nothing here touches the soundcard, the network or the fail-safe.
"""

import json
import math
import os
import threading
import time

import numpy as np

from .diagnostics import LOG

STEPS = (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)
WARM_S = 1.5                # after the workers are up, before counting: the model's first blocks are slow
MEASURE_S = 10.0            # counted, per step
SAME_BLOCK_MS = 8.0         # the engine waits up to this for its own block (engine margin 30 ms)
OK_RATE = 0.005             # late blocks allowed: 0.5 %
RUN_LIMIT = 3               # this many late in a row drops the talker to dry: never allowed
FITS = (("same", "10 ms through (same block)"), ("h1", "20 ms through"), ("h2", "30 ms through"))
PER_WORKER = (1, 2, 3, 4, 6)   # talkers sharing one worker: 1 is how Sotto Voce runs Full clean now
PERIOD = 0.010
FS = 48000
HOP = 480


def first_insert(project):
    """The first plugin insert in the project (any talker or operator), to load-test with, or None."""
    p = project or {}
    for src in list(p.get("operators", [])) + list(p.get("ch", [])):
        ins = (src or {}).get("insert") or {}
        if ins.get("path"):
            return {"path": ins["path"], "class": ins.get("class", ""), "on": True, "state": ins.get("state", "")}
    return None


def insert_name(ins):
    if not ins:
        return ""
    return os.path.splitext(os.path.basename(str(ins.get("path", "")).rstrip("/\\")))[0]


class _Late:
    """Late blocks at one take-off point."""

    def __init__(self):
        self.n = self.late = self.run = self.max_run = 0

    def add(self, ok):
        self.n += 1
        if ok:
            self.run = 0
        else:
            self.late += 1
            self.run += 1
            self.max_run = max(self.max_run, self.run)

    def report(self):
        rate = self.late / self.n if self.n else 1.0
        return {"blocks": self.n, "late": self.late, "late_pct": round(100 * rate, 2), "max_run": self.max_run,
                "fits": bool(self.n) and rate <= OK_RATE and self.max_run < RUN_LIMIT}


def _signals(n, seconds=2.0, seed=3):
    """A loop of test audio: per talker, bursts of band-limited noise (syllables and gaps) with the
    PGM leaking in as echo; and the PGM itself. float32, (n, samples) and (samples,)."""
    rng = np.random.default_rng(seed)
    m = int(seconds * FS) // HOP * HOP
    k = np.ones(24, np.float32) / 24
    pgm = np.convolve(rng.standard_normal(m).astype(np.float32), k, "same") * 0.1
    mics = np.empty((n, m), np.float32)
    for i in range(n):
        x = np.convolve(rng.standard_normal(m).astype(np.float32), k, "same")
        env = np.repeat(rng.random(m // 2400 + 1) > 0.45, 2400)[:m].astype(np.float32)
        d = 240 + 48 * (i % 8)                                  # 5-13 ms of acoustic echo
        mics[i] = 0.08 * x * env + 0.3 * np.concatenate([np.zeros(d, np.float32), pgm[:m - d]])
    return mics, pgm


def _wait_until(t):
    """Sleep to just before t, then spin: Python's sleep alone can wake a millisecond late."""
    while True:
        left = t - time.perf_counter()
        if left <= 0:
            return
        if left > 0.0012:
            time.sleep(left - 0.001)


def _ready(pool, j):
    """True if every worker has written block j's result."""
    if j < 0:
        return False
    for w in pool.workers:
        if int(w.outs[j % len(w.outs)][0][0]) != j:
            return False
    return True


def workers_for(n, per_worker, cores=None):
    """How many workers for n talkers at per_worker each, with the same limit Sotto Voce has (all
    the cores but three): past it, the groups grow. None: Sotto Voce's own plan."""
    if not per_worker:
        return None
    cores = cores or os.cpu_count() or 2
    return max(1, min(math.ceil(n / per_worker), max(1, cores - 3)))


def run_step(n, model, insert=None, mode="full", seconds=MEASURE_S, track_delay=True, cancel=None, workers=None,
             per_worker=None):
    """One talker count. Returns the row for the table. per_worker: talkers sharing a worker (the
    speech model runs them together, which costs less per talker but takes longer per block)."""
    workers = workers or workers_for(n, per_worker)
    from . import processing as P
    from . import workers as W
    settings = [P.Settings(mode=mode, insert=dict(insert) if insert else None) for _ in range(n)]
    labels = [f"Load {i + 1}" for i in range(n)]
    t0 = time.monotonic()
    pool = W.Pool(labels, settings, model=model, workers=workers, track_delay=track_delay, ready_timeout=120.0,
                  headroom=2)                    # the workers keep the two newest blocks, as at 30 ms through
    row = {"talkers": n, "workers": len(pool.workers), "groups": [len(w.rows) for w in pool.workers],
           "startup_s": round(time.monotonic() - t0, 1)}
    try:
        if not pool.ok:
            row.update(error=pool.why or "the processing workers didn't start")
            return row
        if insert:
            st = pool.state()
            bad = [t.get("insert_error") for t in st.get("talkers", []) if t.get("insert_error")]
            if bad:
                row["insert_error"] = bad[0]
        mics, pgm = _signals(n)
        m = mics.shape[1] // HOP
        warm = int(WARM_S / PERIOD)
        total = warm + int(seconds / PERIOD)
        late = {"same": _Late(), "h1": _Late(), "h2": _Late()}
        driver_late = 0
        t_start = time.perf_counter() + 0.02
        for k in range(total):
            if cancel is not None and cancel.is_set():
                row["cancelled"] = True
                return row
            t_k = t_start + k * PERIOD
            _wait_until(t_k)
            if time.perf_counter() - t_k > 0.002:
                driver_late += 1
            counting = k >= warm
            if counting:                                 # taken one and two blocks after the hand-over
                late["h1"].add(_ready(pool, k - 1))
                late["h2"].add(_ready(pool, k - 2))
            s = (k % m) * HOP
            pool.exchange(mics[:, s:s + HOP], pgm[s:s + HOP])
            _wait_until(t_k + SAME_BLOCK_MS / 1000)
            if counting:
                late["same"].add(_ready(pool, k))
        st = pool.state()
        det = st.get("worker_detail") or []
        cpu = [float(d.get("cpu_ms") or 0) for d in det]
        wall = [float(d.get("proc_ms") or 0) for d in det]
        row.update({
            "cpu_ms_per_block": round(sum(cpu), 2),               # all workers, per 10 ms block
            "cpu_ms_per_talker": round(sum(cpu) / n, 2),
            "cores_busy": round(sum(cpu) / (1000 * PERIOD), 2),
            "slowest_worker_ms": round(max(wall), 2) if wall else None,   # wall time per block, average
            "slowest_worker_max_ms": round(float(st.get("proc_ms_max") or 0), 2),
            "speech_model": bool(st.get("model")),
            "errors": sum(int(d.get("errors") or 0) for d in det),
            "restarts": sum(int(d.get("restarts") or 0) for d in det),
            "test_timer_late": driver_late,
            **{key: late[key].report() for key, _ in FITS},
        })
        return row
    finally:
        pool.stop()


def summarise(rows):
    """The headline: the most talkers that fit at each delay, and the cost per talker."""
    good = [r for r in rows if "error" not in r and not r.get("cancelled") and "h2" in r]
    out = {}
    for key, what in FITS:
        most = 0
        for r in good:                                   # the most before the first that doesn't fit
            if not r[key]["fits"]:
                break
            most = r["talkers"]
        out[key] = most
    costs = [r["cpu_ms_per_talker"] for r in good if r["talkers"] >= 2 and r["h2"]["fits"]] or \
            [r["cpu_ms_per_talker"] for r in good]
    cost = round(float(np.median(costs)), 2) if costs else None
    out["cpu_ms_per_talker"] = cost
    out["cores"] = os.cpu_count()
    tried = max((r["talkers"] for r in good), default=0)
    out["tried_up_to"] = tried
    if not good:
        out["headline"] = "The load test didn't get a result."
        return out
    reached = all(r["h2"]["fits"] for r in good)
    parts = [f"{out[k]} talker{'' if out[k] == 1 else 's'} at {what}" if out[k] else f"no talkers at {what}"
             for k, what in FITS]
    out["headline"] = (f"This computer kept up with {', '.join(parts)}"
                       + (f" (the most tried was {tried}, and it still kept up)" if reached else "")
                       + "." + (f" Each Full clean talker used about {cost:g} ms of CPU per 10 ms block, "
                                f"so one core does about {10 / cost:.1f} talkers' worth." if cost else ""))
    return out


class LoadTest:
    """One run at a time, on its own thread, for the setup page: start(), state(), cancel()."""

    def __init__(self, path=None):
        self.lock = threading.Lock()
        self.cancel_ev = threading.Event()
        self.thread = None
        self.path = path                          # where the last result is saved, for the diagnostics
        self.result = load(path) if path else None
        self.history = dict((self.result or {}).get("history") or {})   # the last run of each setting

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, model, insert=None, max_talkers=64, seconds=MEASURE_S, steps=STEPS, mode="full",
              track_delay=True, workers=None, per_worker=1):
        with self.lock:
            if self.running():
                return False, "The load test is already running."
            if model is None and mode == "full":
                return False, "The speech model isn't in this build, so there's nothing to load-test."
            plan = [n for n in steps if n <= max_talkers] or [max_talkers]
            self.cancel_ev.clear()
            self.result = {"state": "running", "started": time.time(), "plan": plan, "rows": [], "step": None,
                           "insert": insert_name(insert), "seconds_per_step": seconds, "mode": mode,
                           "cores": os.cpu_count(), "per_worker": per_worker, "history": dict(self.history)}
            self.thread = threading.Thread(target=self._run, args=(plan, model, insert, seconds, mode, track_delay,
                                                                   workers, per_worker), name="load-test", daemon=True)
            self.thread.start()
            return True, ""

    def cancel(self):
        self.cancel_ev.set()

    def _set(self, **kw):
        with self.lock:
            self.result.update(kw)

    def _run(self, plan, model, insert, seconds, mode, track_delay, workers, per_worker):
        LOG.info("Load test: %s talkers, %s per worker, %s, %.0f s each", plan, per_worker,
                 f"insert {insert_name(insert)}" if insert else "no insert", seconds)
        rows = []
        state, why = "done", ""
        try:
            from . import macos                           # the hand-overs on time, as the engine's are
            self._set(timer_thread=macos.realtime_thread(10.0, 3.0, 10.0)[1])
        except Exception:
            pass
        try:
            for n in plan:
                if self.cancel_ev.is_set():
                    state = "cancelled"
                    break
                self._set(step=n)
                row = run_step(n, model, insert=insert, mode=mode, seconds=seconds, track_delay=track_delay,
                               cancel=self.cancel_ev, workers=workers, per_worker=per_worker)
                if row.get("cancelled"):
                    state = "cancelled"
                    break
                rows.append(row)
                LOG.info("Load test %d talkers: %s", n, json.dumps(row))
                self._set(rows=list(rows), summary=summarise(rows))
                if "error" in row:
                    state, why = "failed", row["error"]
                    break
                if not row["h2"]["fits"]:                # past what this computer can do
                    break
        except Exception as e:
            LOG.exception("Load test failed")
            state, why = "failed", repr(e)
        self._set(state=state, error=why, step=None, rows=rows, summary=summarise(rows), finished=time.time())
        if rows and state != "cancelled":
            sm = summarise(rows)
            key = f"{per_worker or 1}|{insert_name(insert)}"
            with self.lock:
                self.history[key] = {"per_worker": per_worker or 1, "insert": insert_name(insert), "state": state,
                                     "started": self.result["started"], "cpu_ms_per_talker": sm["cpu_ms_per_talker"],
                                     "tried_up_to": sm["tried_up_to"], **{k: sm[k] for k, _ in FITS}}
                self.result["history"] = dict(self.history)
        if self.path:
            save(self.path, self.state())
        LOG.info("Load test %s: %s", state, (self.result.get("summary") or {}).get("headline"))

    def state(self):
        with self.lock:
            return None if self.result is None else json.loads(json.dumps(self.result))


def save(path, result):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    except OSError:
        LOG.exception("Couldn't save the load test result")


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def result_path():
    from .app import config_dir
    return os.path.join(config_dir(), "loadtest.json")


def report(result):
    """Plain text for the diagnostics file."""
    if not result:
        return "(not run)"
    s = result.get("summary") or {}
    lines = [s.get("headline", ""), "",
             f"State: {result.get('state')}{(' (' + result['error'] + ')') if result.get('error') else ''}; "
             f"insert: {result.get('insert') or 'none'}; {result.get('seconds_per_step')} s per step; "
             f"{result.get('cores')} cores; run {time.strftime('%Y-%m-%d %H:%M', time.localtime(result.get('started', 0)))}",
             f"Talkers per worker: {result.get('per_worker') or 1}",
             "",
             "talkers workers  CPU ms/block  per talker  cores busy  slowest worker ms (avg/max)  "
             "late %: same block / 20 ms / 30 ms  (longest run)"]
    for r in result.get("rows", []):
        if "error" in r:
            lines.append(f"{r['talkers']:>7}  {r.get('workers', ''):>6}  {r['error']}")
            continue
        lt = "  ".join(f"{r[k]['late_pct']:>5.2f}{'' if r[k]['fits'] else '!'} ({r[k]['max_run']})" for k, _ in FITS)
        lines.append(f"{r['talkers']:>7}  {r['workers']:>6}  {r['cpu_ms_per_block']:>12.2f}  {r['cpu_ms_per_talker']:>10.2f}  "
                     f"{r['cores_busy']:>10.2f}  {r['slowest_worker_ms']:>13.2f} / {r['slowest_worker_max_ms']:<10.2f}  {lt}"
                     + (f"  timer late {r['test_timer_late']}" if r.get("test_timer_late") else "")
                     + (f"  errors {r['errors']}" if r.get("errors") else "")
                     + (f"  insert: {r['insert_error']}" if r.get("insert_error") else ""))
    lines.append("")
    lines.append("! = doesn't fit (over 0.5 % late, or 3 late in a row). Late is checked where the engine takes "
                 "each block: 8 ms after hand-over (same block), and one and two blocks later.")
    hist = compare(result)
    if hist:
        lines += ["", "Every setting tried (the last run of each):",
                  "per worker  insert              most talkers at 10 / 20 / 30 ms   CPU ms per talker  run"]
        for h in hist:
            lines.append(f"{h['per_worker']:>10}  {(h['insert'] or 'none')[:18]:<18}  {h['same']:>14} / {h['h1']:>3} / {h['h2']:>3}"
                         f"   {h['cpu_ms_per_talker'] if h['cpu_ms_per_talker'] is not None else '':>17}  "
                         f"{time.strftime('%d %b %H:%M', time.localtime(h.get('started', 0)))}")
    return "\n".join(lines)


def compare(result):
    """The last run of each setting, by talkers per worker then insert."""
    hist = list(((result or {}).get("history") or {}).values())
    return sorted(hist, key=lambda h: (h.get("per_worker") or 1, h.get("insert") or ""))
