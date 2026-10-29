"""Processing in separate worker processes, so the audio engine never shares a core (or Python's
global lock) with the heavy work.

Sotto Voce's own process keeps the audio engine: the soundcard callback (Core Audio's real-time thread),
HEARTBEAT, PROBE, the fail-safe checks and the phones bus. The talkers are split into groups and
each group is processed by its own worker process, which runs processing.Chains exactly as the
in-process Processor does.

Per 10 ms block the callback (exchange()) copies the group's mic rows and PGM into the worker's
shared-memory input slot and posts a semaphore; it never waits. It takes back the result for the
block before, if the worker has written it. A missing result is a miss: that group's talkers get
dry at the same delay, and three misses in a row make processing unhealthy, which disarms Sotto Voce
(latched) so the frame goes to dry, exactly as with the in-process Processor.

Memory ordering: the shared-memory slots carry a block tag. The writer stores the data, then posts a
semaphore (a full memory barrier), then stores the tag. The reader loads the tag, then does a
non-blocking semaphore operation (a barrier) before reading the data, then checks the tag again.

Everything else (settings, inserts, plugin controls and saved settings) goes over a control pipe
to a control thread in the worker. Faults, state snapshots and log records come back on one queue.
A worker that dies is logged, its talkers are faulted (tripped to dry), and it is restarted;
re-arming stays manual.
"""
import atexit
import contextlib
import itertools
import logging
import logging.handlers
import math
import multiprocessing as mp
import os
import threading
import time
from multiprocessing import shared_memory

import numpy as np

from .diagnostics import LOG

HOP = 480
RAMP = 120                                  # 2.5 ms: the fade when a talker switches source
SLOTS = 4
HEADER = 16                                     # int64 latest tag, int64 headroom (blocks)
MAX_HEADROOM = 3


def auto_workers(n_talkers, cores=None):
    """Up to three talkers per worker, leaving cores for the audio engine, Dante and the system."""
    cores = cores or os.cpu_count() or 2
    return max(1, min(math.ceil(n_talkers / 3), max(1, cores - 3)))


def split(n, k):
    """n talkers into k contiguous groups, sizes differing by at most one."""
    k = max(1, min(k, n))
    base, extra = divmod(n, k)
    out, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        out.append(list(range(start, start + size)))
        start += size
    return out


class _Layout:
    def __init__(self, g):
        self.g = g
        self.in_slot = 8 + 8 + 4 * HOP * (g + 1)            # tag, spare, mics (g, HOP) f32, ref (HOP) f32
        self.out_slot = 8 + 8 + 4 * HOP * 2 * g             # tag, spare, outs (g, HOP) f32, drys (g, HOP) f32
        self.in_base = HEADER
        self.out_base = HEADER + SLOTS * self.in_slot
        self.size = self.out_base + SLOTS * self.out_slot

    def views(self, buf):
        g = self.g
        latest = np.ndarray((2,), np.int64, buf, 0)
        ins = []
        for s in range(SLOTS):
            o = self.in_base + s * self.in_slot
            ins.append((np.ndarray((2,), np.int64, buf, o), np.ndarray((g, HOP), np.float32, buf, o + 16),
                        np.ndarray((HOP,), np.float32, buf, o + 16 + 4 * HOP * g)))
        outs = []
        for s in range(SLOTS):
            o = self.out_base + s * self.out_slot
            outs.append((np.ndarray((2,), np.int64, buf, o), np.ndarray((g, HOP), np.float32, buf, o + 16),
                         np.ndarray((g, HOP), np.float32, buf, o + 16 + 4 * HOP * g)))
        return latest, ins, outs


# ---------------------------------------------------------------------------- worker process
class _UpHandler(logging.handlers.QueueHandler):
    def __init__(self, q, wid):
        super().__init__(q)
        self.wid = wid

    def enqueue(self, record):
        self.queue.put_nowait((self.wid, "log", record))


def _worker_main(wid, names, settings, model, shm_name, g, in_sem, out_sem, ctrl, up, track_delay):
    """Runs in the worker process."""
    import faulthandler
    import sys
    LOG.handlers[:] = [_UpHandler(up, wid)]
    LOG.setLevel(logging.DEBUG)
    LOG.propagate = False
    try:
        from .diagnostics import logs_dir
        os.makedirs(logs_dir(), exist_ok=True)
        _native = open(os.path.join(logs_dir(), "crash-native.log"), "a", encoding="utf-8")
        _native.write(f"\n--- processing worker {wid + 1} (process {os.getpid()}) started\n")
        _native.flush()
        faulthandler.enable(_native, all_threads=True)
    except OSError:
        pass
    try:
        from . import macos, processing, vst3
        macos.keep_responsive("Sotto Voce processing worker")
        note = "performance cores (processing thread)"
        try:
            lib = vst3.load_library()
        except OSError:
            LOG.exception("Worker %d: the plugin host library could not be loaded", wid + 1)
            lib = None
        work = processing.Chains(names, settings, model=model, vst3_lib=lib)
        work.thread_note = note
        shm = shared_memory.SharedMemory(name=shm_name)   # the parent owns (and unlinks) it
        latest, ins, outs = _Layout(g).views(shm.buf)
        tracker = None
        if track_delay:
            from cct.engine import DelayTracker
            tracker = DelayTracker(work)
    except BaseException as e:
        LOG.exception("Worker %d could not start", wid + 1)
        up.put((wid, "failed", repr(e)))
        return
    stop = threading.Event()
    errors = [0]

    def send_state():
        up.put((wid, "state", {**work.timing(), "talkers": work.talker_states(), "errors": errors[0],
                               "model": work.dfn is not None}))

    def audio():
        """The processing, on its own thread at the highest everyday priority. Plugins are only
        run here (process); they're created, changed and destroyed on the main thread."""
        _, work.thread_note = macos.realtime_thread(10.0, 6.0, 10.0)
        next_state = 0.0
        next_tag = 0
        while not stop.is_set():
            if not in_sem.acquire(timeout=0.2):
                if time.monotonic() >= next_state:  # idle (no audio yet): still report
                    next_state = time.monotonic() + 0.5
                    send_state()
                continue
            while in_sem.acquire(False):
                pass
            # Every block still in time, oldest first. With headroom h, Sotto Voce takes the result for
            # block k-h when it hands over block k, so blocks older than newest-(h-1) are too
            # late: skip them rather than fall further behind.
            newest = int(latest[0])
            h = min(MAX_HEADROOM, max(1, int(latest[1])))
            for tag in range(max(next_tag, newest - (h - 1)), newest + 1):
                tg, mics_v, ref_v = ins[tag % SLOTS]
                if int(tg[0]) != tag:
                    continue
                mics, ref = mics_v.copy(), ref_v.copy()
                if int(tg[0]) != tag:                # overwritten while copying: too late anyway
                    continue
                try:
                    o, d = work.process(mics, ref)
                    otg, ov, dv = outs[tag % SLOTS]
                    ov[:] = o
                    dv[:] = d
                    out_sem.release()                # barrier: the data is visible before the tag
                    otg[0] = tag
                except Exception as e:
                    errors[0] += 1
                    if errors[0] == 1 or errors[0] % 100 == 0:
                        LOG.exception("Worker %d: processing error (%d so far)", wid + 1, errors[0])
                        up.put((wid, "faults", [(i, f"processing error: {e!r}") for i in range(len(names))]))
            next_tag = newest + 1
            f = work.take_faults()
            if f:
                up.put((wid, "faults", f))
            now = time.monotonic()
            if now >= next_state:
                next_state = now + 0.5
                send_state()

    th = threading.Thread(target=audio, name=f"worker{wid + 1}-processing", daemon=True)
    th.start()
    threading.current_thread().name = f"worker{wid + 1}-main"
    LOG.info("Processing worker %d ready: %d talkers (%s), process %d, %s", wid + 1, len(names), ", ".join(names),
             os.getpid(), note)
    up.put((wid, "ready", {"pid": os.getpid(), "thread": note}))
    # The main thread: requests from Sotto Voce, and creating and destroying plugins (VST3 wants
    # both on the main thread; several plugins abort otherwise).
    while not stop.is_set():
        try:
            if not ctrl.poll(0.2):
                work.close_retired()
                continue
            rid, op, args = ctrl.recv()
        except (EOFError, OSError):
            stop.set()
            break
        try:
            if op == "stop":
                stop.set()
                res = True
            elif op == "apply":
                res = work.apply(*args)
            elif op == "insert_info":
                res = work.insert_info(*args)
            elif op == "plugin_param":
                res = work.plugin_param(*args)
            elif op == "plugin_states":
                res = work.plugin_states()
            else:
                res = {"ok": False, "error": f"unknown request {op}"}
            up.put((wid, "reply", (rid, True, res)))
        except Exception as e:
            LOG.exception("Worker %d: request %s failed", wid + 1, op)
            up.put((wid, "reply", (rid, False, repr(e))))
        work.close_retired()
    th.join(timeout=2)
    if tracker:
        tracker.stop()
    work.close()
    shm.close()
    LOG.info("Processing worker %d stopped", wid + 1)


# ---------------------------------------------------------------------------- Sotto Voce's side
class _Worker:
    def __init__(self, pool, wid, rows):
        self.pool, self.wid, self.rows = pool, wid, rows
        self.g = len(rows)
        self.layout = _Layout(self.g)
        self.shm = shared_memory.SharedMemory(create=True, size=self.layout.size)
        self.latest, self.ins, self.outs = self.layout.views(self.shm.buf)
        self.latest[:] = -1
        for t, _, _ in self.ins + self.outs:
            t[:] = -1
        self.state = {}
        self.ready = False
        self.ever_ready = False
        self.failed = None
        self.pid = None
        self.restarts = 0
        self.restart_times = []
        self.given_up = False
        self.loading = set()                             # plugins being opened right now
        self.loaded_paths = set()                        # plugins this worker process has open
        self.proc = None
        self.start()

    def start(self):
        ctx = self.pool.ctx
        self.in_sem, self.out_sem = ctx.Semaphore(0), ctx.Semaphore(0)
        self.ctrl, child = ctx.Pipe()
        self.ctrl_lock = threading.Lock()
        self.ready, self.failed = False, None
        self.loaded_paths = set()                        # a new process has nothing open
        names = [self.pool.names[i] for i in self.rows]
        settings = [self.pool.settings[i] for i in self.rows]
        self.proc = ctx.Process(
            # Not a daemon: the worker's delay tracker starts its own helper process, which Python
            # forbids daemons. A worker still exits when Sotto Voce does: its control pipe closes.
            target=_worker_main, name=f"sotto-worker-{self.wid + 1}", daemon=False,
            args=(self.wid, names, settings, self.pool.model, self.shm.name, self.g, self.in_sem, self.out_sem,
                  child, self.pool.up, self.pool.track_delay))
        self.proc.start()

    def request(self, op, *args, timeout=30.0):
        return self.pool._request(self, op, args, timeout)


class Pool:
    """The same interface as processing.Processor, with the work in worker processes."""

    def __init__(self, names, settings, model=None, workers=None, track_delay=True, ready_timeout=60.0, headroom=1):
        self.names = list(names)
        self.n = len(self.names)
        self.settings = list(settings)
        self.model = model
        self.dfn = model                                 # truthy when the speech model is in use
        self.track_delay = track_delay
        self.ctx = mp.get_context("spawn")
        self.up = self.ctx.Queue()
        k = workers if workers else auto_workers(self.n)
        self.workers = [_Worker(self, w, rows) for w, rows in enumerate(split(self.n, k))]
        self.of = {}                                     # talker -> (worker, row in its group)
        for w in self.workers:
            for j, i in enumerate(w.rows):
                self.of[i] = (w, j)
        self.history = [np.zeros((self.n, HOP), np.float32) for _ in range(MAX_HEADROOM + 2)]
        self.headroom = 1
        self.grace_until = 0                             # no misses counted while headroom changes
        self.out_buf = np.zeros((self.n, HOP), np.float32)
        self.dry_buf = np.zeros((self.n, HOP), np.float32)
        self._missed_rows = np.zeros(self.n, bool)
        self._was_missed = np.zeros(self.n, bool)
        self._last_out = np.zeros(self.n, np.float32)
        self._fade = np.linspace(1.0, 0.0, RAMP, endpoint=False).astype(np.float32)
        self.blocks = 0
        self.misses = 0
        self.last_missed = 0
        self.miss_run = 0
        self.max_miss_run = 0
        self.good_run = 0
        self.faults = []
        self.ok = True
        self.why = ""
        self.live = False                                # counting misses only once every worker is up
        self.set_headroom(headroom)
        self._rid = itertools.count(1)
        self.crashed = {}                                # plugin path -> why it isn't loaded (it crashed a worker)
        self.insert_notes = {}                           # talker -> that reason, for the console
        self._waiting = {}
        self._wait_lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.listener = threading.Thread(target=self._listen, name="workers-listener", daemon=True)
        self.listener.start()
        self.watchdog = threading.Thread(target=self._watch, name="workers-watchdog", daemon=True)
        self.watchdog.start()
        self._stopped = False
        atexit.register(self.stop)                       # workers aren't daemons: never leave them behind
        t0 = time.monotonic()
        while time.monotonic() - t0 < ready_timeout and not all(w.ready or w.failed for w in self.workers):
            time.sleep(0.05)
        bad = [w for w in self.workers if not w.ready]
        if bad:
            LOG.error("Processing workers not ready after %.0f s: %s", time.monotonic() - t0,
                      ", ".join(f"{w.wid + 1} ({w.failed or 'no answer'})" for w in bad))
            self.ok, self.why = False, "processing workers didn't start"
        LOG.info("Processing in %d worker processes: %s", len(self.workers),
                 "; ".join(f"worker {w.wid + 1}: talkers {[i + 1 for i in w.rows]} (pid {w.pid})" for w in self.workers))

    # ------------------------------------------------------------ audio callback side
    def exchange(self, mics, ref):
        """Never waits. Hands over block k and returns (outputs, dry) for block k-headroom."""
        k = self.blocks
        self.blocks += 1
        h = self.headroom
        nh = len(self.history)
        self.history[k % nh][:] = mics
        missed = False
        n_missed = 0
        missed_rows = self._missed_rows
        missed_rows[:] = False
        for w in self.workers:
            tg, mv, rv = w.ins[k % SLOTS]
            mv[:] = mics[w.rows[0]:w.rows[-1] + 1]
            rv[:] = ref
            tg[0] = k
            w.latest[0] = k
            w.in_sem.release()                           # wakes the worker (and is a barrier)
            j = k - h                                    # the result Sotto Voce plays now
            otg, ov, dv = w.outs[j % SLOTS]
            got = False
            if j >= 0 and int(otg[0]) == j:
                w.out_sem.acquire(False)                 # barrier before reading the data
                self.out_buf[w.rows[0]:w.rows[-1] + 1] = ov
                self.dry_buf[w.rows[0]:w.rows[-1] + 1] = dv
                got = int(otg[0]) == j
            if not got:
                missed = True
                n_missed += 1
                missed_rows[w.rows[0]:w.rows[-1] + 1] = True
                # dry at the same delay as the processed path: the chain's own 10 ms + headroom
                rows = slice(w.rows[0], w.rows[-1] + 1)
                if k - h - 1 >= 0:
                    old = self.history[(k - h - 1) % nh]
                    self.out_buf[rows] = old[rows]
                    self.dry_buf[rows] = old[rows]
                else:
                    self.out_buf[rows] = 0
                    self.dry_buf[rows] = 0
        self.last_missed = n_missed                      # for the diagnostics' timing record
        self._declick(missed_rows)
        if not missed:
            self.miss_run = 0
            self.good_run += 1
            if not self.ok and self.good_run >= 100 and all(w.ready for w in self.workers):
                self.ok, self.why = True, ""
        else:
            self.good_run = 0
            if self.live and self.blocks > 50 and k >= self.grace_until:
                self.misses += 1
                self.miss_run += 1
                self.max_miss_run = max(self.max_miss_run, self.miss_run)
                if self.miss_run >= 3 and self.ok:
                    self.ok, self.why = False, "processing can't keep up (3 blocks late in a row)"
        if not self.live and all(w.ready for w in self.workers):
            self.live = True
        return self.out_buf, self.dry_buf

    def _declick(self, missed_rows):
        """A talker switching between processed and unprocessed audio (a late result, or the next
        one on time again) would step from one signal to the other: a click. Instead the step is
        faded out over 2.5 ms, so the switch is continuous."""
        out = self.out_buf
        changed = missed_rows != self._was_missed
        if changed.any():
            n = min(RAMP, out.shape[1])
            for r in np.nonzero(changed)[0]:
                out[r, :n] += (self._last_out[r] - out[r, 0]) * self._fade[:n]
        self._was_missed[:] = missed_rows
        self._last_out[:] = out[:, -1]

    def set_headroom(self, blocks):
        """How many blocks the workers get for each block (1 = 10 ms: 20 ms through Sotto Voce; 2 =
        20 ms: 30 ms through Sotto Voce). Changing it moves every talker's delay, so the app only
        allows it while disarmed."""
        h = min(MAX_HEADROOM, max(1, int(blocks)))
        for w in self.workers:
            w.latest[1] = h
        self.grace_until = self.blocks + 2 * MAX_HEADROOM
        if h != self.headroom:
            LOG.info("Processing headroom: %d block(s), %d ms through Sotto Voce", h, 10 * (h + 1))
        self.headroom = h

    def health(self):
        return self.ok, self.why

    def clear_health(self):
        if not self.ok and self.miss_run == 0 and all(w.ready for w in self.workers):
            self.ok, self.why = True, ""

    def drain_faults(self):
        f, self.faults = self.faults, []
        return f

    # ------------------------------------------------------------ messages from the workers
    def _listen(self):
        while not self.stop_ev.is_set():
            try:
                wid, kind, payload = self.up.get(timeout=0.5)
            except Exception:
                continue
            sp = getattr(self, "span", None)            # the app's timing record, to explain engine stalls
            try:
                ctx = sp(f"processing: {kind} message") if sp is not None else contextlib.nullcontext()
            except Exception:
                ctx = contextlib.nullcontext()
            with ctx:
                self._handle(wid, kind, payload)

    def _handle(self, wid, kind, payload):
        try:
            w = self.workers[wid] if wid < len(getattr(self, "workers", [])) else None
            if kind == "log":
                LOG.handle(payload)
            elif kind == "ready" and w:
                w.ready, w.pid, w.ever_ready = True, payload.get("pid"), True
            elif kind == "failed" and w:
                w.failed = payload
            elif kind == "state" and w:
                w.state = payload
            elif kind == "faults" and w:
                self.faults.extend((w.rows[i], why) for i, why in payload)
            elif kind == "reply":
                rid, ok, res = payload
                with self._wait_lock:
                    slot = self._waiting.get(rid)
                if slot:
                    slot["res"], slot["ok"] = res, ok
                    slot["ev"].set()
        except Exception:
            LOG.exception("Could not handle a message from processing worker %s", wid)

    def _watch(self):
        """Notices a worker that died, faults its talkers and starts it again. A worker that keeps
        failing (3 times in a minute, or every time it starts) is left stopped: its talkers stay on
        dry and processing stays unhealthy, so Sotto Voce can't be armed until it's fixed."""
        while not self.stop_ev.wait(0.5):
            for w in getattr(self, "workers", []):
                if w.given_up or w.proc is None or w.proc.is_alive() or self.stop_ev.is_set():
                    continue
                code = w.proc.exitcode
                why = w.failed or f"exit code {code}"
                w.ready = False
                # Record the plugin that did it first, then tell whoever is waiting: their next
                # request must already leave it out.
                for path in getattr(w, "loading", set()) or set():
                    name = os.path.basename(path).rsplit(".", 1)[0]
                    self.crashed[path] = (f"{name} crashed Sotto Voce's processing when it was opened ({why}), so it's "
                                          "no longer loaded. Use another plugin.")
                    LOG.error("Processing worker %d stopped while opening the plugin %s: not loading it again this session",
                              w.wid + 1, path)
                w.loading = set()
                self._fail_requests(w, f"stopped ({why})")
                self.faults.extend((i, f"processing worker {w.wid + 1} stopped ({why})") for i in w.rows)
                now = time.monotonic()
                w.restart_times = [t for t in w.restart_times if now - t < 60] + [now]
                if len(w.restart_times) > 3 or (w.failed and not w.ever_ready):
                    w.given_up = True
                    self.ok, self.why = False, f"processing worker {w.wid + 1} can't run: {why}"
                    LOG.critical("Processing worker %d (talkers %s) keeps stopping (%s). Not restarting it: those "
                                 "talkers stay on dry and Sotto Voce can't be armed. The traceback above says why.",
                                 w.wid + 1, [i + 1 for i in w.rows], why)
                    continue
                LOG.error("Processing worker %d (talkers %s) stopped unexpectedly (%s). Restarting it.",
                          w.wid + 1, [i + 1 for i in w.rows], why)
                self.ok, self.why = False, f"processing worker {w.wid + 1} stopped"
                w.restarts += 1
                w.state = {}
                w.start()
                if self.last_apply is not None:          # give the new worker the current settings
                    threading.Thread(target=self._reapply, args=(w,), daemon=True).start()

    last_apply = None

    def _reapply(self, w):
        t0 = time.monotonic()
        while not w.ready and time.monotonic() - t0 < 60:
            time.sleep(0.1)
        settings, labels = self.last_apply
        try:
            self._apply_one(w, settings, labels)
        except Exception:
            LOG.exception("Could not give restarted worker %d its settings", w.wid + 1)

    def _without_crashed(self, settings):
        """The settings with any insert that has crashed a worker taken out (and why, per talker)."""
        import copy
        out, notes = [], {}
        for i, st in enumerate(settings):
            path = (st.insert or {}).get("path")
            if path and path in self.crashed:
                st = copy.copy(st)
                st.insert = {}
                notes[i] = self.crashed[path]
            out.append(st)
        return out, notes

    def _apply_one(self, w, settings, labels):
        mine, notes = self._without_crashed([settings[i] for i in w.rows])
        for j, why in notes.items():
            self.insert_notes[w.rows[j]] = why
        for i in w.rows:
            if i not in [w.rows[j] for j in notes]:
                self.insert_notes.pop(i, None)
        paths = {(s.insert or {}).get("path") for s in mine if (s.insert or {}).get("path")}
        w.loading = paths - w.loaded_paths               # only these are opened by this request
        try:
            r = w.request("apply", mine, [labels[i] for i in w.rows])
            w.loaded_paths = paths
        finally:
            w.loading = set()
        return r, list(notes.values())

    def _fail_requests(self, w, why):
        """A worker that died can't answer: tell everyone waiting on it now, not after 30 s."""
        with self._wait_lock:
            slots = [sl for sl in self._waiting.values() if sl.get("wid") == w.wid]
        for sl in slots:
            sl["ok"], sl["res"] = False, why
            sl["ev"].set()

    def _request(self, w, op, args, timeout):
        # A worker that has just died (or is restarting) can't take a request: wait for the
        # restarted one, up to 15 s, rather than writing into a closed pipe.
        t0 = time.monotonic()
        while not (w.ready and w.proc is not None and w.proc.is_alive()):
            if w.given_up or self.stop_ev.is_set() or time.monotonic() - t0 > 15:
                raise RuntimeError(f"processing worker {w.wid + 1} isn't running"
                                   + (f": {self.why}" if self.why else ""))
            time.sleep(0.05)
        rid = next(self._rid)
        slot = {"ev": threading.Event(), "wid": w.wid}
        with self._wait_lock:
            self._waiting[rid] = slot
        try:
            try:
                with w.ctrl_lock:
                    w.ctrl.send((rid, op, args))
            except (BrokenPipeError, EOFError, OSError):
                raise RuntimeError(f"processing worker {w.wid + 1} stopped; it's being restarted, try again in a moment")
            if not slot["ev"].wait(timeout):
                raise TimeoutError(f"processing worker {w.wid + 1} didn't answer {op} in {timeout:.0f} s")
            if not slot["ok"]:
                raise RuntimeError(f"processing worker {w.wid + 1}: {slot['res']}")
            return slot["res"]
        finally:
            with self._wait_lock:
                self._waiting.pop(rid, None)

    # ------------------------------------------------------------ settings (any thread)
    def apply(self, settings, labels):
        self.last_apply = (list(settings), list(labels))
        errors, events = [], []
        for w in self.workers:
            try:
                r, notes = self._apply_one(w, settings, labels)
                errors += notes + r.get("errors", [])
                events += r.get("events", [])
            except Exception as e:
                LOG.exception("Settings could not be applied in worker %d", w.wid + 1)
                errors.append(str(e))
        return {"errors": errors, "events": events}

    def insert_info(self, k):
        w, j = self.of[k]
        try:
            return w.request("insert_info", j, timeout=10)
        except Exception as e:
            return {"loaded": False, "error": str(e)}

    def plugin_param(self, k, pid, norm):
        w, j = self.of[k]
        try:
            return w.request("plugin_param", j, pid, norm, timeout=5)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def plugin_states(self):
        out = {}
        for w in self.workers:
            for j, v in w.request("plugin_states", timeout=10).items():
                out[w.rows[j]] = v
        return out

    # ------------------------------------------------------------ the rest
    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self.stop_ev.set()
        for w in self.workers:
            try:
                with w.ctrl_lock:
                    w.ctrl.send((0, "stop", ()))
            except Exception:
                pass
        for w in self.workers:
            try:
                if w.proc is not None and w.proc.pid is not None:
                    w.proc.join(timeout=3)
                    if w.proc.is_alive():
                        LOG.warning("Processing worker %d didn't stop: ending it", w.wid + 1)
                        w.proc.terminate()
                        w.proc.join(timeout=2)
            except Exception:
                LOG.exception("Could not stop processing worker %d", w.wid + 1)
            try:
                w.shm.close()
                w.shm.unlink()
            except Exception:
                pass

    def state(self):
        ws = [w.state for w in self.workers]
        talkers = []
        for w in self.workers:
            st = w.state.get("talkers") or [{"name": self.names[i], "mode": "starting"} for i in w.rows]
            talkers += [dict(t) for t in st]
        for i, why in self.insert_notes.items():
            if i < len(talkers):
                talkers[i]["insert_error"] = why
        num = lambda k: max((s.get(k) or 0) for s in ws) if ws else 0   # noqa: E731
        stage = {k: max(((s.get("stage_ms") or {}).get(k) or 0) for s in ws) for k in ("front", "model", "back")}
        return {
            "on": True, "model": bool(self.model), "workers": len(self.workers), "misses": self.misses, "ok": self.ok,
            "why": self.why, "latency_ms": round(1000 * ((self.headroom + 1) * HOP) / 48000, 1),
            "headroom": self.headroom,
            "proc_ms": num("proc_ms"), "proc_ms_max": num("proc_ms_max"), "cpu_ms": num("cpu_ms"), "stage_ms": stage,
            "model_rows": sum((s.get("model_rows") or 0) for s in ws), "pgm_db": num("pgm_db") if ws else -120,
            "thread": ws[0].get("thread", "") if ws else "",
            "worker_detail": [{"worker": w.wid + 1, "talkers": [i + 1 for i in w.rows], "pid": w.pid, "ready": w.ready,
                               "restarts": w.restarts, "proc_ms": w.state.get("proc_ms"), "cpu_ms": w.state.get("cpu_ms"),
                               "stage_ms": w.state.get("stage_ms"), "errors": w.state.get("errors")} for w in self.workers],
            "talkers": talkers,
        }
