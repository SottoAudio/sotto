"""Sotto Voce's engine in C (engine/audio-io/engine.c): stage 3 of the engine-in-C plan.

The 10 ms engine runs on a real-time thread in C and never takes Python's lock. Python is the
control room: it sends commands (arm, disarm, trip, faults, phones plan, headroom), reads back the
fail-safe state, events, timing, the recording and the phones, and renders test sources ahead.

The app keeps its Python FailsafeCore object as the model the console and the fault finder read:
CEngine.sync() copies the C core's state into it, and the ArmState / core methods the app calls are
wrapped so they act on the C core first (attach()). The C core decides; the model shows.
"""
import ctypes
import threading
import time

import numpy as np

from . import ccore
from .arming import Event
from .diagnostics import LOG

CMD_ARM, CMD_DISARM, CMD_TRIP, CMD_INJECT, CMD_CLEAR, CMD_SOLO, CMD_HEADROOM, CMD_WORKER, CMD_FREEZE = range(1, 10)
WHY_PROBE, WHY_STARTING, WHY_EXTERNAL, WHY_REQUEST, WHY_LATE, WHY_PROC = 1, 2, 3, 4, 5, 6
TIMING_COLS = 10
STATS = ("running", "blocks", "proc_blocks", "misses", "miss_run", "max_miss_run", "proc_ok", "last_missed",
         "late_total", "late_until", "rec_dropped", "test_short", "headroom", "live")
_ready = False


def library():
    global _ready
    lib = ccore.library()
    if lib is None or not hasattr(lib, "se_create"):
        return None
    if not _ready:
        P, I, D, I64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int64
        IP, DP, FP = ctypes.POINTER(I), ctypes.POINTER(D), ctypes.POINTER(ctypes.c_float)
        U8P = ctypes.POINTER(ctypes.c_uint8)
        lib.se_create.restype = P
        lib.se_create.argtypes = [P, P, I, I, I, I, I, IP, I, I, IP, IP, I, I]
        lib.se_destroy.argtypes = [P]
        lib.se_set_proc.argtypes = [P, I, I, I, I]
        lib.se_stage_worker.argtypes = [P, I, I, I, P, P, P]
        lib.se_stage_solo.argtypes = [P, I, IP, IP, I, IP, IP, D, I, D, D]
        lib.se_set_ext.argtypes = [P, I]
        lib.se_set_ready.argtypes = [P, I, I]
        lib.se_set_workgroup.argtypes = [P, P]
        lib.se_command_p.restype = I64
        lib.se_command_p.argtypes = [P, I, I, I, D, P, P, P]
        lib.se_clock.restype = D
        lib.se_clock.argtypes = []
        lib.sc_not_green.argtypes = [P]
        lib.se_set_timing.argtypes = [P, D]
        lib.se_command.restype = I64
        lib.se_command.argtypes = [P, I, I, I, D]
        lib.se_result.restype = I
        lib.se_result.argtypes = [P, I64, DP]
        lib.se_start.restype = I
        lib.se_start.argtypes = [P]
        lib.se_stop.argtypes = [P]
        lib.se_stats.argtypes = [P, DP]
        lib.se_exchange_test.argtypes = [P, FP, DP, DP]
        lib.se_thread_note.restype = ctypes.c_char_p
        lib.se_thread_note.argtypes = [P]
        lib.se_workgroup_note.restype = ctypes.c_char_p
        lib.se_workgroup_note.argtypes = [P]
        lib.se_timing.restype = I
        lib.se_timing.argtypes = [P, DP, I]
        lib.se_record.restype = I
        lib.se_record.argtypes = [P, FP, I]
        lib.se_phones.restype = I
        lib.se_phones.argtypes = [P, FP, I]
        lib.se_test_queued.restype = I
        lib.se_test_queued.argtypes = [P]
        lib.se_test_push.restype = I
        lib.se_test_push.argtypes = [P, FP, U8P]
        lib.sc_disarm_why.argtypes = [P, I, I]
        lib.sc_armed.restype = I
        lib.sc_armed.argtypes = [P]
        _ready = True
    return lib


def _ints(xs):
    xs = list(xs)
    return (ctypes.c_int * max(1, len(xs)))(*xs)


class CEngine:
    TEST_AHEAD = 8                   # test-source blocks rendered ahead (80 ms)

    def __init__(self, native, cfg, n_in, n_out, block):
        lib = library()
        if lib is None:
            raise RuntimeError("this build has no C engine")
        self.lib, self.cfg, self.n_in, self.n_out, self.block = lib, cfg, n_in, n_out, block
        self.native = native
        self.cc = ccore.CCore(cfg, n_in, n_out)
        self.rec_cols = [(0, c) for c in cfg.talker_in + cfg.probe_in] + \
                        ([(0, cfg.listen_in)] if cfg.listen_in is not None else []) + \
                        [(1, cfg.heartbeat_out), (1, cfg.talker_out[0])]
        mon = list(cfg.monitor_out)
        self.h = lib.se_create(native.h, self.cc.h, n_in, n_out, block, cfg.fs, len(cfg.talker_in), _ints(cfg.talker_in),
                               -1 if cfg.pgm_in is None else cfg.pgm_in, len(self.rec_cols),
                               _ints([s for s, _ in self.rec_cols]), _ints([c for _, c in self.rec_cols]),
                               mon[0] if mon else -1, mon[1] if len(mon) > 1 else -1)
        if not self.h:
            raise RuntimeError("the C engine refused this channel layout")
        self.lock = threading.RLock()           # one producer at a time: staging + command + wait together
        self._stats = (ctypes.c_double * len(STATS))()
        self._tim = np.zeros((1024, TIMING_COLS), np.float64)
        self._rec = np.zeros((64, block, len(self.rec_cols)), np.float32)
        self._ph = np.zeros((64, block, 2), np.float32)
        self._test_zero = np.zeros((block, n_in), np.float32)
        self._keep = []                         # semaphores the C side may still hold
        self.pool = None
        self.started = False
        self.ext_why = ""
        self._solo_seq = None
        self.test_short = 0

    # ------------------------------------------------------------ commands
    def cmd(self, op, a=0, b=0, x=0.0, wait=0.25, p=(None, None, None)):
        """Sends a command and waits (up to `wait` s) for the engine to apply it. Returns its result,
        or None when the engine didn't answer in time (it still applies it when it can, except an
        arm request: the engine refuses one older than 0.2 s)."""
        with self.lock:
            if not self.h:
                return None
            if op == CMD_ARM:
                x = self.lib.se_clock()          # when it was asked, on the engine's own clock
            n = self.lib.se_command_p(self.h, op, int(a), int(b), float(x), *p)
            if n < 0:
                LOG.warning("C engine command ring full: command %d dropped", op)
                return None
            res = ctypes.c_double(0)
            end = time.monotonic() + wait
            while not self.lib.se_result(self.h, n, ctypes.byref(res)):
                if time.monotonic() > end:
                    return None
                time.sleep(0.0005)
            return res.value

    # ------------------------------------------------------------ processing
    def attach_pool(self, pool):
        """The processing workers' shared memory and semaphores, as workers.Pool.exchange uses them."""
        self.pool = pool
        pool.on_worker_start = self._worker_started      # first: a restart from now on is handed over
        for w in pool.workers:
            self._stage_worker(w)
        self.lib.se_set_proc(self.h, 1, len(pool.workers), pool.headroom, int(bool(pool.ok)))
        for w in pool.workers:
            self.lib.se_set_ready(self.h, w.wid, int(bool(w.ready)))
        set_headroom = pool.set_headroom

        def headroom(blocks):
            set_headroom(blocks)
            self.cmd(CMD_HEADROOM, pool.headroom)
        pool.set_headroom = headroom

    def _stage_worker(self, w):
        base = w.latest.ctypes.data                 # the start of the worker's shared memory
        self._keep += [w.in_sem, w.out_sem]
        self.lib.se_stage_worker(self.h, w.wid, w.rows[0], w.g, base, w.in_sem._semlock.handle,
                                 w.out_sem._semlock.handle)

    def _worker_started(self, w):
        """A worker restarted with new semaphores: hand them to the engine, by value in the command."""
        if not self.h:
            return
        with self.lock:
            self._keep += [w.in_sem, w.out_sem]
            if not self.started:
                self._stage_worker(w)
                return
        self.cmd(CMD_WORKER, w.wid, w.rows[0] | (w.g << 16), wait=1.0,
                 p=(w.latest.ctypes.data, w.in_sem._semlock.handle, w.out_sem._semlock.handle))
        LOG.info("C engine: processing worker %d's new semaphores handed over", w.wid + 1)

    # ------------------------------------------------------------ running
    def start(self):
        if self.lib.se_start(self.h) != 0:
            raise RuntimeError("couldn't start the C engine thread")
        self.started = True

    def stop(self):
        if self.started:
            self.lib.se_stop(self.h)
            self.started = False

    def destroy(self):
        self.stop()
        with self.lock:
            if self.h:
                self.lib.se_destroy(self.h)
                self.h = None
            self.cc.destroy()

    def set_workgroup_device(self, dev):
        """The soundcard's I/O workgroup, read here from Core Audio (off the real-time thread) and
        handed to the engine thread to join."""
        from . import coreaudio
        wg = coreaudio.io_workgroup(dev) if dev else None
        if wg:
            self.lib.se_set_workgroup(self.h, wg)

    def thread_note(self):
        return (f"{(self.lib.se_thread_note(self.h) or b'').decode()}, "
                f"{(self.lib.se_workgroup_note(self.h) or b'').decode()}")

    def stats(self):
        if not self.h:
            return {k: 0 for k in STATS}
        self.lib.se_stats(self.h, self._stats)
        return dict(zip(STATS, list(self._stats)))

    def set_ext(self, ok, why=""):
        self.ext_why = why
        self.lib.se_set_ext(self.h, int(bool(ok)))

    # ------------------------------------------------------------ the model the app reads
    def attach(self, core):
        """The app keeps calling its FailsafeCore and ArmState: they act on the C core first."""
        self.core = core
        arm = core.arm
        ra, rd, tt, uh = arm.request_arm, arm.request_disarm, arm.trip_talker, arm.update_health
        inj, clr, solo = core.inject, core.clear_faults, core.set_solo
        me = self

        def request_arm(t, source, talker=None):
            res = me.cmd(CMD_ARM, -1 if talker is None else talker)
            me.sync()
            if res == 1:
                arm.reason = ""
                arm.events.append(Event(core.t, "armed" if talker is None else "talker_armed",
                                        "everyone" if talker is None else f"talker {talker + 1}", source))
                return True, "armed" if talker is None else f"talker {talker + 1} armed"
            if res is None or res == -2:
                msg = "Sotto Voce's engine didn't answer in time; ask again"
            elif res == -1:
                return False, "no such talker"
            elif res == -3:
                msg = f"checks not green: {core.health_why or me.ext_why or 'a check outside the core is red'}"
            elif res == -4:
                msg = "no audio is arriving from the soundcard"
            elif talker is not None and arm.can_arm(core.t) and not arm.armed:
                msg = "arm everyone first: the Riedel is holding all talkers on dry"
            elif arm.green_since is None:
                msg = "checks not green"
            else:
                g = arm.green_for(core.t)
                msg = f"checks green for {g:.1f} s, need {arm.arm_hold_s:.0f} s ({arm.arm_hold_s - g:.1f} s to go)"
            arm.events.append(Event(core.t, "arm_refused", msg if talker is None else f"talker {talker + 1}: {msg}", source))
            return False, msg

        def request_disarm(t, source, why="disarmed by engineer"):
            me.cmd(CMD_DISARM, WHY_REQUEST)
            rd(t, source, why)

        def trip_talker(t, talker, why, source="auto"):
            me.cmd(CMD_TRIP, talker)
            tt(t, talker, why, source)

        def update_health(t, healthy, why=""):
            if not healthy:
                me.set_ext(False, why)
                if arm.armed:
                    me.cmd(CMD_DISARM, WHY_EXTERNAL)
            uh(t, healthy, why)

        def inject(name, seconds=None, ms=None):
            inj(name, seconds=seconds, ms=ms)
            if name == "freeze":                 # the engine thread itself misses its deadlines
                core.pending_freeze_ms = 0
                me.cmd(CMD_FREEZE, int(ms or 300), wait=0.0)
            elif name in ccore.FAULTS:
                me.cmd(CMD_INJECT, ccore.FAULTS[name], 0, -1.0 if seconds is None else float(seconds))

        def clear_faults():
            clr()
            me.cmd(CMD_CLEAR)

        def set_solo(*a, **k):
            plan = solo(*a, **k)
            left, right = core._solo_plan
            s = core.solo
            with me.lock:
                if not me.h:                      # stopped: the model only
                    return plan
                if not me.started:                # not running yet: straight into the core, no waiting
                    me.cc.set_solo(core._solo_plan, s)
                    return plan
                if me._solo_seq is not None:      # the last plan must be taken before staging the next
                    res = ctypes.c_double(0)
                    end = time.monotonic() + 1.0
                    while not me.lib.se_result(me.h, me._solo_seq, ctypes.byref(res)):
                        if time.monotonic() > end:
                            LOG.warning("C engine: the last phones plan hasn't been taken; this one is skipped")
                            return plan
                        time.sleep(0.001)
                    me._solo_seq = None
                me.lib.se_stage_solo(me.h, len(left), _ints([0 if q == "i" else 1 for q, _ in left]),
                                     _ints([n for _, n in left]), len(right),
                                     _ints([0 if q == "i" else 1 for q, _ in right]), _ints([n for _, n in right]),
                                     float(s.get("level_db", 0.0)), int(bool(s.get("dim"))),
                                     float(s.get("dim_db", -20.0)), float(s.get("limit_db", -3.0)))
                n = me.lib.se_command(me.h, CMD_SOLO, 0, 0, 0.0)
                res = ctypes.c_double(0)
                end = time.monotonic() + 0.25
                while n >= 0 and not me.lib.se_result(me.h, n, ctypes.byref(res)):
                    if time.monotonic() > end:
                        me._solo_seq = n                 # still queued: don't overwrite its staging
                        break
                    time.sleep(0.0005)
            return plan
        arm.request_arm, arm.request_disarm, arm.trip_talker, arm.update_health = \
            request_arm, request_disarm, trip_talker, update_health
        core.inject, core.clear_faults, core.set_solo = inject, clear_faults, set_solo
        core.set_solo(core.solo["ids"], core.solo["mode"])        # the current plan, once

    def why_text(self, why, arg):
        if why == WHY_PROBE:
            ms = 1000 * min(self.cc.lib.sc_probe_missing(self.cc.h, arg), 99)
            return f"PROBE {arg + 1} missing for {ms:.0f} ms"
        if why == WHY_STARTING:
            return "starting up"
        if why == WHY_EXTERNAL:
            if self.ext_why:
                return self.ext_why
            st = self.stats()
            if st["late_until"] > self.cc.t:
                return "Sotto Voce's engine was late: not green for 5 s, then re-arm by hand"
            if self.pool is not None and not st["proc_ok"]:
                return "processing can't keep up (3 blocks late in a row)"
            return "a check outside the core"
        if why == WHY_LATE:
            return "Sotto Voce's engine was late: soundcard cycles played dry and HEARTBEAT held off until re-armed"
        if why == WHY_PROC:
            return "processing can't keep up (3 blocks late in a row)"
        return ""

    def sync(self):
        """The C core's state into the Python model (main loop, ~20 times a second, and after commands)."""
        core, cc = self.core, self.cc
        if not self.h or not cc.h:                 # stopped: the model keeps its last state
            return None
        st = cc.state()
        t = cc.t
        core.samples = int(round(t * self.cfg.fs))
        arm = core.arm
        arm.armed = st["armed"]
        arm.talker_armed = list(st["talker_armed"])
        arm.green_since = (t - st["green_for_s"]) if st["green"] else None
        core.health_why = "" if st["green"] else self.why_text(st["why"], st["why_arg"])
        for i, w in enumerate(core.watch):
            m = cc.lib.sc_probe_missing(cc.h, i)
            w.last_seen = None if m == float("inf") else t - m
        lv_in = np.zeros(self.n_in)
        lv_out = np.zeros(self.n_out)
        tdb = np.zeros(max(1, cc.nt))
        cc.lib.sc_levels(cc.h, lv_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         lv_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         tdb.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        core.lv_in, core.lv_out = lv_in, lv_out
        core.levels["talkers"] = [float(v) for v in tdb[:cc.nt]]
        if self.cfg.listen_in is not None:
            core.levels["listen"] = float(lv_in[self.cfg.listen_in])
        return st

    def events(self):
        """The C core's own events (automatic disarms, faults ending) as the app's Events."""
        out = []
        if not self.h:
            return out
        for t, kind, talker, why, arg in self.cc.events():
            if kind == "disarmed" and why != WHY_REQUEST:
                text = self.why_text(why, arg)
                self.core.arm.reason = text
                if why == WHY_LATE:
                    continue                     # counted and logged as engine_late by the app
                out.append(Event(t, "disarmed", text, "auto"))
            elif kind == "fault_ended":
                name = {1: "silence_all", 2: "mute_returns"}.get(why, "?")
                self.core.faults.pop(name, None)
                out.append(Event(t, "fault_ended", name, "auto"))
        return out

    def timing(self):
        n = self.lib.se_timing(self.h, self._tim.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(self._tim))
        return self._tim[:n]

    def recording(self):
        """Every block waiting (the ring holds 5 s)."""
        out = []
        while True:
            n = self.lib.se_record(self.h, self._rec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(self._rec))
            out += [b.copy() for b in self._rec[:n]]
            if n < len(self._rec):
                return out

    def phones(self):
        n = self.lib.se_phones(self.h, self._ph.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(self._ph))
        return self._ph[:n]

    def feed_tests(self, tests):
        """Test sources rendered ahead for the engine to mix in (never while armed: the engine checks)."""
        if not tests.plan:
            self._tests_fed = False
            return
        mask = np.zeros(self.n_in, np.uint8)
        if self.lib.se_test_queued(self.h) == 0 and getattr(self, "_tests_fed", False):
            self.test_short += 1                 # the engine ran out: real input played for a moment
        self._tests_fed = True
        while self.lib.se_test_queued(self.h) < self.TEST_AHEAD:
            plan = tests.plan
            if not plan:
                return
            blk = tests.apply(self._test_zero, self.block)
            if blk is self._test_zero:
                return
            mask[:] = 0
            for ch in plan:
                if ch < self.n_in:
                    mask[ch] = 1
            blk = np.ascontiguousarray(blk, np.float32)
            if not self.lib.se_test_push(self.h, blk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))):
                return

    def sync_pool(self, st):
        """The engine's view of processing into the Pool (its state() and the fault finder read it)."""
        p = self.pool
        if p is None:
            return
        for w in p.workers:
            self.lib.se_set_ready(self.h, w.wid, int(bool(w.ready)))
        p.misses = int(st["misses"])
        p.blocks = int(st["proc_blocks"])
        p.last_missed = int(st["last_missed"])
        p.max_miss_run = int(st["max_miss_run"])
        p.miss_run = int(st["miss_run"])
        ok = bool(st["proc_ok"])
        if ok != p.ok:
            p.ok, p.why = ok, ("" if ok else "processing can't keep up (3 blocks late in a row)")
