"""The fail-safe core in C (engine/audio-io/core.c), through ctypes, and its shadow check.

Stage 2 of the engine-in-C plan: the C core runs beside the Python core (core.FailsafeCore) on
every block, from the same input, the same processing results and the same commands, and the
Shadow class compares the two: every output sample, and the fail-safe state (armed, HEARTBEAT,
each talker, green or not). Python's core still drives the outputs; nothing the C core does
reaches the soundcard yet. The diagnostics file reports how many blocks matched.
"""
import collections
import ctypes

import numpy as np

from . import nativeio

EV = {1: "armed", 2: "disarmed", 3: "arm_refused", 4: "talker_armed", 5: "talker_tripped", 6: "fault_ended"}
FAULTS = {"silence_all": 1, "mute_returns": 2}
_ready = False


def library():
    """The native library with the core's functions declared, or None."""
    global _ready
    lib = nativeio.library()
    if lib is None or not hasattr(lib, "sc_create"):
        return None
    if not _ready:
        P, I, D = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
        IP, DP, FP = ctypes.POINTER(I), ctypes.POINTER(D), ctypes.POINTER(ctypes.c_float)
        lib.sc_create.restype = P
        lib.sc_create.argtypes = [I, I, I, I, IP, IP, I, IP, I, I, I, I, I, IP, I, D, D, D, D, D, D, D, D, D]
        lib.sc_destroy.argtypes = [P]
        lib.sc_time.restype = D
        lib.sc_time.argtypes = [P]
        lib.sc_request_arm.restype = I
        lib.sc_request_arm.argtypes = [P, I]
        lib.sc_request_disarm.argtypes = [P]
        lib.sc_trip_talker.argtypes = [P, I]
        lib.sc_inject.argtypes = [P, I, D]
        lib.sc_clear_faults.argtypes = [P]
        lib.sc_set_solo.argtypes = [P, I, IP, IP, I, IP, IP, D, I, D, D]
        lib.sc_process.restype = I
        lib.sc_process.argtypes = [P, FP, FP, I, I, DP, DP, I]
        lib.sc_state.argtypes = [P, DP, IP]
        lib.sc_probe_missing.restype = D
        lib.sc_probe_missing.argtypes = [P, I]
        lib.sc_levels.argtypes = [P, DP, DP, DP]
        lib.sc_events.restype = I
        lib.sc_events.argtypes = [P, DP, I]
        _ready = True
    return lib


def _ints(xs):
    xs = list(xs)
    return (ctypes.c_int * max(1, len(xs)))(*xs)


class CCore:
    """The C core for one channel layout (core.Config)."""

    def __init__(self, cfg, n_in, n_out):
        lib = library()
        if lib is None:
            raise RuntimeError("this build has no C core")
        self.lib, self.cfg, self.n_in, self.n_out = lib, cfg, n_in, n_out
        nt = len(cfg.talker_in)
        mon = list(cfg.monitor_out)[:2]
        self.h = lib.sc_create(cfg.fs, n_in, n_out, nt, _ints(cfg.talker_in), _ints(cfg.talker_out),
                               len(cfg.probe_in), _ints(cfg.probe_in), cfg.probe_out, cfg.heartbeat_out,
                               -1 if cfg.listen_in is None else cfg.listen_in, -1 if cfg.pgm_in is None else cfg.pgm_in,
                               len(mon), _ints(mon), int(bool(cfg.marker_on)),
                               cfg.heartbeat_hz, cfg.heartbeat_dbfs, cfg.probe_hz, cfg.probe_dbfs,
                               cfg.marker_hz, cfg.marker_dbfs, cfg.probe_timeout_s, cfg.arm_hold_s, cfg.startup_grace_s)
        if not self.h:
            raise ValueError("the C core refused this channel layout")
        self.nt = nt
        self._v = (ctypes.c_double * 8)()
        self._ta = (ctypes.c_int * max(1, nt))()
        self._ev = np.zeros((64, 5), np.float64)
        self._ret = np.zeros((4096, max(1, nt)), np.float64)
        self._dry = np.zeros((4096, max(1, nt)), np.float64)

    @property
    def t(self):
        return self.lib.sc_time(self.h)

    def request_arm(self, talker=None):
        return self.lib.sc_request_arm(self.h, -1 if talker is None else int(talker)) == 1

    def request_disarm(self):
        self.lib.sc_request_disarm(self.h)

    def trip_talker(self, talker):
        self.lib.sc_trip_talker(self.h, int(talker))

    def inject(self, name, seconds=None):
        if name in FAULTS:
            self.lib.sc_inject(self.h, FAULTS[name], -1.0 if seconds is None else float(seconds))

    def clear_faults(self):
        self.lib.sc_clear_faults(self.h)

    def set_solo(self, plan, solo):
        left, right = plan
        lk, lc = [0 if k == "i" else 1 for k, _ in left], [n for _, n in left]
        rk, rc = [0 if k == "i" else 1 for k, _ in right], [n for _, n in right]
        self.lib.sc_set_solo(self.h, len(left), _ints(lk), _ints(lc), len(right), _ints(rk), _ints(rc),
                             float(solo.get("level_db", 0.0)), int(bool(solo.get("dim"))),
                             float(solo.get("dim_db", -20.0)), float(solo.get("limit_db", -3.0)))

    def process(self, indata, outdata, status=0, returns=None, dry=None, ext_ok=True):
        """indata (frames, n_in) and outdata (frames, n_out) float32 C-contiguous; returns / dry
        (frames, n_talkers) or None. Returns True when every check is green."""
        frames = indata.shape[0]
        fp = ctypes.POINTER(ctypes.c_float)
        rp = dp = None
        if returns is not None:
            r, d = self._ret[:frames], self._dry[:frames]
            r[:] = returns
            d[:] = dry
            rp = r.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            dp = d.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return self.lib.sc_process(self.h, indata.ctypes.data_as(fp), outdata.ctypes.data_as(fp), frames,
                                   int(bool(status)), rp, dp, int(bool(ext_ok))) == 1

    def state(self):
        self.lib.sc_state(self.h, self._v, self._ta)
        v = list(self._v)
        return {"armed": bool(v[0]), "heartbeat_on": bool(v[1]), "green": bool(v[2]), "green_for_s": v[3],
                "why": int(v[4]), "why_arg": int(v[5]), "xruns": int(v[6]), "blocks": int(v[7]),
                "talker_armed": [bool(self._ta[i]) for i in range(self.nt)]}

    def events(self):
        n = self.lib.sc_events(self.h, self._ev.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(self._ev))
        return [(r[0], EV.get(int(r[1]), "?"), int(r[2]), int(r[3]), int(r[4])) for r in self._ev[:n]]

    def destroy(self):
        if self.h:
            self.lib.sc_destroy(self.h)
            self.h = None


class Shadow:
    """Runs the C core beside the Python one and counts every difference."""
    TOL = 1e-5                     # float32 outputs; the two cores round the same sums at most this apart

    def __init__(self, core, n_in, n_out):
        self.core = core
        self.c = CCore(core.cfg, n_in, n_out)
        self.out = np.zeros((4096, n_out), np.float32)
        self.blocks = self.matched = 0
        self.out_bad = self.state_bad = 0
        self.worst = 0.0
        self.first = []            # the first differences, for the diagnostics file
        self.errors = 0
        self._sync_solo = None
        self.q = collections.deque()           # commands from any thread, applied on the audio thread

    # commands, mirrored from the Python core's. They can come from any thread (console, engine,
    # main loop), so they are queued and applied at the start of the next block, where the C core
    # is still at the same audio time the Python core was when it acted.
    def arm(self, talker, accepted):
        self.q.append(("arm", talker, accepted))

    def disarm(self):
        self.q.append(("disarm",))

    def trip(self, talker):
        self.q.append(("trip", talker))

    def inject(self, name, seconds=None):
        self.q.append(("inject", name, seconds))

    def clear_faults(self):
        self.q.append(("clear",))

    def _apply(self, cmd):
        k = cmd[0]
        if k == "arm":
            self._arm(cmd[1], cmd[2])
        elif k == "disarm":
            self.c.request_disarm()
        elif k == "trip":
            self.c.trip_talker(cmd[1])
        elif k == "inject":
            self.c.inject(cmd[1], cmd[2])
        elif k == "clear":
            self.c.clear_faults()

    def _arm(self, talker, accepted):
        ok = self.c.request_arm(talker)
        if ok != bool(accepted):
            self._note(f"arm {'everyone' if talker is None else f'talker {talker + 1}'}: Python "
                       f"{'accepted' if accepted else 'refused'}, C {'accepted' if ok else 'refused'}")
            self.state_bad += 1

    def _note(self, s):
        if len(self.first) < 20:
            self.first.append(f"block {self.blocks}: {s}")

    def block(self, indata, py_out, status, returns, dry, ext_ok):
        """After the Python core has written py_out for this block."""
        try:
            while self.q:
                self._apply(self.q.popleft())
            core = self.core
            sig = (core._solo_plan, core.solo.get("level_db"), core.solo.get("dim"), core.solo.get("dim_db"),
                   core.solo.get("limit_db"))
            if sig != self._sync_solo:
                self._sync_solo = sig
                self.c.set_solo(core._solo_plan, core.solo)
            frames = indata.shape[0]
            out = self.out[:frames]
            self.c.process(np.ascontiguousarray(indata, np.float32), out, status,
                           None if returns is None else returns, None if dry is None else dry, ext_ok)
            self.blocks += 1
            d = float(np.max(np.abs(out - py_out))) if out.size else 0.0
            self.worst = max(self.worst, d)
            ok = True
            if d > self.TOL:
                ok = False
                self.out_bad += 1
                ch = int(np.argmax(np.max(np.abs(out - py_out), axis=0)))
                self._note(f"output {ch + 1} differs by {d:.2e}")
            cs = self.c.state()
            a = core.arm
            py = (a.armed, a.heartbeat_on and "silence_all" not in core.faults, a.green_since is not None,
                  list(a.talker_armed))
            cc = (cs["armed"], cs["heartbeat_on"], cs["green"], cs["talker_armed"])
            if py != cc:
                ok = False
                self.state_bad += 1
                self._note(f"state differs: Python armed={py[0]} heartbeat={py[1]} green={py[2]} talkers={py[3]}; "
                           f"C armed={cc[0]} heartbeat={cc[1]} green={cc[2]} talkers={cc[3]}")
            self.matched += ok
            self.c.events()                      # stage 2 doesn't use them; keep the ring moving
        except Exception as e:                   # never let the shadow disturb the real audio
            self.errors += 1
            if self.errors == 1:
                self._note(f"error: {e!r}")

    def report(self):
        if not self.blocks:
            return "The C core hasn't run yet."
        pct = 100.0 * self.matched / self.blocks
        lines = [f"C core (shadow): {self.blocks} blocks compared with the Python core; {self.matched} matched "
                 f"({pct:.3f} %). Output differences: {self.out_bad} blocks (largest {self.worst:.2e}); "
                 f"fail-safe state differences: {self.state_bad}; errors: {self.errors}."]
        if self.first:
            lines.append("First differences:\n" + "\n".join("  " + s for s in self.first))
        return "\n".join(lines)

    def state(self):
        return {"blocks": self.blocks, "matched": self.matched, "output_diffs": self.out_bad,
                "state_diffs": self.state_bad, "largest_diff": self.worst, "errors": self.errors}
