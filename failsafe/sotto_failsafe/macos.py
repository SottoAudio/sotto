"""Keeps macOS from slowing Sotto Voce down.

Sotto Voce has no window of its own, so macOS treats it as a background app: App Nap throttles
its timers and the scheduler may run its threads on the efficiency cores. Both make the 10 ms
processing deadline impossible. Two calls fix that (standard library ctypes only, no-ops elsewhere):

  keep_responsive()    once at start-up: tells macOS this is latency-critical, user-started work
                       (NSProcessInfo beginActivityWithOptions), which turns App Nap off
  fast_thread()        from the processing thread: QoS "user interactive", so it stays on the
                       performance cores
Each returns (ok, message) for the status page.
"""
import ctypes
import ctypes.util
import sys

_activity = None                    # the token must be kept, or the activity ends

NS_ACTIVITY_USER_INITIATED = 0x00FFFFFF
NS_ACTIVITY_LATENCY_CRITICAL = 0xFF00000000
NS_ACTIVITY_IDLE_DISPLAY_SLEEP_DISABLED = 1 << 40   # no display sleep or screensaver while it runs
DISPLAY_AWAKE = False
QOS_CLASS_USER_INTERACTIVE = 0x21
QOS = {"user_interactive": 0x21, "user_initiated": 0x19, "default": 0x15, "utility": 0x11, "background": 0x09}


def _objc():
    ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc") or "/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    return objc


def _send(objc, restype, argtypes):
    """objc_msgSend with an exact prototype (required on Apple silicon)."""
    f = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)
    return f(ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value)


def keep_responsive(reason="Sotto Voce is processing live comms audio"):
    """Also keeps the display awake: the screensaver starting stalled DVS and Sotto Voce for about
    3 s on 29 Oct."""
    global _activity, DISPLAY_AWAKE
    if sys.platform != "darwin":
        return True, "not needed on this system"
    if _activity:
        return True, "App Nap off"
    try:
        objc = _objc()
        sel = objc.sel_registerName
        info = _send(objc, ctypes.c_void_p, [])(objc.objc_getClass(b"NSProcessInfo"), sel(b"processInfo"))
        text = _send(objc, ctypes.c_void_p, [ctypes.c_char_p])(objc.objc_getClass(b"NSString"),
                                                               sel(b"stringWithUTF8String:"), reason.encode())
        token = _send(objc, ctypes.c_void_p, [ctypes.c_uint64, ctypes.c_void_p])(
            info, sel(b"beginActivityWithOptions:reason:"),
            NS_ACTIVITY_USER_INITIATED | NS_ACTIVITY_LATENCY_CRITICAL | NS_ACTIVITY_IDLE_DISPLAY_SLEEP_DISABLED, text)
        if not token:
            return False, "macOS refused to turn App Nap off"
        _send(objc, ctypes.c_void_p, [])(token, sel(b"retain"))
        _activity = token
        DISPLAY_AWAKE = True
        return True, "App Nap off"
    except Exception as e:                               # never stop the app over this
        return False, f"couldn't turn App Nap off: {e!r}"


def fast_thread():
    ok, msg = set_thread_qos("user_interactive")
    return ok, ("performance cores" if ok else msg)


def set_thread_qos(name):
    """Sets the calling thread's macOS quality of service ('background' keeps it on the efficiency
    cores, 'user_interactive' prefers the performance cores)."""
    if sys.platform != "darwin":
        return True, "not needed on this system"
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libSystem.B.dylib")
        f = libc.pthread_set_qos_class_self_np
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_uint, ctypes.c_int]
        rc = f(QOS[name], 0)
        return (rc == 0), (name if rc == 0 else f"couldn't set thread priority {name} (error {rc})")
    except Exception as e:
        return False, f"couldn't set thread priority: {e!r}"


THREAD_TIME_CONSTRAINT_POLICY = 2


class _TimeConstraint(ctypes.Structure):
    _fields_ = [("period", ctypes.c_uint32), ("computation", ctypes.c_uint32),
                ("constraint", ctypes.c_uint32), ("preemptible", ctypes.c_int)]


def realtime_thread(period_ms=10.0, computation_ms=4.0, constraint_ms=10.0):
    """Makes the calling thread a real-time thread, the way DAWs schedule their audio helper
    threads: every period_ms it needs about computation_ms of CPU, finished within constraint_ms.
    macOS then runs it ahead of ordinary work (the screen, browsers, other apps). Returns (ok,
    message). Elsewhere, or if macOS refuses, falls back to the highest everyday priority."""
    if sys.platform != "darwin":
        return True, "not needed on this system"
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libSystem.B.dylib")

        class _TB(ctypes.Structure):
            _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]
        tb = _TB()
        libc.mach_timebase_info(ctypes.byref(tb))
        to_abs = lambda ms: int(ms * 1e6 * tb.denom / tb.numer)       # noqa: E731
        libc.mach_thread_self.restype = ctypes.c_uint32
        libc.thread_policy_set.restype = ctypes.c_int
        libc.thread_policy_set.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        pol = _TimeConstraint(to_abs(period_ms), to_abs(computation_ms), to_abs(constraint_ms), 1)
        rc = libc.thread_policy_set(libc.mach_thread_self(), THREAD_TIME_CONSTRAINT_POLICY, ctypes.byref(pol), 4)
        if rc == 0:
            return True, f"real-time ({computation_ms:g} ms every {period_ms:g} ms)"
        ok, msg = fast_thread()
        return ok, f"real-time refused (error {rc}); {msg}"
    except Exception as e:
        ok, msg = fast_thread()
        return ok, f"real-time not available ({e!r}); {msg}"


class Workgroup:
    """The calling thread joins a Core Audio device's I/O workgroup: Apple's way for an app's own
    audio threads to be scheduled as part of the soundcard's real-time work, to its deadline and on
    the performance cores (macOS 11+). Join and leave from the same thread; the thread must already
    be real-time (realtime_thread)."""

    def __init__(self):
        self.wg = None
        self.token = None
        self.note = "not joined"

    def join(self, device_id, name="the soundcard"):
        self.leave()
        if sys.platform != "darwin":
            self.note = "not on this system"
            return False
        try:
            from . import coreaudio
            wg = coreaudio.io_workgroup(device_id)
            if not wg:
                self.note = f"{name} has no audio workgroup"
                return False
            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            libc.os_workgroup_join.restype = ctypes.c_int
            libc.os_workgroup_join.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            token = ctypes.create_string_buffer(64)           # os_workgroup_join_token_s (40 bytes)
            rc = libc.os_workgroup_join(wg, token)
            if rc != 0:
                libc.os_release.argtypes = [ctypes.c_void_p]
                libc.os_release(wg)
                self.note = f"couldn't join {name}'s audio workgroup (error {rc})"
                return False
            self.wg, self.token = wg, token
            self.note = f"in {name}'s audio workgroup"
            return True
        except Exception as e:
            self.note = f"audio workgroup not available ({e!r})"
            return False

    def leave(self):
        if self.wg is None:
            return
        try:
            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            libc.os_workgroup_leave.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            libc.os_workgroup_leave(self.wg, self.token)
            libc.os_release.argtypes = [ctypes.c_void_p]
            libc.os_release(self.wg)
        except Exception:
            pass
        self.wg = self.token = None
        self.note = "not joined"


THERMAL = ("normal", "warm", "hot (slowing down)", "critical (slowing down)")


def power_state():
    """Low Power Mode, thermal state and power source, for the status page. {} elsewhere."""
    if sys.platform != "darwin":
        return {}
    out = {}
    try:
        objc = _objc()
        sel = objc.sel_registerName
        info = _send(objc, ctypes.c_void_p, [])(objc.objc_getClass(b"NSProcessInfo"), sel(b"processInfo"))
        out["low_power"] = bool(_send(objc, ctypes.c_bool, [])(info, sel(b"isLowPowerModeEnabled")))
        t = int(_send(objc, ctypes.c_long, [])(info, sel(b"thermalState")))
        out["thermal"] = THERMAL[t] if 0 <= t < len(THERMAL) else str(t)
    except Exception as e:
        out["error"] = repr(e)
    out["power"] = power_source()
    return out


_iokit = None


def power_source():
    """'mains', 'battery' or 'unknown', asked of macOS directly. (Running pmset instead started a
    new process every 10 s, and starting a process holds up every Python thread, the engine's too.)"""
    global _iokit
    try:
        if _iokit is None:
            io = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/IOKit.framework/IOKit")
            cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            io.IOPSCopyPowerSourcesInfo.restype = ctypes.c_void_p
            io.IOPSGetProvidingPowerSourceType.restype = ctypes.c_void_p
            io.IOPSGetProvidingPowerSourceType.argtypes = [ctypes.c_void_p]
            cf.CFStringGetCString.restype = ctypes.c_bool
            cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
            cf.CFRelease.argtypes = [ctypes.c_void_p]
            _iokit = (io, cf)
        io, cf = _iokit
        blob = io.IOPSCopyPowerSourcesInfo()
        if not blob:
            return "unknown"
        try:
            typ = io.IOPSGetProvidingPowerSourceType(blob)       # not ours to release
            buf = ctypes.create_string_buffer(64)
            if not typ or not cf.CFStringGetCString(typ, buf, 64, 0x08000100):
                return "unknown"
            v = buf.value.decode(errors="replace")
        finally:
            cf.CFRelease(blob)
        return "battery" if v == "Battery Power" else "mains" if v in ("AC Power", "UPS Power") else "unknown"
    except Exception:
        return "unknown"


def mic_permission():
    """macOS microphone permission for this app: 'allowed', 'denied', 'not asked', 'restricted',
    or why it couldn't be read. Core Audio inputs are silent without it."""
    if sys.platform != "darwin":
        return "allowed"
    try:
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
        avf = ctypes.CDLL("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
        media = ctypes.c_void_p.in_dll(avf, "AVMediaTypeAudio")
        objc = _objc()
        cls = objc.objc_getClass(b"AVCaptureDevice")
        st = _send(objc, ctypes.c_long, [ctypes.c_void_p])(cls, objc.sel_registerName(b"authorizationStatusForMediaType:"),
                                                          media.value)
        return {0: "not asked", 1: "restricted", 2: "denied", 3: "allowed"}.get(int(st), f"status {st}")
    except Exception as e:
        return f"couldn't read it ({e!r})"
