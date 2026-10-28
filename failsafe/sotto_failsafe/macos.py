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
    global _activity
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
            NS_ACTIVITY_USER_INITIATED | NS_ACTIVITY_LATENCY_CRITICAL, text)
        if not token:
            return False, "macOS refused to turn App Nap off"
        _send(objc, ctypes.c_void_p, [])(token, sel(b"retain"))
        _activity = token
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
    try:
        import subprocess
        r = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=3)
        out["power"] = "battery" if "Battery Power" in r.stdout else "mains" if "AC Power" in r.stdout else "unknown"
    except Exception:
        pass
    return out


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
