"""Keeps macOS from slowing Sotto Bench down.

Sotto Bench has no window of its own, so macOS treats it as a background app: App Nap throttles
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


def keep_responsive(reason="Sotto Bench is processing live comms audio"):
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
