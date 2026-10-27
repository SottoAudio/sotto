"""What Core Audio itself says about the audio devices (macOS), standard library ctypes only.

PortAudio reports one latency figure per direction. Core Audio knows what it is made of, and
those parts decide what Sotto Voce can do about it:

  buffer_frames        the device's I/O buffer (one Core Audio cycle), and the sizes it allows
  latency_frames       what the driver adds (in and out)
  safety_frames        the safety offset Core Audio keeps (in and out)
  stream_latency       extra latency per stream (in and out)
  rate                 the nominal sample rate
  running_elsewhere    whether another app has the device running (and so may set its buffer)

devices() lists every device; find(name) returns one. Elsewhere both return [] / None.
"""
import ctypes
import ctypes.util
import struct
import sys


def _fourcc(s):
    return struct.unpack(">I", s.encode())[0]


SYSTEM_OBJECT = 1
GLOBAL, INPUT, OUTPUT = _fourcc("glob"), _fourcc("inpt"), _fourcc("outp")
MAIN = 0
P = {
    "devices": _fourcc("dev#"),
    "name": _fourcc("lnam"),
    "buffer": _fourcc("fsiz"),
    "buffer_range": _fourcc("fsz#"),
    "latency": _fourcc("ltnc"),
    "safety": _fourcc("saft"),
    "rate": _fourcc("nsrt"),
    "streams": _fourcc("stm#"),
    "stream_latency": _fourcc("ltnc"),
    "running_somewhere": _fourcc("gone"),
    "transport": _fourcc("tran"),
    "config": _fourcc("slay"),
}


class _Addr(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]


class _Range(ctypes.Structure):
    _fields_ = [("minimum", ctypes.c_double), ("maximum", ctypes.c_double)]


_lib = None


def _libs():
    global _lib
    if _lib is None:
        ca = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        ca.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
        ca.AudioObjectGetPropertyDataSize.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32,
                                                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        ca.AudioObjectGetPropertyData.restype = ctypes.c_int32
        ca.AudioObjectGetPropertyData.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32,
                                                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        ca.AudioObjectHasProperty.restype = ctypes.c_bool
        ca.AudioObjectHasProperty.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Addr)]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        _lib = (ca, cf)
    return _lib


def _get(obj, sel, scope=GLOBAL, ctype=ctypes.c_uint32):
    ca, _ = _libs()
    a = _Addr(sel, scope, MAIN)
    if not ca.AudioObjectHasProperty(obj, ctypes.byref(a)):
        return None
    v = ctype()
    size = ctypes.c_uint32(ctypes.sizeof(v))
    if ca.AudioObjectGetPropertyData(obj, ctypes.byref(a), 0, None, ctypes.byref(size), ctypes.byref(v)) != 0:
        return None
    return v


def _array(obj, sel, scope=GLOBAL):
    ca, _ = _libs()
    a = _Addr(sel, scope, MAIN)
    size = ctypes.c_uint32(0)
    if ca.AudioObjectGetPropertyDataSize(obj, ctypes.byref(a), 0, None, ctypes.byref(size)) != 0 or not size.value:
        return []
    n = size.value // 4
    arr = (ctypes.c_uint32 * n)()
    if ca.AudioObjectGetPropertyData(obj, ctypes.byref(a), 0, None, ctypes.byref(size), arr) != 0:
        return []
    return list(arr)


def _name(obj):
    _, cf = _libs()
    ref = _get(obj, P["name"], GLOBAL, ctypes.c_void_p)
    if ref is None or not ref.value:
        return None
    buf = ctypes.create_string_buffer(512)
    ok = cf.CFStringGetCString(ref.value, buf, 512, 0x08000100)       # UTF-8
    cf.CFRelease(ref.value)
    return buf.value.decode("utf-8", "replace") if ok else None


def _channels(obj, scope):
    """Channels in one direction, from the stream configuration (an AudioBufferList)."""
    ca, _ = _libs()
    a = _Addr(P["config"], scope, MAIN)
    size = ctypes.c_uint32(0)
    if ca.AudioObjectGetPropertyDataSize(obj, ctypes.byref(a), 0, None, ctypes.byref(size)) != 0 or not size.value:
        return 0
    buf = ctypes.create_string_buffer(size.value)
    if ca.AudioObjectGetPropertyData(obj, ctypes.byref(a), 0, None, ctypes.byref(size), buf) != 0:
        return 0
    raw = buf.raw
    n = struct.unpack_from("<I", raw, 0)[0]
    total, off = 0, 8                                   # mNumberBuffers, then 8-byte aligned AudioBuffers
    for _ in range(n):
        total += struct.unpack_from("<I", raw, off)[0]  # mNumberChannels
        off += 16                                       # UInt32 channels, UInt32 bytes, void* data
    return total


def describe(obj):
    def u(sel, scope=GLOBAL):
        v = _get(obj, sel, scope)
        return None if v is None else int(v.value)

    rng = _get(obj, P["buffer_range"], GLOBAL, _Range)
    rate = _get(obj, P["rate"], GLOBAL, ctypes.c_double)
    out = {"id": obj, "name": _name(obj),
           "inputs": _channels(obj, INPUT), "outputs": _channels(obj, OUTPUT),
           "rate": None if rate is None else rate.value,
           "buffer_frames": u(P["buffer"]),
           "buffer_range": None if rng is None else [int(rng.minimum), int(rng.maximum)],
           "latency_frames": {"in": u(P["latency"], INPUT), "out": u(P["latency"], OUTPUT)},
           "safety_frames": {"in": u(P["safety"], INPUT), "out": u(P["safety"], OUTPUT)},
           "running_elsewhere": bool(u(P["running_somewhere"]) or 0)}
    sl = {}
    for key, scope in (("in", INPUT), ("out", OUTPUT)):
        lat = [u2 for s in _array(obj, P["streams"], scope)
               for u2 in [(_get(s, P["stream_latency"]) or ctypes.c_uint32(0)).value]]
        sl[key] = max(lat) if lat else None
    out["stream_latency_frames"] = sl
    # What Core Audio adds per direction, before any extra buffering by the app or PortAudio.
    b = out["buffer_frames"] or 0
    for key in ("in", "out"):
        out.setdefault("total_frames", {})[key] = (b + (out["latency_frames"][key] or 0) + (out["safety_frames"][key] or 0)
                                                   + (sl[key] or 0))
    if out["rate"]:
        out["total_ms"] = {k: round(1000 * v / out["rate"], 2) for k, v in out["total_frames"].items()}
    return out


def devices():
    if sys.platform != "darwin":
        return []
    try:
        return [describe(d) for d in _array(SYSTEM_OBJECT, P["devices"])]
    except Exception as e:
        return [{"error": repr(e)}]


def find(name):
    """The device whose name matches (exactly, else the first that contains it)."""
    if not name:
        return None
    devs = [d for d in devices() if d.get("name")]
    for d in devs:
        if d["name"] == name:
            return d
    for d in devs:
        if name.lower() in d["name"].lower():
            return d
    return None
