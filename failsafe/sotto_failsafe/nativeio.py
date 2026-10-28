"""Sotto Voce's native soundcard side (engine/audio-io/audio_io.c), through ctypes.

The soundcard's real-time callback runs in C and only copies audio through two lock-free rings.
Sotto Voce's 10 ms engine runs on its own Python thread: read a block, work on it, write it back.
The output plays a fixed safety margin behind the input; if the engine is ever later than that,
the C side plays the talkers dry with a short crossfade and holds HEARTBEAT off until a manual
re-arm (the fail-safe rules: any fault latches).
"""
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LIBNAME = {"darwin": "sotto_audio_io.dylib", "win32": "sotto_audio_io.dll"}.get(sys.platform, "sotto_audio_io.so")
STATS = ("cycles", "underrun_cycles", "fallback_frames", "overruns", "dropped_blocks", "resyncs", "margin_frames",
         "trace_rows")


def find():
    dirs = [os.environ.get("SOTTO_AUDIO_IO_DIR")]
    if getattr(sys, "frozen", False):
        dirs.append(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    dirs += [os.path.join(ROOT, "build", "audio-io"), os.path.join(ROOT, "build", "audio-io", "Release"), HERE]
    for d in dirs:
        if d and os.path.isfile(os.path.join(d, LIBNAME)):
            return os.path.join(d, LIBNAME)
    return None


_lib = None


def library():
    """The loaded library, or None if this build doesn't have it."""
    global _lib
    if _lib is None:
        path = find()
        if not path:
            return None
        lib = ctypes.CDLL(path)
        P, I, D, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_float)
        lib.sio_version.restype = I
        lib.sio_create.restype = P
        lib.sio_create.argtypes = [I, I, I, I, I, I]
        lib.sio_set_fallback.restype = I
        lib.sio_set_fallback.argtypes = [P, I, ctypes.POINTER(I), ctypes.POINTER(I), I]
        lib.sio_cycle.argtypes = [P, F, F, I, D, I]
        lib.sio_read.restype = I
        lib.sio_read.argtypes = [P, F, I, I]
        lib.sio_write.restype = I
        lib.sio_write.argtypes = [P, F, I]
        lib.sio_clear_heartbeat.argtypes = [P]
        lib.sio_heartbeat_blocked.restype = I
        lib.sio_heartbeat_blocked.argtypes = [P]
        lib.sio_stats.argtypes = [P, ctypes.POINTER(ctypes.c_int64)]
        lib.sio_trace.restype = I
        lib.sio_trace.argtypes = [P, ctypes.c_int64, ctypes.POINTER(ctypes.c_double), I, ctypes.POINTER(ctypes.c_int64)]
        lib.sio_error.restype = ctypes.c_char_p
        lib.sio_error.argtypes = [P]
        lib.sio_open.restype = I
        lib.sio_open.argtypes = [P, ctypes.c_char_p, I, I, D, ctypes.POINTER(D), ctypes.POINTER(D)]
        lib.sio_close.argtypes = [P]
        lib.sio_destroy.argtypes = [P]
        _lib = lib
    return _lib


def portaudio_path():
    """The PortAudio library python-sounddevice uses (so device numbers match)."""
    try:
        import sounddevice as sd
        name = getattr(sd, "_libname", None)
        if name and os.path.isfile(name):
            return name
    except Exception:
        pass
    import ctypes.util
    return ctypes.util.find_library("portaudio")


def _fptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class NativeIO:
    """One soundcard stream with its two rings. `margin` = the safety margin in samples: how much
    later than the input the output plays, beyond the block itself (the engine's deadline is the
    margin plus one soundcard buffer)."""

    def __init__(self, n_in, n_out, block=480, io=160, fs=48000, margin=480, seconds=1.0):
        lib = library()
        if lib is None:
            raise RuntimeError("this build has no native audio library")
        self.lib, self.n_in, self.n_out, self.block, self.io, self.fs = lib, n_in, n_out, block, io, fs
        self.margin = int(margin)
        self.prefill = max(0, block - io + self.margin)       # output trails input by this many samples
        self.h = lib.sio_create(n_in, n_out, block, fs, self.prefill, int(seconds * fs))
        if not self.h:
            raise MemoryError("couldn't create the audio rings")
        self._stats = (ctypes.c_int64 * len(STATS))()
        self._tbuf = np.zeros((16384, 6), np.float64)
        self._trace_from = 0
        self.open_ = False

    @property
    def added_ms(self):
        return 1000.0 * self.prefill / self.fs

    def set_fallback(self, talker_in, talker_out, heartbeat_out):
        n = len(talker_in)
        a = (ctypes.c_int * max(n, 1))(*talker_in)
        b = (ctypes.c_int * max(n, 1))(*talker_out)
        rc = self.lib.sio_set_fallback(self.h, n, a, b, -1 if heartbeat_out is None else int(heartbeat_out))
        if rc:
            raise ValueError(f"fallback channels out of range ({rc})")

    def open(self, device, latency_s, pa_path=None):
        pa = pa_path or portaudio_path()
        if not pa:
            raise RuntimeError("PortAudio library not found")
        il, ol = ctypes.c_double(0), ctypes.c_double(0)
        rc = self.lib.sio_open(self.h, pa.encode(), int(device), int(self.io), float(latency_s),
                               ctypes.byref(il), ctypes.byref(ol))
        if rc:
            raise RuntimeError((self.lib.sio_error(self.h) or b"").decode(errors="replace") or f"error {rc}")
        self.open_ = True
        return il.value, ol.value

    def cycle(self, indata, outdata, frames, adc=float("nan"), status=0):
        """One soundcard buffer by hand (the simulated frame and tests): float32 C-contiguous arrays."""
        self.lib.sio_cycle(self.h, _fptr(indata), _fptr(outdata), int(frames), float(adc), int(status))

    def read(self, buf, timeout_ms=100):
        return self.lib.sio_read(self.h, _fptr(buf), buf.shape[0], int(timeout_ms))

    def write(self, buf):
        return self.lib.sio_write(self.h, _fptr(buf), buf.shape[0])

    def clear_heartbeat(self):
        self.lib.sio_clear_heartbeat(self.h)

    def heartbeat_blocked(self):
        return bool(self.lib.sio_heartbeat_blocked(self.h))

    def stats(self):
        self.lib.sio_stats(self.h, self._stats)
        return dict(zip(STATS, (int(v) for v in self._stats)))

    def new_trace_rows(self):
        """Rows recorded since the last call: wall time, capture time, frames, callback ms, status, margin."""
        end = ctypes.c_int64(0)
        n = self.lib.sio_trace(self.h, self._trace_from, self._tbuf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                               len(self._tbuf), ctypes.byref(end))
        self._trace_from = end.value
        return self._tbuf[:n].copy()

    def close(self):
        if self.h and self.open_:
            self.lib.sio_close(self.h)
            self.open_ = False

    def destroy(self):
        if self.h:
            self.lib.sio_destroy(self.h)
            self.h = None
