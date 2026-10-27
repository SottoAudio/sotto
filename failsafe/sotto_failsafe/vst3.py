"""VST3 plugin inserts, through Sotto's small C hosting library (engine/plugin-test/vst3host.cpp).

  lib = load_library()               # None if the library isn't there (inserts are then off)
  p = Plugin(lib, path, rate=48000, block=480, state=b"")
  p.process(x)                       # mono float32 block in, same-length block out
  p.info()                           # name, maker, latency, controls
  p.set_param(id, 0.5); p.get_state()

Scanning for plugins is done by the sotto-plugin-test program in its own process, so a plugin
that crashes while loading can't take Sotto down with it.
"""
import ctypes
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LIBNAME = {"darwin": "sotto_vst3host.dylib", "win32": "sotto_vst3host.dll"}.get(sys.platform, "sotto_vst3host.so")
EXENAME = "sotto-plugin-test.exe" if sys.platform == "win32" else "sotto-plugin-test"


def _candidates(name):
    env = os.environ.get("SOTTO_VST3HOST_DIR")
    dirs = [env] if env else []
    if getattr(sys, "frozen", False):
        dirs.append(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    dirs += [os.path.join(ROOT, "build", "plugin-test"), os.path.join(ROOT, "build", "plugin-test", "Release"), HERE]
    return [os.path.join(d, name) for d in dirs if d]


def find(name):
    for c in _candidates(name):
        if os.path.isfile(c):
            return c
    return None


def load_library():
    path = find(LIBNAME)
    if not path:
        return None
    lib = ctypes.CDLL(path)
    lib.svh_version.restype = ctypes.c_int
    lib.svh_open.restype = ctypes.c_void_p
    lib.svh_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_int, ctypes.c_void_p,
                             ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.svh_close.argtypes = [ctypes.c_void_p]
    lib.svh_process.restype = ctypes.c_int
    lib.svh_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.svh_latency.restype = ctypes.c_int
    lib.svh_latency.argtypes = [ctypes.c_void_p]
    lib.svh_set_param.restype = ctypes.c_int
    lib.svh_set_param.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_double]
    lib.svh_info.restype = ctypes.c_int
    lib.svh_info.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.svh_param_text.restype = ctypes.c_int
    lib.svh_param_text.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_double, ctypes.c_char_p, ctypes.c_int]
    lib.svh_get_state.restype = ctypes.c_int
    lib.svh_get_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.path = path
    return lib


class Plugin:
    """One running plugin instance, used as a mono insert."""

    def __init__(self, lib, path, cls="", rate=48000, block=480, state=b""):
        self.lib, self.path, self.block = lib, path, block
        err = ctypes.create_string_buffer(512)
        st = state or b""
        buf = ctypes.create_string_buffer(st, len(st)) if st else None
        self.h = lib.svh_open(path.encode(), (cls or "").encode(), float(rate), int(block), buf, len(st), err, 512)
        if not self.h:
            raise RuntimeError(err.value.decode(errors="replace") or "could not open the plugin")
        self.latency = int(lib.svh_latency(self.h))
        self._in = np.zeros(block, np.float32)
        self._out = np.zeros(block, np.float32)
        self.failures = 0

    def process(self, x):
        n = len(x)
        self._in[:n] = x
        ok = self.lib.svh_process(self.h, self._in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  self._out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)
        if not ok:
            self.failures += 1
        return self._out[:n].copy()

    def info(self):
        n = self.lib.svh_info(self.h, None, 0)
        buf = ctypes.create_string_buffer(n + 1)
        self.lib.svh_info(self.h, buf, n + 1)
        return json.loads(buf.value.decode(errors="replace"))

    def set_param(self, pid, norm):
        self.lib.svh_set_param(self.h, int(pid), float(norm))

    def param_text(self, pid, norm):
        buf = ctypes.create_string_buffer(128)
        self.lib.svh_param_text(self.h, int(pid), float(norm), buf, 128)
        return buf.value.decode(errors="replace")

    def get_state(self):
        n = self.lib.svh_get_state(self.h, None, 0)
        if n <= 0:
            return b""
        buf = ctypes.create_string_buffer(n)
        self.lib.svh_get_state(self.h, buf, n)
        return buf.raw[:n]

    def close(self):
        if self.h:
            self.lib.svh_close(self.h)
            self.h = None


def scan(timeout_s=60):
    """Lists installed VST3 effects by running sotto-plugin-test --list --json in its own process.
    SOTTO_EXTRA_VST3 (folders, separated like PATH) adds plugins outside the standard folders."""
    exe = find(EXENAME)
    found = []
    if exe:
        try:
            r = subprocess.run([exe, "--list", "--json"], capture_output=True, text=True, timeout=timeout_s)
            found = json.loads(r.stdout or "{}").get("plugins", [])
        except (subprocess.TimeoutExpired, ValueError, OSError) as e:
            return {"ok": False, "error": f"scan failed: {e!r}", "plugins": []}
    extra = os.environ.get("SOTTO_EXTRA_VST3", "")
    for d in filter(None, extra.split(os.pathsep)):
        for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if name.endswith(".vst3") and exe:
                try:
                    r = subprocess.run([exe, os.path.join(d, name), "--json", "--instances", "1"], capture_output=True,
                                       text=True, timeout=timeout_s)
                    j = json.loads(r.stdout)
                    found.append({"name": j["plugin"], "vendor": j["vendor"], "version": j["version"], "type": j["type"],
                                  "path": os.path.join(d, name)})
                except (subprocess.TimeoutExpired, ValueError, OSError, KeyError):
                    pass
    return {"ok": exe is not None, "error": None if exe else "sotto-plugin-test is not installed with this build",
            "plugins": found}
