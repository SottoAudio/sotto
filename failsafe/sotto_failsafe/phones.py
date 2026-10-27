"""Phones on this computer: the phones bus (solo) also played on the computer's own headphones or
speakers, so you can listen without a Dante headphone device.

The soundcard Sotto Voce runs on (Dante Virtual Soundcard) and the computer's own output run on
different clocks, so a small buffer sits between them: about 20 ms is kept, and when the clocks
drift apart a few samples are skipped or repeated. That's fine for listening, but it isn't for
show audio, which keeps going out on the soundcard's phones outputs as before.

The engine's callback pushes each block (no waiting, no allocation); the computer's own output
stream pulls from the buffer on its own thread.
"""
import threading

import numpy as np

from .diagnostics import LOG

FS = 48000


class LocalPhones:
    def __init__(self):
        self.stream = None
        self.device = None
        self.name = None
        self.cap = FS // 2                                  # 0.5 s ring
        self.ring = np.zeros((self.cap, 2), np.float32)
        self.w = self.r = 0
        self.count = 0
        self.target = int(0.020 * FS)                       # keep ~20 ms buffered
        self.high = int(0.080 * FS)
        self.skips = self.gaps = 0
        self.lock = threading.Lock()
        self.error = None

    # ------------------------------------------------------------ the engine (audio thread)
    def push(self, left, right):
        if self.stream is None:
            return
        n = len(left)
        if self.count + n > self.cap:                       # output stopped pulling: drop the oldest
            self.r = (self.r + n) % self.cap
            self.count -= n
        w = self.w
        first = min(n, self.cap - w)
        self.ring[w:w + first, 0] = left[:first]
        self.ring[w:w + first, 1] = right[:first]
        if first < n:
            self.ring[:n - first, 0] = left[first:]
            self.ring[:n - first, 1] = right[first:]
        self.w = (w + n) % self.cap
        self.count += n

    # ------------------------------------------------------------ the computer's output (its thread)
    def _pull(self, outdata, frames, tinfo, status):
        if self.count > self.high:                          # clocks drifted: catch up
            skip = self.count - self.target
            self.r = (self.r + skip) % self.cap
            self.count -= skip
            self.skips += 1
        take = min(frames, self.count)
        r = self.r
        first = min(take, self.cap - r)
        outdata[:first] = self.ring[r:r + first]
        if first < take:
            outdata[first:take] = self.ring[:take - first]
        if take < frames:
            outdata[take:] = 0.0
            if self.stream is not None and self.count == 0 and take == 0:
                pass                                        # nothing playing yet
            else:
                self.gaps += 1
        self.r = (r + take) % self.cap
        self.count -= take

    # ------------------------------------------------------------ choosing (web server thread)
    def open(self, device):
        """device: a PortAudio output device index, or None to stop."""
        with self.lock:
            self.close()
            if device is None:
                return {"ok": True, "device": None}
            import sounddevice as sd
            info = sd.query_devices(int(device))
            if info["max_output_channels"] < 1:
                raise ValueError(f"{info['name']} has no outputs")
            ch = min(2, info["max_output_channels"])
            self.w = self.r = self.count = 0
            self.skips = self.gaps = 0

            def cb(outdata, frames, tinfo, status):
                if ch == 2:
                    self._pull(outdata, frames, tinfo, status)
                else:                                       # mono output: the left side
                    tmp = np.zeros((frames, 2), np.float32)
                    self._pull(tmp, frames, tinfo, status)
                    outdata[:, 0] = tmp[:, 0]

            s = sd.OutputStream(device=int(device), samplerate=FS, channels=ch, dtype="float32",
                                latency="low", callback=cb)
            s.start()
            self.stream, self.device, self.name = s, int(device), info["name"]
            LOG.info("Phones on this computer: %s", info["name"])
            return {"ok": True, "device": self.device, "name": self.name}

    def close(self):
        s, self.stream = self.stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                LOG.exception("Could not close the phones output")
            LOG.info("Phones on this computer: off")
        self.device = self.name = None

    def state(self):
        return {"device": self.device, "name": self.name, "skips": self.skips, "gaps": self.gaps,
                "buffered_ms": round(1000 * self.count / FS, 1) if self.stream else None}
