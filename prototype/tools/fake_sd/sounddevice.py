"""Stand-in for the sounddevice package, for testing without an audio interface.
Feeds FAKE_SD_INPUT (.npy, frames x channels) to the callback in real time and saves what the
callback writes to FAKE_SD_OUTPUT (.npy)."""
import os, threading, time
import numpy as np


class _Base:
    def __init__(self, device=None, samplerate=48000, blocksize=480, channels=2, dtype="float32",
                 latency="low", callback=None):
        self.fs, self.bs, self.cb = samplerate, blocksize, callback
        self.nin, self.nout = channels if isinstance(channels, tuple) else (0, channels)
        self.latency = (0.0058, 0.0063) if self.nin else 0.0063
        self.stop_ev = threading.Event()

    def _run(self):
        x = np.load(os.environ["FAKE_SD_INPUT"]) if self.nin else None
        n = (x.shape[0] if x is not None else int(30 * self.fs)) // self.bs
        outs = []
        t0 = time.perf_counter()
        for b in range(n):
            if self.stop_ev.is_set():
                break
            out = np.zeros((self.bs, self.nout), np.float32)
            if self.nin:
                ind = np.zeros((self.bs, self.nin), np.float32)
                blk = x[b * self.bs:(b + 1) * self.bs, : self.nin]
                ind[:, : blk.shape[1]] = blk
                self.cb(ind, out, self.bs, None, None)
            else:
                self.cb(out, self.bs, None, None)
            outs.append(out)
            wait = t0 + (b + 1) * self.bs / self.fs - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        np.save(os.environ.get("FAKE_SD_OUTPUT", "fake_out.npy"), np.concatenate(outs))

    def __enter__(self):
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop_ev.set()
        self.th.join()


class Stream(_Base):
    pass


class OutputStream(_Base):
    pass


def query_devices():
    return [{"name": "Fake interface", "max_input_channels": 8, "max_output_channels": 8, "hostapi": 0}]


def query_hostapis():
    return [{"name": "Fake"}]
