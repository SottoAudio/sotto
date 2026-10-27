"""Multitrack recorder for the live prototype: 24-bit WAV, new file every 30 minutes, markers and a
once-a-second status log, plus a track map so the recording drops straight into run.py."""
import csv
import json
import os
import queue
import threading
import time
import wave

import numpy as np


def _to_int24(x):
    x = np.clip(np.nan_to_num(x), -1.0, 1.0 - 1.0 / 8388608)
    i = (x * 8388608.0).astype("<i4")
    b = i.view(np.uint8).reshape(-1, 4)[:, :3]
    return b.tobytes()


class Recorder:
    def __init__(self, folder, channel_names, fs=48000, roll_s=1800, talkers=None, refs=None):
        self.folder = folder
        self.names = channel_names
        self.fs = fs
        self.roll = int(roll_s * fs)
        self.talkers = talkers or {}
        self.refs = refs or {}
        self.q = queue.Queue(maxsize=2000)
        self.on = False
        self.dropped = 0
        self.wav = None
        self.files = []
        self.samples_total = 0
        self.samples_file = 0
        self.t0 = None
        self.take = None
        self.th = None
        self._marker_f = self._log_f = None

    # -- control (any thread)
    def start(self):
        if self.on:
            return
        os.makedirs(self.folder, exist_ok=True)
        self.take = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(self.folder, self.take)
        os.makedirs(self.dir, exist_ok=True)
        self.files, self.samples_total, self.samples_file = [], 0, 0
        self._marker_f = open(os.path.join(self.dir, "markers.csv"), "w", newline="")
        self._markers = csv.writer(self._marker_f)
        self._markers.writerow(["take_time_s", "file", "file_time_s", "clock", "note"])
        self._log_f = open(os.path.join(self.dir, "status_log.jsonl"), "w")
        with open(os.path.join(self.dir, "channels.txt"), "w") as f:
            for i, n in enumerate(self.names):
                f.write(f"{i + 1}\t{n}\n")
            f.write("\nProcessed tracks run 10 ms (480 samples) behind the mic tracks "
                    "(the STFT overlap). Block buffering is the same for both.\n")
        self.t0 = time.time()
        self.on = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def stop(self):
        if not self.on:
            return
        self.on = False
        self.q.put(None)
        if self.th:
            self.th.join(timeout=5)
        self._close_file()
        for f in (self._marker_f, self._log_f):
            if f:
                f.close()
        self._write_trackmap()

    def push(self, block):
        """block: (n, ch) float. Called from the audio thread; never blocks."""
        if not self.on:
            return
        try:
            self.q.put_nowait(block.copy())
        except queue.Full:
            self.dropped += 1

    def mark(self, note=""):
        if not self.on:
            return None
        t = self.samples_total / self.fs
        ft = self.samples_file / self.fs
        fname = os.path.basename(self.files[-1]) if self.files else ""
        self._markers.writerow([f"{t:.2f}", fname, f"{ft:.2f}", time.strftime("%H:%M:%S"), note])
        self._marker_f.flush()
        return t

    def log(self, state):
        if self.on and self._log_f:
            self._log_f.write(json.dumps({"t": round(self.samples_total / self.fs, 2), **state}) + "\n")
            self._log_f.flush()

    def status(self):
        return {"on": self.on, "take": self.take, "seconds": round(self.samples_total / self.fs, 1),
                "file": os.path.basename(self.files[-1]) if self.files else None,
                "dropped_blocks": self.dropped, "folder": getattr(self, "dir", None)}

    # -- writer thread
    def _open_file(self):
        idx = len(self.files) + 1
        path = os.path.join(self.dir, f"{self.take}_part{idx:02d}.wav")
        w = wave.open(path, "wb")
        w.setnchannels(len(self.names))
        w.setsampwidth(3)
        w.setframerate(self.fs)
        self.wav = w
        self.files.append(path)
        self.samples_file = 0

    def _close_file(self):
        if self.wav:
            self.wav.close()
            self.wav = None

    def _loop(self):
        self._open_file()
        while True:
            b = self.q.get()
            if b is None:
                break
            if self.samples_file + len(b) > self.roll:
                self._close_file()
                self._open_file()
            self.wav.writeframes(_to_int24(b.reshape(-1)))
            self.samples_file += len(b)
            self.samples_total += len(b)

    def _write_trackmap(self):
        """One track map per part file, ready for: python run.py process --map <file>."""
        for path in self.files:
            tm = {"files": [os.path.basename(path)], "talkers": self.talkers,
                  "references": self.refs,
                  "_note": "Edit 'segments' from markers.csv: pa_only = PA playing, nobody "
                           "talking; talk = known calls. Times in seconds within this file.",
                  "segments": {"pa_only": [], "talk": []}}
            with open(os.path.splitext(path)[0] + "_trackmap.json", "w") as f:
                json.dump(tm, f, indent=2)
