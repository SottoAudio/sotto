"""Test sources: audio Sotto Voce plays into its own inputs, so the processing can be heard and checked
without a mic, a PGM feed or any Dante routing.

Kinds (per input channel; talker inputs and the PGM input only):
  tone         1 kHz sine
  pink         pink noise
  speech       speech: a WAV file you load, or speech this computer makes with its own
               text-to-speech (macOS `say`, Windows SAPI, espeak on Linux)
  bleed        speech plus PA bleed: the speech, plus the built-in programme as it would reach a
               mic on stage (delayed, coloured, with room reverb). The PGM input gets the same
               programme at the same time, so the echo canceller can be tested end to end.
  programme    the built-in programme (for the PGM input): chords, bass and drums, made here
  file         a WAV file you load (looped)

Fail-safe: test sources only play while Sotto Voce is disarmed. Arming switches them all off (and the
app logs it). They can't be put on a PROBE input, so they can't make a check go green.

Audio-thread safety: every signal is made or loaded when it's chosen (on the web server's thread);
the audio thread only copies samples, from loops held in memory, into a buffer made in advance.
"""
import os
import subprocess
import sys
import tempfile
import threading
import wave

import numpy as np

FS = 48000
KINDS = ("tone", "pink", "speech", "bleed", "programme", "file")
MAX_FILE_S = 300                     # longest test audio kept (the start of the file); any file size is fine
SPEECH_TEXT = ("Stand by lighting cue forty two. Lighting forty two, go. Stand by video seven and pyro one. "
               "Video seven, go. Pyro one, go. Follow spots, pick up the artist downstage centre. "
               "Sound, stand by for the band intro. Band intro, go. Thank you, nice work everyone.")


def db(x):
    return 10 ** (x / 20)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) + 1e-12


# ---------------------------------------------------------------------------------- signals
def tone(seconds=1.0, hz=1000.0):
    n = int(FS * seconds)
    return np.sin(2 * np.pi * hz * np.arange(n) / FS).astype(np.float32)          # whole cycles: loops cleanly


def pink(seconds=10.0, seed=7):
    n = int(FS * seconds)
    w = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    f[0] = f[1]
    x = np.fft.irfft(w / np.sqrt(f), n)                                          # circular: loops cleanly
    return (x / rms(x) * 0.1).astype(np.float32)


def programme(seconds=16.0, seed=3):
    """A stand-in for a show's programme feed: chords, a bass line and a drum loop at 120 bpm.
    Broadband and dynamic, like music through a PA, which is what an echo canceller needs."""
    n = int(FS * seconds)
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    beat = 0.5
    x = np.zeros(n)
    chords = [(220.0, 277.18, 329.63), (196.0, 246.94, 293.66), (174.61, 220.0, 261.63), (196.0, 246.94, 311.13)]
    bar = 4 * beat
    for b in range(int(seconds / bar)):
        s, e = int(b * bar * FS), int((b + 1) * bar * FS)
        tt = t[s:e] - t[s]
        env = np.minimum(1.0, tt / 0.02) * np.exp(-tt / 3.0)
        for f0 in chords[b % 4]:
            for h, a in ((1, 1.0), (2, 0.35), (3, 0.18), (4, 0.08)):
                x[s:e] += 0.06 * a * env * np.sin(2 * np.pi * f0 * h * tt + rng.uniform(0, 6.28))
        root = chords[b % 4][0] / 2
        for q in range(4):
            qs = s + int(q * beat * FS)
            qe = min(e, qs + int(beat * FS))
            tq = t[qs:qe] - t[qs]
            x[qs:qe] += 0.25 * np.exp(-tq / 0.25) * np.sin(2 * np.pi * root * tq)
    for k in range(int(seconds / beat)):
        s = int(k * beat * FS)
        m = min(n - s, int(0.25 * FS))
        tt = np.arange(m) / FS
        if k % 2 == 0:                                                           # kick
            x[s:s + m] += 0.6 * np.exp(-tt / 0.06) * np.sin(2 * np.pi * (50 + 80 * np.exp(-tt / 0.03)) * tt)
        else:                                                                    # snare
            x[s:s + m] += 0.25 * np.exp(-tt / 0.08) * rng.standard_normal(m)
        hs = s + int(0.25 * FS)                                                  # hi-hat on the off-beat
        hm = min(n - hs, int(0.05 * FS))
        if hm > 0:
            x[hs:hs + hm] += 0.06 * np.exp(-np.arange(hm) / FS / 0.015) * np.diff(rng.standard_normal(hm + 1))
    return (x / rms(x) * 0.1).astype(np.float32)


def room(prog, delay_ms=40.0, seconds_rt=0.4, seed=11):
    """The programme as a stage mic hears it: the PA's direct sound after delay_ms, then the room
    (reverb about 5 dB below the direct sound, high frequencies rolled off). Circular, so it loops
    with the programme."""
    n = len(prog)
    rng = np.random.default_rng(seed)
    m = int(seconds_rt * FS)
    tail = rng.standard_normal(m) * np.exp(-6.9 * np.arange(m) / m)              # -60 dB over RT60
    tail = np.convolve(tail, np.ones(8) / 8)[:m] * 0.15                           # gentle high cut
    tail[:48] = 0.0                                                               # first reflections after 1 ms
    ir = tail
    ir[0] = 1.0                                                                   # the direct sound
    d = int(delay_ms * FS / 1000)
    full = np.zeros(m + d)
    full[d:] = ir
    y = np.fft.irfft(np.fft.rfft(prog, n) * np.fft.rfft(full, n), n)            # circular convolution
    return (y / rms(y) * rms(prog)).astype(np.float32)


AUDIO_EXTS = (".wav", ".wave", ".bwf", ".aif", ".aiff", ".aifc", ".caf", ".mp3", ".m4a", ".aac", ".flac", ".mp4")


def read_wav(f):
    """Any WAV a DAW writes, from an open binary file: PCM 8/16/24/32-bit, 32/64-bit float, plain or
    WAVE_FORMAT_EXTENSIBLE, with any extra chunks (bext, iXML, ...), any size (RF64-style 0 or
    0xFFFFFFFF data sizes too). Reads only the first MAX_FILE_S seconds.
    Returns (samples (frames, channels) float32, rate)."""
    import struct
    head = f.read(12)
    if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    fmt, pos = None, 12
    while True:
        f.seek(pos)
        h = f.read(8)
        if len(h) < 8:
            break
        cid, size = h[:4], struct.unpack("<I", h[4:])[0]
        if cid == b"fmt ":
            body = f.read(min(size, 64))
            tag, ch, rate, _, align, bits = struct.unpack_from("<HHIIHH", body, 0)
            if tag == 0xFFFE and len(body) >= 26:            # extensible: the real format is in the GUID
                tag = struct.unpack_from("<H", body, 24)[0]
            fmt = (tag, ch, rate, align, bits)
        elif cid == b"data":
            if fmt is None:
                raise ValueError("the WAV file's format comes after its audio, which Sotto Voce can't read")
            tag, ch, rate, align, bits = fmt
            want = rate * MAX_FILE_S * align if align else 0
            raw = f.read(want if size in (0, 0xFFFFFFFF) else min(size, want))
            return _decode(raw, fmt), rate
        pos += 8 + size + (size & 1)
    raise ValueError("the WAV file has no audio in it" if fmt else "the WAV file has no format chunk")


def _decode(raw, fmt):
    tag, ch, rate, align, bits = fmt
    width = align // ch if ch else 0
    raw = raw[:(len(raw) // align) * align] if align else b""
    if tag == 1 and width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif tag == 1 and width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        i = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
        x = np.where(i >= 1 << 23, i - (1 << 24), i).astype(np.float32) / float(1 << 23)
    elif tag == 1 and width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / float(1 << 31)
    elif tag == 1 and width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128.0
    elif tag == 3 and width == 4:
        x = np.frombuffer(raw, "<f4").astype(np.float32)
    elif tag == 3 and width == 8:
        x = np.frombuffer(raw, "<f8").astype(np.float32)
    else:
        raise ValueError(f"this WAV's format ({'float' if tag == 3 else 'PCM' if tag == 1 else f'type {tag}'}, "
                         f"{bits}-bit) isn't one Sotto Voce reads")
    return np.nan_to_num(x.reshape(-1, ch))


def parse_wav(data):
    """read_wav() for WAV bytes."""
    import io
    return read_wav(io.BytesIO(data))


def write_wav(path, x):
    """Mono float32 WAV at 48 kHz (how Sotto Voce keeps test audio)."""
    import struct
    x = np.ascontiguousarray(x, "<f4")
    fmt = struct.pack("<HHIIHH", 3, 1, FS, FS * 4, 4, 32)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + x.nbytes) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        f.write(b"data" + struct.pack("<I", x.nbytes) + x.tobytes())


def _afconvert(src, name):
    """Other audio formats (AIFF, MP3, M4A, CAF, FLAC...) converted with macOS's own afconvert;
    returns the path of a WAV in a temporary folder the caller removes."""
    if sys.platform != "darwin":
        raise ValueError("Sotto Voce reads WAV files on this computer. Save the audio as a WAV and upload that.")
    d = tempfile.mkdtemp(prefix="sotto-convert-")
    dst = os.path.join(d, "out.wav")
    r = subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEF32@48000", src, dst], capture_output=True, text=True,
                       timeout=600)
    if r.returncode != 0 or not os.path.isfile(dst):
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        raise ValueError(f"macOS couldn't read {os.path.basename(name)} as audio: "
                         f"{(r.stderr or r.stdout).strip()[:200] or 'unknown format'}")
    return dst


def _finish(x, rate):
    """(frames, channels) at any rate to mono float32 at 48 kHz, checked."""
    x = x.mean(axis=1)
    if rate != FS:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(FS, int(rate))
        x = resample_poly(x, FS // g, int(rate) // g).astype(np.float32)
    if len(x) < FS // 10:
        raise ValueError("the file is too short (under 0.1 s)")
    if float(np.max(np.abs(x))) < 1e-6:
        raise ValueError(f"the first {MAX_FILE_S // 60} minutes of the file are silent" if len(x) >= FS * MAX_FILE_S
                         else "the file is silent")
    return np.ascontiguousarray(x, np.float32)


def load_audio_path(path, name=None):
    """An audio file of any size on disk to mono float32 at 48 kHz: its first MAX_FILE_S seconds."""
    name = name or os.path.basename(path)
    if os.path.getsize(path) == 0:
        raise ValueError("the file is empty")
    with open(path, "rb") as f:
        head = f.read(12)
    if head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE":
        with open(path, "rb") as f:
            return _finish(*read_wav(f))
    wav = _afconvert(path, name)
    try:
        with open(wav, "rb") as f:
            return _finish(*read_wav(f))
    finally:
        import shutil
        shutil.rmtree(os.path.dirname(wav), ignore_errors=True)


def load_audio(data, name="audio.wav"):
    """Audio file bytes to mono float32 at 48 kHz, at most MAX_FILE_S seconds."""
    if not data:
        raise ValueError("the file is empty")
    if data[:4] in (b"RIFF", b"RF64") and data[8:12] == b"WAVE":
        return _finish(*parse_wav(data))
    d = tempfile.mkdtemp(prefix="sotto-in-")
    try:
        src = os.path.join(d, "in" + (os.path.splitext(name)[1] or ".audio"))
        with open(src, "wb") as f:
            f.write(data)
        return load_audio_path(src, name)
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def load_wav(data_or_path):
    """Kept for callers that pass a path or WAV bytes."""
    if not isinstance(data_or_path, (bytes, bytearray)):
        return load_audio_path(data_or_path)
    return load_audio(bytes(data_or_path))


def make_speech(folder):
    """Speech made by this computer's own text-to-speech, cached as speech.wav. Returns (path, how)."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "speech.wav")
    if os.path.isfile(path):
        return path, "made earlier"
    tmp = os.path.join(folder, "speech-making.wav")
    if sys.platform == "darwin":
        cmd = ["say", "-o", tmp, "--data-format=LEI16@48000", SPEECH_TEXT]
        how = "macOS text-to-speech"
    elif sys.platform == "win32":
        script = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.SetOutputToWaveFile('{tmp}'); $s.Speak('{SPEECH_TEXT}'); $s.Dispose()")
        cmd = ["powershell", "-NoProfile", "-Command", script]
        how = "Windows text-to-speech"
    else:
        cmd = ["espeak", "-w", tmp, SPEECH_TEXT]
        how = "espeak"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"this computer can't make speech ({how}: {e.__class__.__name__}). "
                           "Upload a WAV of speech instead.")
    if r.returncode != 0 or not os.path.isfile(tmp):
        raise RuntimeError(f"{how} couldn't make speech: {(r.stderr or r.stdout).strip()[:300]}")
    os.replace(tmp, path)
    return path, how


# ---------------------------------------------------------------------------------- the player
class _Loop:
    def __init__(self, x, gain):
        self.x = np.ascontiguousarray(x, np.float32)
        self.gain = float(gain)


class TestSources:
    """Holds the chosen sources and plays them into the input block. set()/off() run on another
    thread and replace the plan in one assignment; apply() runs on the audio thread."""

    def __init__(self, folder, allowed, names=None):
        self.folder = folder                       # where loaded files and made speech are kept
        self.allowed = dict(allowed)               # input channel (0-based) -> what it is ("talker 1", "PGM")
        self.names = names or {}
        self.plan = {}                             # channel -> list of _Loop (mixed), replaces the input
        self.desc = {}                             # channel -> what's playing, for the console
        self.clock = 0                             # samples played: every loop follows it, so bleed and PGM line up
        self.work = None
        self.lock = threading.Lock()
        self._programme = None
        self._room = None
        self._rooms = {}
        self.files = {}                            # name -> samples, loaded this session
        self.switched_off = 0

    # ------------------------------------------------------------ choosing (web server thread)
    def _prog(self, file=None):
        """The programme (built in, or a WAV file) and how it sounds in the mic."""
        if file:
            key = ("prog", file)
            if key not in self._rooms:
                x = self._file(file)
                self._rooms[key] = (x, room(x))
            return self._rooms[key]
        if self._programme is None:
            self._programme = programme()
            self._room = room(self._programme)
        return self._programme, self._room

    def _speech(self):
        if "speech" not in self.files:
            path, how = make_speech(self.folder)
            self.files["speech"] = load_wav(path)
            self.speech_how = how
        return self.files["speech"]

    def _file(self, name):
        """A file loaded this session, or one saved by an earlier session."""
        if name not in self.files:
            path = os.path.join(self.folder, os.path.basename(name))
            if not os.path.isfile(path):
                raise ValueError("load the file first")
            self.files[name] = load_audio_path(path, name)
        return self.files[name]

    def saved_files(self):
        try:
            on_disk = [f for f in os.listdir(self.folder) if f.lower().endswith(AUDIO_EXTS) and f not in
                       ("speech.wav", "speech-making.wav")]
        except OSError:
            on_disk = []
        return sorted(set(on_disk) | {k for k in self.files if k != "speech"})

    def set(self, channel, kind, level_db=-20.0, file=None, bleed_db=-6.0, programme_file=None):
        """channel: 0-based input. level_db: speech/tone level (RMS dBFS). bleed_db: how loud the PA
        bleed is next to the speech (for 'bleed'). file: the WAV for 'file' or 'speech' (else the
        computer's own speech). programme_file: a WAV to use as the programme ('programme', and
        the bleed's programme), else the built-in one. Returns what's now playing."""
        if channel not in self.allowed:
            raise ValueError("test sources can only go on a talker input or the PGM input")
        if kind not in KINDS:
            raise ValueError(f"unknown test source {kind!r}")
        level = db(float(level_db))
        loops = []
        if kind == "tone":
            x = tone()
            loops.append(_Loop(x, level / rms(x)))
            what = f"1 kHz tone at {level_db:.0f} dBFS"
        elif kind == "pink":
            x = pink()
            loops.append(_Loop(x, level / rms(x)))
            what = f"pink noise at {level_db:.0f} dBFS"
        elif kind in ("speech", "file"):
            if kind == "file" or file:
                name = file or ""
                x = self._file(name)
                what = f"{name} at {level_db:.0f} dBFS"
            else:
                x = self._speech()
                what = f"speech ({self.speech_how}) at {level_db:.0f} dBFS"
            loops.append(_Loop(x, level / rms(x)))
        elif kind == "programme":
            prog, _ = self._prog(programme_file)
            loops.append(_Loop(prog, level / rms(prog)))
            what = f"{programme_file or 'built-in programme'} at {level_db:.0f} dBFS"
        elif kind == "bleed":
            x = self._file(file) if file else self._speech()
            prog, rm = self._prog(programme_file)
            loops.append(_Loop(x, level / rms(x)))
            loops.append(_Loop(rm, level * db(float(bleed_db)) / rms(rm)))
            what = (f"{file or 'speech'} at {level_db:.0f} dBFS with PA bleed {bleed_db:+.0f} dB "
                    f"({programme_file or 'built-in programme'} delayed 40 ms, with room reverb)")
        with self.lock:
            plan = dict(self.plan)
            desc = dict(self.desc)
            plan[channel] = loops
            desc[channel] = {"kind": kind, "level_db": float(level_db), "file": file, "programme_file": programme_file,
                             "bleed_db": float(bleed_db), "what": what}
            if kind == "bleed":                      # the PGM input needs the programme that bleeds
                pgm = [c for c, w in self.allowed.items() if w == "PGM"]
                if pgm and (pgm[0] not in desc or desc[pgm[0]]["kind"] != "programme"
                            or desc[pgm[0]].get("programme_file") != programme_file):
                    prog, _ = self._prog(programme_file)
                    plan[pgm[0]] = [_Loop(prog, db(-20.0) / rms(prog))]
                    desc[pgm[0]] = {"kind": "programme", "level_db": -20.0, "file": None, "programme_file": programme_file,
                                    "what": f"{programme_file or 'built-in programme'} at -20 dBFS (for the PA bleed)"}
            self.plan, self.desc = plan, desc       # one assignment each: the audio thread sees old or new
        return desc[channel]

    def off(self, channel=None):
        with self.lock:
            if channel is None:
                self.plan, self.desc = {}, {}
            else:
                plan, desc = dict(self.plan), dict(self.desc)
                plan.pop(channel, None)
                desc.pop(channel, None)
                self.plan, self.desc = plan, desc

    def remove(self, name):
        """Forgets an uploaded file (and stops anything playing it)."""
        name = os.path.basename(name or "")
        if name in ("", "speech.wav"):
            raise ValueError("no such file")
        with self.lock:
            used = [c for c, d in self.desc.items() if name in (d.get("file"), d.get("programme_file"))]
        for c in used:
            self.off(c)
        self.files.pop(name, None)
        for k in [k for k in self._rooms if k[1] == name]:
            self._rooms.pop(k, None)
        try:
            os.remove(os.path.join(self.folder, name))
        except OSError:
            pass
        return {"removed": name, "stopped": [c + 1 for c in used]}

    def load(self, name, data):
        """Audio file bytes (small files; tests)."""
        name = self._check_name(name)
        return self._keep(name, load_audio(data, name))

    def load_path(self, path, name=None):
        """An audio file of any size on disk: only its first MAX_FILE_S seconds are read and kept."""
        name = self._check_name(name or os.path.basename(path))
        return self._keep(name, load_audio_path(path, name))

    @staticmethod
    def _check_name(name):
        name = os.path.basename(name or "recording.wav")
        if not name.lower().endswith(AUDIO_EXTS):
            raise ValueError(f"{name} doesn't look like an audio file (WAV, AIFF, MP3, M4A, CAF or FLAC)")
        return name

    def _keep(self, name, x):
        """Keeps the audio Sotto Voce uses (mono, 48 kHz, at most MAX_FILE_S) in the test audio
        folder under the file's own name, not the original file, so a 2 GB recording stays small."""
        os.makedirs(self.folder, exist_ok=True)
        write_wav(os.path.join(self.folder, name), x)
        self.files[name] = x
        for k in [k for k in self._rooms if k[1] == name]:     # a new file under an old name
            self._rooms.pop(k, None)
        return {"name": name, "seconds": round(len(x) / FS, 1), "cut": len(x) >= FS * MAX_FILE_S - FS // 10}

    def state(self):
        return {"playing": {str(c + 1): d for c, d in self.desc.items()},
                "allowed": {str(c + 1): w for c, w in self.allowed.items()},
                "files": self.saved_files(),
                "switched_off_by_arming": self.switched_off,
                "finder": sys.platform == "darwin"}

    # ------------------------------------------------------------ playing (audio thread)
    def apply(self, indata, frames):
        """Returns indata itself when nothing is playing, else a copy with the sources in it."""
        plan = self.plan
        if not plan:
            self.clock += frames
            return indata
        if self.work is None or self.work.shape != indata.shape:
            self.work = np.zeros(indata.shape, np.float32)  # once (and again only if the block size changes)
        w = self.work
        w[:] = indata
        t0 = self.clock
        for ch, loops in plan.items():
            if ch >= w.shape[1]:
                continue
            col = w[:, ch]
            col[:] = 0.0
            for lp in loops:
                n = len(lp.x)
                s = t0 % n
                first = min(frames, n - s)
                col[:first] += lp.x[s:s + first] * lp.gain
                done = first
                while done < frames:                 # wrap (loops are at least 0.1 s long)
                    take = min(frames - done, n)
                    col[done:done + take] += lp.x[:take] * lp.gain
                    done += take
        self.clock += frames
        return w
