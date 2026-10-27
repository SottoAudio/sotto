"""Test sources and phones on this computer."""
import io
import os
import shutil
import sys
import tempfile
import unittest
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "failsafe"))

from sotto_failsafe import testsrc as T  # noqa: E402
from sotto_failsafe.phones import LocalPhones  # noqa: E402


def wav_bytes(x, rate=48000, width=3):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(rate)
        if width == 3:
            i = np.clip(np.round(x * (1 << 23)), -(1 << 23), (1 << 23) - 1).astype("<i4")
            w.writeframes(i.view(np.uint8).reshape(-1, 4)[:, :3].tobytes())
        else:
            w.writeframes(np.clip(np.round(x * 32767), -32768, 32767).astype("<i2").tobytes())
    return b.getvalue()


def riff(fmt_tag, ch, rate, bits, payload, extensible=False, extra_chunks=b""):
    """A WAV written by hand, the way DAWs do: float, extensible, extra chunks before the audio."""
    import struct
    align = ch * bits // 8
    if extensible:
        fmt = struct.pack("<HHIIHHHHI", 0xFFFE, ch, rate, rate * align, align, bits, 22, bits, 0)
        fmt += struct.pack("<H", fmt_tag) + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    else:
        fmt = struct.pack("<HHIIHH", fmt_tag, ch, rate, rate * align, align, bits)
    chunk = lambda cid, body: cid + struct.pack("<I", len(body)) + body + (b"\x00" if len(body) & 1 else b"")
    body = b"WAVE" + chunk(b"fmt ", fmt) + extra_chunks + chunk(b"data", payload)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def play(ts, seconds, n_in=4):
    blocks = []
    for _ in range(int(seconds * 100)):
        x = np.zeros((480, n_in), np.float32)
        blocks.append(ts.apply(x, 480).copy())
    return np.concatenate(blocks)


class TestSourceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ts = T.TestSources(self.dir, {0: "talker 1", 1: "talker 2", 3: "PGM"})

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_nothing_playing_passes_the_input_through_untouched(self):
        x = np.ones((480, 4), np.float32)
        self.assertIs(self.ts.apply(x, 480), x)

    def test_tone_at_the_level_asked_for(self):
        self.ts.set(1, "tone", level_db=-20)
        y = play(self.ts, 1.0)
        self.assertAlmostEqual(20 * np.log10(T.rms(y[:, 1])), -20.0, places=1)
        self.assertEqual(float(np.max(np.abs(y[:, 0]))), 0.0)            # other inputs untouched

    def test_only_talker_inputs_and_pgm(self):
        with self.assertRaises(ValueError):
            self.ts.set(2, "tone")                                        # e.g. the PROBE return

    def test_bleed_reaches_the_mic_40_ms_after_the_pgm(self):
        x = (np.random.default_rng(0).standard_normal(48000) * 0.05).astype(np.float32)
        self.ts.load("voice.wav", wav_bytes(x))
        self.ts.set(0, "bleed", level_db=-20, file="voice.wav", bleed_db=0)
        self.assertEqual(self.ts.desc[3]["kind"], "programme")          # PGM gets the same programme
        y = play(self.ts, 4.0)
        pgm, mic = y[:, 3].astype(np.float64), y[:, 0].astype(np.float64)
        n = len(pgm)
        cross = np.fft.rfft(mic, 2 * n) * np.conj(np.fft.rfft(pgm, 2 * n))
        xc = np.fft.irfft(cross / (np.abs(cross) + 1e-12), 2 * n)[:4800]      # whitened (GCC-PHAT), as delay trackers do
        self.assertAlmostEqual(int(np.argmax(np.abs(xc))) / 48.0, 40.0, delta=0.1)

    def test_wav_files_24_bit_and_resampled(self):
        x = 0.25 * np.sin(2 * np.pi * 441 * np.arange(44100) / 44100)
        y = T.load_wav(wav_bytes(x, rate=44100, width=3))
        self.assertEqual(len(y), 48000)
        self.assertAlmostEqual(float(np.max(np.abs(y[1000:-1000]))), 0.25, places=2)
        with self.assertRaises(Exception):
            T.load_wav(b"not a wav file")

    def test_every_wav_a_daw_writes(self):
        import struct
        x = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)).astype(np.float32)
        odd = b"bext" + struct.pack("<I", 3) + b"abc\x00"                  # odd-sized chunk, padded
        cases = {
            "float 32": riff(3, 1, 48000, 32, x.astype("<f4").tobytes(), extra_chunks=odd),
            "float 64": riff(3, 1, 48000, 64, x.astype("<f8").tobytes()),
            "extensible float stereo": riff(3, 2, 48000, 32, np.repeat(x, 2).astype("<f4").tobytes(), extensible=True),
            "extensible 24-bit": riff(1, 1, 48000, 24, np.frombuffer(wav_bytes(x, width=3)[44:], np.uint8).tobytes(),
                                      extensible=True),
            "PCM 32": riff(1, 1, 48000, 32, np.round(x * 2 ** 31 * 0.999).astype("<i4").tobytes()),
            "PCM 8": riff(1, 1, 48000, 8, np.round(x * 127 + 128).astype(np.uint8).tobytes()),
        }
        for name, data in cases.items():
            with self.subTest(name):
                y = T.load_audio(data, "test.wav")
                self.assertEqual(len(y), 48000)
                self.assertAlmostEqual(float(np.max(np.abs(y))), 0.5, places=1 if "8" in name else 3)

    def test_files_sotto_cant_use_say_why(self):
        silent = riff(1, 1, 48000, 16, bytes(96000))
        for data, words in ((b"", "empty"), (silent, "silent"), (riff(1, 1, 48000, 16, bytes(200)), "too short"),
                            (riff(2, 1, 48000, 4, bytes(9600)), "isn't one Sotto Voce reads")):
            with self.assertRaises(ValueError) as cm:
                T.load_audio(data, "x.wav")
            self.assertIn(words, str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            self.ts.load("notes.txt", b"hello")
        self.assertIn("doesn't look like an audio file", str(cm.exception))

    @unittest.skipUnless(sys.platform == "darwin" and shutil.which("afconvert"), "needs macOS afconvert")
    def test_aiff_through_macos_converter(self):
        import subprocess
        x = 0.25 * np.sin(2 * np.pi * 441 * np.arange(44100) / 44100)
        src, dst = os.path.join(self.dir, "a.wav"), os.path.join(self.dir, "a.aiff")
        with open(src, "wb") as f:
            f.write(wav_bytes(x, rate=44100, width=2))
        subprocess.run(["afconvert", "-f", "AIFF", "-d", "BEI16", src, dst], check=True)
        with open(dst, "rb") as f:
            r = self.ts.load("show.aiff", f.read())
        self.assertAlmostEqual(r["seconds"], 1.0, places=1)
        self.assertIn("show.aiff", self.ts.state()["files"])

    def test_off(self):
        self.ts.set(0, "pink")
        self.ts.off(0)
        self.assertEqual(self.ts.plan, {})

    @unittest.skipUnless(shutil.which("say") or shutil.which("espeak") or sys.platform == "win32",
                         "no text-to-speech on this computer")
    def test_speech_from_this_computer(self):
        self.ts.set(0, "speech", level_db=-20)
        y = play(self.ts, 2.0)
        self.assertGreater(T.rms(y[:, 0]), 0.01)


class LocalPhonesTests(unittest.TestCase):
    def test_passes_audio_and_catches_up_after_drift(self):
        p = LocalPhones()
        p.stream = object()                                             # stands in for the output stream
        sig = np.arange(48000, dtype=np.float32) / 48000
        for s in range(0, 960, 480):
            p.push(sig[s:s + 480], -sig[s:s + 480])
        out = np.zeros((960, 2), np.float32)
        p._pull(out, 960, None, None)
        np.testing.assert_array_equal(out[:, 0], sig[:960])
        np.testing.assert_array_equal(out[:, 1], -sig[:960])
        for s in range(960, 48000, 480):                                # output stalls: far too much buffered
            p.push(sig[s:s + 480], sig[s:s + 480])
        p._pull(out, 960, None, None)
        self.assertEqual(p.skips, 1)
        self.assertLessEqual(p.count, p.target)


if __name__ == "__main__":
    unittest.main()
