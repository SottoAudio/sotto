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
