"""Core Audio's own view of the devices (macOS only; runs on the build Mac's real devices)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "failsafe"))

from sotto_failsafe import coreaudio  # noqa: E402


@unittest.skipUnless(sys.platform == "darwin", "Core Audio is macOS only")
class CoreAudioTests(unittest.TestCase):
    def test_devices_have_names_buffers_and_latency(self):
        devs = coreaudio.devices()
        self.assertFalse(any("error" in d for d in devs), devs)
        if not devs:
            self.skipTest("this Mac has no audio devices")
        for d in devs:
            self.assertTrue(d["name"], d)
            self.assertGreaterEqual(d["inputs"] + d["outputs"], 0)
        with_buffer = [d for d in devs if d["buffer_frames"]]
        self.assertTrue(with_buffer, devs)
        d = with_buffer[0]
        self.assertGreater(d["rate"], 0)
        lo, hi = d["buffer_range"]
        self.assertTrue(lo <= d["buffer_frames"] <= hi, d)
        self.assertGreater(d["total_ms"]["out"] + d["total_ms"]["in"], 0)
        self.assertEqual(coreaudio.find(d["name"])["id"], d["id"])
        print("\n" + "\n".join(f"  {x['name']}: {x['inputs']} in / {x['outputs']} out, buffer {x['buffer_frames']} "
                                f"{x['buffer_range']}, total {x.get('total_ms')}" for x in devs))


class ElsewhereTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == "darwin", "macOS has Core Audio")
    def test_nothing_elsewhere(self):
        self.assertEqual(coreaudio.devices(), [])
        self.assertIsNone(coreaudio.find("anything"))


if __name__ == "__main__":
    unittest.main()
