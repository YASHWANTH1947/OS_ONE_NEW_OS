import struct
import sys
import tempfile
import unittest
from pathlib import Path

from tools.generate_demo_samples import INPUTS, MAGIC, VERSION, main


class DemoSampleTests(unittest.TestCase):
    def test_generator_produces_deterministic_valid_digit_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "samples.bin"
            original_argv = sys.argv
            try:
                sys.argv = ["generate_demo_samples.py", str(output)]
                main()
                first = output.read_bytes()
                main()
                second = output.read_bytes()
            finally:
                sys.argv = original_argv

        self.assertEqual(first, second)
        magic, version, count, reserved = struct.unpack_from("<4I", first)
        self.assertEqual((magic, version, count, reserved),
                         (MAGIC, VERSION, 10, 0))
        self.assertEqual(len(first), 16 + count * (4 + INPUTS * 4))

        labels = []
        offset = 16
        for _ in range(count):
            label = struct.unpack_from("<I", first, offset)[0]
            labels.append(label)
            pixels = struct.unpack_from(f"<{INPUTS}f", first, offset + 4)
            self.assertTrue(all(0.0 <= pixel <= 1.0 for pixel in pixels))
            offset += 4 + INPUTS * 4
        self.assertEqual(labels, list(range(10)))


if __name__ == "__main__":
    unittest.main()
