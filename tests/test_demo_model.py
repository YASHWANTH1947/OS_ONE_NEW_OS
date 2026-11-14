import struct
import sys
import tempfile
import unittest
from pathlib import Path

from tools.generate_demo_model import INPUTS, HIDDEN, MAGIC, OUTPUTS, VERSION, main


class DemoModelTests(unittest.TestCase):
    def test_generator_produces_deterministic_model_in_kernel_format(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "model.bin"
            original_argv = sys.argv
            try:
                sys.argv = ["generate_demo_model.py", str(output)]
                main()
                first = output.read_bytes()
                main()
                second = output.read_bytes()
            finally:
                sys.argv = original_argv

        self.assertEqual(first, second)
        self.assertEqual(len(first), 407100)
        header = struct.unpack_from("<5I", first)
        self.assertEqual(header, (MAGIC, VERSION, INPUTS, HIDDEN, OUTPUTS))


if __name__ == "__main__":
    unittest.main()
