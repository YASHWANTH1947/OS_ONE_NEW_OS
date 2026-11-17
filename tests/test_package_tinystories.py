import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools import package_tinystories


def model_fixture(vocab=4):
    config = (8, 16, 1, 2, 1, vocab, 128)
    kv_dim = config[0] * config[4] // config[3]
    floats = (
        config[5] * config[0]
        + config[2] * config[0]
        + config[2] * config[0] * config[0]
        + config[2] * config[0] * kv_dim * 2
        + config[2] * config[0] * config[0]
        + config[2] * config[0]
        + config[2] * config[1] * config[0] * 2
        + config[2] * config[0] * config[1]
        + config[0]
        + config[6] * (config[0] // config[3])
    )
    return struct.pack("<7i", *config) + struct.pack(
        f"<{floats}f", *([0.0] * floats)
    )


def tokenizer_fixture(vocab=4):
    values = [b"<0>", b"<1>", b"<2>", b"<3>"][:vocab]
    result = bytearray(struct.pack("<I", 3))
    for index, value in enumerate(values):
        result.extend(struct.pack("<fI", float(index), len(value)))
        result.extend(value)
    return bytes(result)


class TinyStoriesPackagingTests(unittest.TestCase):
    def test_packages_and_validates_matching_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.bin"
            tokenizer = root / "tokenizer.bin"
            model_out = root / "llm-model.bin"
            tokenizer_out = root / "tokenizer.pkg"
            manifest = root / "manifest.json"
            checkpoint.write_bytes(model_fixture())
            tokenizer.write_bytes(tokenizer_fixture())
            args = type("Args", (), {
                "checkpoint": checkpoint,
                "tokenizer": tokenizer,
                "model_out": model_out,
                "tokenizer_out": tokenizer_out,
                "manifest": manifest,
            })()
            package_tinystories.write_package(args)
            result = package_tinystories.validate_outputs(args)
            self.assertEqual(result["model"]["vocab_size"], 4)
            self.assertTrue(result["model"]["shared_classifier"])
            self.assertEqual(
                json.loads(manifest.read_text())["format"],
                "ai-os-tinystories",
            )

    def test_rejects_truncated_checkpoint(self):
        with self.assertRaises(package_tinystories.PackageError):
            package_tinystories.parse_model(b"\0" * 20)

    def test_rejects_tokenizer_vocab_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.bin"
            tokenizer = root / "tokenizer.bin"
            checkpoint.write_bytes(model_fixture())
            tokenizer.write_bytes(tokenizer_fixture(3))
            with self.assertRaises(package_tinystories.PackageError):
                package_tinystories.parse_tokenizer(
                    tokenizer.read_bytes(), 4
                )
