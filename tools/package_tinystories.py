#!/usr/bin/env python3
"""Validate and package a llama2.c TinyStories checkpoint for the kernel."""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

MODEL_MAGIC = 0x314D4C4C  # LLM1
TOKENIZER_MAGIC = 0x315A544B  # KT Z1
FORMAT_VERSION = 1
MODEL_HEADER = struct.Struct("<11I")
TOKENIZER_HEADER = struct.Struct("<7I")
CONFIG = struct.Struct("<7i")
TOKEN = struct.Struct("<fI")


class PackageError(ValueError):
    """Raised when an input does not match the supported package format."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checksum(data: bytes) -> int:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def parse_model(data: bytes) -> dict:
    if len(data) < CONFIG.size:
        raise PackageError("checkpoint is truncated before its config header")
    dim, hidden, layers, heads, kv_heads, vocab, seq_len = CONFIG.unpack_from(data)
    # llama2.c uses a positive vocabulary size to signal tied classifier
    # weights; a negative value means the classifier is stored separately.
    shared = vocab > 0
    vocab = abs(vocab)
    if not (1 <= dim <= 4096 and 1 <= hidden <= 16384):
        raise PackageError("checkpoint dimensions are outside supported bounds")
    if not (1 <= layers <= 128 and 1 <= heads <= 128):
        raise PackageError("checkpoint layer/head counts are invalid")
    if not (1 <= kv_heads <= heads and heads % kv_heads == 0):
        raise PackageError("checkpoint KV-head configuration is invalid")
    if dim % heads != 0:
        raise PackageError("model dimension must be divisible by head count")
    if not (1 <= vocab <= 1_000_000 and 1 <= seq_len <= 4096):
        raise PackageError("checkpoint vocabulary or context length is invalid")
    kv_dim = dim * kv_heads // heads
    head_size = dim // heads
    float_count = (
        vocab * dim
        + layers * dim
        + layers * dim * dim
        + layers * dim * kv_dim
        + layers * dim * kv_dim
        + layers * dim * dim
        + layers * dim
        + layers * hidden * dim
        + layers * dim * hidden
        + layers * hidden * dim
        + dim
        + seq_len * head_size
        + (0 if shared else vocab * dim)
    )
    expected = CONFIG.size + float_count * 4
    if len(data) != expected:
        raise PackageError(
            f"checkpoint size {len(data)} does not match expected {expected}"
        )
    return {
        "dim": dim,
        "hidden_dim": hidden,
        "layers": layers,
        "heads": heads,
        "kv_heads": kv_heads,
        "vocab_size": vocab,
        "seq_len": seq_len,
        "shared_classifier": shared,
        "raw_bytes": len(data),
        "expected_bytes": expected,
    }


def parse_tokenizer(data: bytes, vocab_size: int) -> dict:
    if len(data) < 4:
        raise PackageError("tokenizer is truncated before its header")
    max_token_length = struct.unpack_from("<I", data)[0]
    if not (1 <= max_token_length <= 1024):
        raise PackageError("tokenizer maximum token length is invalid")
    offset = 4
    tokens = []
    for index in range(vocab_size):
        if offset + TOKEN.size > len(data):
            raise PackageError(f"tokenizer is truncated at token {index}")
        score, length = TOKEN.unpack_from(data, offset)
        offset += TOKEN.size
        if length > max_token_length or offset + length > len(data):
            raise PackageError(f"token {index} has an invalid byte length")
        value = data[offset:offset + length]
        offset += length
        if b"\x00" in value:
            raise PackageError(f"token {index} contains an embedded NUL")
        tokens.append((score, value))
    if offset != len(data):
        raise PackageError("tokenizer contains trailing bytes")
    return {
        "vocab_size": vocab_size,
        "max_token_length": max_token_length,
        "tokens": tokens,
        "raw_bytes": len(data),
    }


def package_model(data: bytes, config: dict) -> bytes:
    flags = 1 if config["shared_classifier"] else 0
    payload = data[CONFIG.size:]
    header = MODEL_HEADER.pack(
        MODEL_MAGIC,
        FORMAT_VERSION,
        MODEL_HEADER.size,
        config["dim"],
        config["hidden_dim"],
        config["layers"],
        config["heads"],
        config["kv_heads"],
        config["vocab_size"],
        config["seq_len"] | (flags << 16),
        checksum(payload),
    )
    return header + payload


def package_tokenizer(data: bytes, tokenizer: dict) -> bytes:
    records = bytearray()
    for score, value in tokenizer["tokens"]:
        records.extend(TOKEN.pack(score, len(value)))
        records.extend(value)
    header = TOKENIZER_HEADER.pack(
        TOKENIZER_MAGIC,
        FORMAT_VERSION,
        TOKENIZER_HEADER.size,
        tokenizer["vocab_size"],
        tokenizer["max_token_length"],
        len(records),
        checksum(records),
    )
    return header + records


def write_package(args: argparse.Namespace) -> dict:
    model_data = args.checkpoint.read_bytes()
    tokenizer_data = args.tokenizer.read_bytes()
    model = parse_model(model_data)
    tokenizer = parse_tokenizer(tokenizer_data, model["vocab_size"])
    model_package = package_model(model_data, model)
    tokenizer_package = package_tokenizer(tokenizer_data, tokenizer)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.tokenizer_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_bytes(model_package)
    args.tokenizer_out.write_bytes(tokenizer_package)
    manifest = {
        "format": "ai-os-tinystories",
        "version": FORMAT_VERSION,
        "model": {
            "path": str(args.model_out),
            "sha256": sha256(model_package),
            "bytes": len(model_package),
            **{key: value for key, value in model.items() if key != "tokens"},
        },
        "tokenizer": {
            "path": str(args.tokenizer_out),
            "sha256": sha256(tokenizer_package),
            "bytes": len(tokenizer_package),
            "vocab_size": tokenizer["vocab_size"],
            "max_token_length": tokenizer["max_token_length"],
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_package(path: Path, magic: int, header: struct.Struct) -> bytes:
    data = path.read_bytes()
    if len(data) < header.size:
        raise PackageError(f"{path} is truncated")
    values = header.unpack_from(data)
    if values[0] != magic or values[1] != FORMAT_VERSION:
        raise PackageError(f"{path} has an unsupported package header")
    if values[2] != header.size:
        raise PackageError(f"{path} has an invalid header size")
    return data


def validate_outputs(args: argparse.Namespace) -> dict:
    model_data = validate_package(args.model_out, MODEL_MAGIC, MODEL_HEADER)
    tokenizer_data = validate_package(
        args.tokenizer_out, TOKENIZER_MAGIC, TOKENIZER_HEADER
    )
    model_values = MODEL_HEADER.unpack_from(model_data)
    config = {
        "dim": model_values[3],
        "hidden_dim": model_values[4],
        "layers": model_values[5],
        "heads": model_values[6],
        "kv_heads": model_values[7],
        "vocab_size": model_values[8],
        "seq_len": model_values[9] & 0xFFFF,
        "shared_classifier": bool(model_values[9] >> 16),
    }
    tokenizer_values = TOKENIZER_HEADER.unpack_from(tokenizer_data)
    if model_values[10] != checksum(model_data[MODEL_HEADER.size:]):
        raise PackageError("model payload checksum mismatch")
    if tokenizer_values[6] != checksum(
        tokenizer_data[TOKENIZER_HEADER.size:]
    ):
        raise PackageError("tokenizer payload checksum mismatch")
    if tokenizer_values[3] != config["vocab_size"]:
        raise PackageError("model and tokenizer vocabularies do not match")
    return {"model": config, "model_bytes": len(model_data),
            "tokenizer_bytes": len(tokenizer_data)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--tokenizer-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.validate_only:
            result = validate_outputs(args)
        else:
            result = write_package(args)
    except (OSError, PackageError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
