#!/usr/bin/env python3
"""Download a small public TinyStories checkpoint for kernel integration."""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL_BASE_URL = (
    "https://huggingface.co/karpathy/tinyllamas/resolve/main/"
)
DEFAULT_TOKENIZER_260K_URL = (
    DEFAULT_MODEL_BASE_URL + "stories260K/tok512.bin?download=true"
)
DEFAULT_TOKENIZER_15M_URL = (
    "https://raw.githubusercontent.com/karpathy/llama2.c/master/"
    "tokenizer.bin"
)


def download(url: str, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-size", choices=("stories260K", "stories15M"),
                        default="stories260K")
    parser.add_argument("--model-url", default=None)
    parser.add_argument("--tokenizer-url", default=None)
    args = parser.parse_args()

    model_path = args.output_dir / f"{args.model_size}.bin"
    tokenizer_path = args.output_dir / "tokenizer.bin"
    model_url = args.model_url
    tokenizer_url = args.tokenizer_url
    if model_url is None:
        if args.model_size == "stories260K":
            model_url = (DEFAULT_MODEL_BASE_URL +
                         "stories260K/stories260K.bin?download=true")
        else:
            model_url = (DEFAULT_MODEL_BASE_URL +
                         "stories15M.bin?download=true")
    if tokenizer_url is None:
        tokenizer_url = (DEFAULT_TOKENIZER_260K_URL
                         if args.model_size == "stories260K"
                         else DEFAULT_TOKENIZER_15M_URL)
    try:
        model_sha256 = download(model_url, model_path)
        tokenizer_sha256 = download(tokenizer_url, tokenizer_path)
    except (OSError, urllib.error.URLError) as error:
        print(f"error: model download failed: {error}", file=sys.stderr)
        return 1

    manifest = {
        "format": "llama2.c-checkpoint",
        "model": model_path.name,
        "tokenizer": tokenizer_path.name,
        "model_url": model_url,
        "tokenizer_url": tokenizer_url,
        "model_sha256": model_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "execution_target": "ai-os-kernel",
        "status": "downloaded; ready for kernel packaging and LLM ISO boot",
    }
    manifest_path = args.output_dir / f"{args.model_size}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded {model_path} ({model_path.stat().st_size} bytes)")
    print(f"Downloaded {tokenizer_path} ({tokenizer_path.stat().st_size} bytes)")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
