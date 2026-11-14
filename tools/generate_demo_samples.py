#!/usr/bin/env python3
"""Generate labelled seven-segment fixtures for inference smoke tests."""

import random
import struct
import sys
from pathlib import Path

INPUTS = 784
MAGIC = 0x504D4153
VERSION = 1

SEGMENTS = {
    "A": (4, 3, 24, 6),
    "B": (21, 4, 24, 14),
    "C": (21, 14, 24, 24),
    "D": (4, 22, 24, 25),
    "E": (3, 14, 6, 24),
    "F": (3, 4, 6, 14),
    "G": (4, 12, 24, 15),
}
DIGIT_SEGMENTS = (
    "ABCDEF",
    "BC",
    "ABGED",
    "ABGCD",
    "FGBC",
    "AFGCD",
    "AFGECD",
    "ABC",
    "ABCDEFG",
    "ABFGCD",
)


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/samples.bin")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20250308)
    records = []
    for label, active_segments in enumerate(DIGIT_SEGMENTS):
        pixels = [0.0] * INPUTS
        for segment in active_segments:
            x1, y1, x2, y2 = SEGMENTS[segment]
            for y in range(y1, y2):
                for x in range(x1, x2):
                    pixels[y * 28 + x] = 1.0
        for index, value in enumerate(pixels):
            if value == 0.0 and rng.random() < 0.01:
                pixels[index] = 0.12
        records.append((label, pixels))

    with output.open("wb") as samples_file:
        samples_file.write(struct.pack("<4I", MAGIC, VERSION, len(records), 0))
        for label, pixels in records:
            samples_file.write(struct.pack("<I", label))
            samples_file.write(struct.pack(f"<{INPUTS}f", *pixels))
    print(
        f"Wrote {output} ({output.stat().st_size} bytes; "
        "synthetic smoke fixtures, not MNIST)"
    )


if __name__ == "__main__":
    main()
