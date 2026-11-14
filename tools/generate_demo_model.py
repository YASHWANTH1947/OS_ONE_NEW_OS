#!/usr/bin/env python3
"""Write deterministic, untrained weights for boot and inference smoke tests."""

import random
import struct
import sys
from pathlib import Path

MAGIC = 0x41494D4C
VERSION = 1
INPUTS = 784
HIDDEN = 128
OUTPUTS = 10
SEED = 20250308


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/model.bin")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    layers = (
        [rng.gauss(0.0, 0.035) for _ in range(INPUTS * HIDDEN)],
        [0.0] * HIDDEN,
        [rng.gauss(0.0, 0.06) for _ in range(HIDDEN * OUTPUTS)],
        [0.0] * OUTPUTS,
    )

    with output.open("wb") as model_file:
        model_file.write(
            struct.pack("<5I", MAGIC, VERSION, INPUTS, HIDDEN, OUTPUTS)
        )
        for layer in layers:
            model_file.write(struct.pack(f"<{len(layer)}f", *layer))
    print(
        f"Wrote {output} ({output.stat().st_size} bytes; "
        "untrained demo weights)"
    )


if __name__ == "__main__":
    main()
