#!/usr/bin/env python3
"""Create a tiny deterministic character model for the ring-0 prompt shell."""

import struct
import sys
from pathlib import Path

MAGIC = 0x31494154
VERSION = 1
VOCABULARY = " abcdefghijklmnopqrstuvwxyz.,!?0123456789\n"
HIDDEN = 128

RESPONSES = {
    "o": " ready from ring zero\n",
    "s": " online and ready\n",
    "p": " try ask hello or ask status\n",
    "t": " inference runs in the kernel\n",
}


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "build/text-model.bin")
    output.parent.mkdir(parents=True, exist_ok=True)
    input_weights = [0.0] * (len(VOCABULARY) * HIDDEN)
    recurrent = [0.0] * (HIDDEN * HIDDEN)
    output_weights = [0.0] * (HIDDEN * len(VOCABULARY))
    hidden_bias = [0.0] * HIDDEN
    output_bias = [0.0] * len(VOCABULARY)
    state = 0

    for trigger, response in RESPONSES.items():
        start = state
        token = VOCABULARY.index(trigger)
        input_weights[token * HIDDEN + start] = 8.0
        for character in response:
            output_weights[state * len(VOCABULARY) +
                           VOCABULARY.index(character)] = 8.0
            next_state = state + 1
            if next_state >= HIDDEN:
                raise ValueError("demo response state table is too large")
            recurrent[state * HIDDEN + next_state] = 8.0
            state = next_state
        state += 1

    default_state = state
    input_weights[VOCABULARY.index(" ") * HIDDEN + default_state] = 8.0
    for character in " hello from the kernel\n":
        output_weights[default_state * len(VOCABULARY) +
                       VOCABULARY.index(character)] = 8.0
        next_state = default_state + 1
        recurrent[default_state * HIDDEN + next_state] = 8.0
        default_state = next_state

    with output.open("wb") as model_file:
        model_file.write(
            struct.pack("<5I", MAGIC, VERSION, len(VOCABULARY), HIDDEN, 0)
        )
        for values in (input_weights, recurrent, output_weights,
                       hidden_bias, output_bias):
            model_file.write(struct.pack(f"<{len(values)}f", *values))
    print(f"Wrote {output} ({output.stat().st_size} bytes; demo char model)")


if __name__ == "__main__":
    main()
