#!/usr/bin/env python3
"""Boot the unikernel ISO in QEMU and report whether the run completed.

The kernel terminates the VM itself through the isa-debug-exit device as soon
as it has written the benchmark block and the footprint summary. That makes a
run unattended, but it also means QEMU's exit status is not 0 on success, so
this wrapper translates the device's status and verifies the transcript instead
of letting a silent failure pass as a finished measurement.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# isa-debug-exit reports (port_value << 1) | 1, and the kernel writes 0x10.
EXIT_KERNEL_DONE = (0x10 << 1) | 1          # 33
EXIT_KERNEL_FAILED = ((0x10 | 1) << 1) | 1  # 35


def tail(path: Path, lines: int = 12) -> str:
    try:
        text = path.read_text(errors="replace").replace("\r", "")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qemu", default="qemu-system-x86_64")
    parser.add_argument("--iso", type=Path, required=True)
    parser.add_argument("--serial", choices=("stdio", "file"), default=None,
                        help="serial destination; defaults to file when "
                             "--serial-file is given, otherwise stdio")
    parser.add_argument("--serial-file", type=Path, default=None,
                        help="capture the transcript here (implies --serial file)")
    parser.add_argument("--expect-benchmark", action="store_true",
                        help="fail unless the transcript contains a result block")
    parser.add_argument("--memory", default="256M")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--qemu-args", default=None,
                        help="extra QEMU machine/CPU arguments passed "
                             "verbatim, e.g. '-cpu max' (split on spaces; "
                             "quote values that contain spaces)")
    args = parser.parse_args()

    if not args.iso.exists():
        print(f"error: ISO not found: {args.iso}", file=sys.stderr)
        return 1

    mode = args.serial or ("file" if args.serial_file else "stdio")
    serial_path = args.serial_file or Path("build/unikernel.log")

    command = [
        args.qemu,
        "-m", args.memory,
        "-cdrom", str(args.iso),
        "-display", "none",
        "-no-reboot",
        "-device", "isa-debug-exit,iobase=0xf4,iosize=0x04",
    ]
    if args.qemu_args:
        command += args.qemu_args.split()
    if mode == "stdio":
        command += ["-serial", "stdio"]
    else:
        serial_path.parent.mkdir(parents=True, exist_ok=True)
        serial_path.unlink(missing_ok=True)
        command += ["-serial", f"file:{serial_path}"]

    started = time.time()
    try:
        completed = subprocess.run(command, timeout=args.timeout)
    except FileNotFoundError:
        print(f"error: {args.qemu} is not installed or not on PATH",
              file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print(f"error: QEMU did not finish within {args.timeout:.0f} s",
              file=sys.stderr)
        if mode == "file":
            print(tail(serial_path), file=sys.stderr)
        return 1

    elapsed = time.time() - started
    status = completed.returncode

    if status == EXIT_KERNEL_FAILED:
        print("error: the kernel reported a failure before exiting",
              file=sys.stderr)
        print(tail(serial_path), file=sys.stderr)
        return 1
    if status not in (0, EXIT_KERNEL_DONE):
        print(f"error: QEMU exited with status {status}", file=sys.stderr)
        print(tail(serial_path), file=sys.stderr)
        return 1

    if mode == "stdio":
        return 0

    if not serial_path.exists() or serial_path.stat().st_size == 0:
        print(f"error: no serial output captured in {serial_path}",
              file=sys.stderr)
        return 1

    text = serial_path.read_text(errors="replace")
    if args.expect_benchmark:
        if "@@BENCH_BEGIN" not in text or "@@BENCH_END" not in text:
            print("error: the transcript has no benchmark block; the kernel "
                  "did not reach the end of the measurement", file=sys.stderr)
            print(tail(serial_path), file=sys.stderr)
            return 1

    lines = len(text.splitlines())
    print(f"QEMU finished in {elapsed:.1f} s (kernel signalled completion); "
          f"{serial_path} has {lines} lines"
          + (", benchmark block present" if args.expect_benchmark else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
