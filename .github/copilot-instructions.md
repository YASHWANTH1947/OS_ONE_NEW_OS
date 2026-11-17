# Copilot instructions for OS_ONE_NEW_OS

## Project shape

This repository builds a freestanding x86-64 bare-metal AI unikernel. GRUB
loads `build/kernel.elf` plus the binary model/sample modules into memory;
`boot/` enters 64-bit long mode, and `src/kernel.c` runs the eight-stage boot
and inference flow. The kernel has no libc, filesystem, scheduler, syscalls,
or general-purpose OS beneath it.

The inference engine is deliberately shared between the unikernel and the
native host baseline. `src/ai.c` owns the portable dispatch policy and
correctness gate, while `src/kernels_scalar.c`, `src/kernels_avx2.c`, and
`src/kernels_fma.c` provide mathematically equivalent implementations.
`src/cpuid.c` enables and reports AVX state in the kernel; `src/host_caps.c`
provides the host equivalent. Do not move ISA-specific instructions into the
portable engine/registry translation unit: the Makefile compiles each SIMD
translation unit with its own flags and only executes it after eligibility and
scalar-equivalence checks.

There are two product workflows:

- **AI OS mode** (`make run` / `make ai-os`) boots the interactive ring-0
  serial shell and supports commands such as `status`, `predict N`, `ask TEXT`,
  and `shutdown`.
- **Dashboard mode** (`make bench` / `make dashboard-mode`) captures a
  non-interactive QEMU run, runs the matched native baseline, and uses
  `tools/make_report.py` to generate `dashboard/data/results.json` and the
  checked-in `results.js`. The dashboard is static and is not an in-kernel UI.

`tools/run_qemu.py` validates the serial transcript and translates the QEMU
debug-exit status. `tools/make_report.py` is the single report assembly path;
generated dashboard data should not be hand-edited. `probe/` is a separate
capability/performance probe used by `make probe` and `make probe-max`.

## Build, test, and run commands

### Linux or the toolchain container

Install the dependencies listed in `README.md`/CI (`gcc`, binutils, `make`,
GRUB tools, `xorriso`, QEMU, and Python 3), then use:

```sh
make all              # kernel ELF, bootable ISO, and native host binary
make test             # C artifact/dispatch tests plus Python unittest suite
make bench            # boot, benchmark, compare host, regenerate dashboard data
make run              # interactive QEMU serial console
make ai-os            # alias for make run
make dashboard-mode   # alias for make bench
make dispatch-csv     # forced scalar/AVX2/FMA host factorial cells
make probe            # default QEMU CPU capability probe
make probe-max        # probe with QEMU -cpu max
make download-model   # download the stories260K checkpoint and tokenizer
make convert-model    # validate/package them into build/ LLM modules
make validate-model   # verify package headers and checksums
make llm-iso          # build the explicit TinyStories GRUB image
make serve            # benchmark if needed, then serve dashboard on port 8000
make clean            # remove build/
```

The CI sequence is the authoritative broad check: `make test`, `make bench`,
`make dispatch-csv`, and `make probe`. `make bench` expects QEMU to complete
the benchmark and writes diagnostics to `build/unikernel.log`,
`build/host.log`, and `build/host.json`.

For a focused test, build the test binaries and artifacts first with
`make test`, then run either C test directly:

```sh
build/test-ai build/model.bin build/samples.bin
build/test-dispatch build/model.bin build/samples.bin
```

Or run one Python unittest by fully qualified name, for example:

```sh
python3 -m unittest tests.test_report.BenchmarkReportTests.test_dispatch_view_prefers_fastest_measured_kernel
```

On Windows, Docker Desktop is the supported build environment. Run
`.\scripts\build.ps1` once to build `ai-os-lab:toolchain` and `make all`, then
use `.\scripts\run.ps1`, `.\scripts\benchmark.ps1`, or
`.\scripts\dashboard-mode.ps1`. These scripts mount the repository at `/work`
and run the corresponding Make target as a non-root container user.

## Inputs and generated artifacts

`build/model.bin` and `build/samples.bin` are created automatically as
deterministic synthetic fixtures when absent. They are smoke-test data and
must not be described as MNIST accuracy. `make demo` explicitly recreates
those fixtures. Real MNIST artifacts require host Python dependencies from
`requirements-train.txt` and are produced by:

```sh
python3 tools/train_mnist.py --output-dir build
# or: make train
```

`make download-model` downloads the TinyStories model artifacts into
`models/`; this is separate from the current MNIST MLP path.

`make convert-model` emits `build/llm-model.bin`, `build/tokenizer.bin`, and
`build/llm-manifest.json`. `make llm-iso` is separate from the normal ISO and
loads those modules under the stable GRUB names `llm-model` and `tokenizer`.

The model and sample binary formats are documented in `README.md` and
`docs/quickstart.md`. Preserve their little-endian headers, exact dimensions,
and validation rules when changing exporters or kernel consumers. The ISO
rules copy the artifacts into `build/iso-root/boot/` or
`build/iso-dashboard-root/boot/` before invoking `grub-mkrescue`.

## Conventions that matter here

- Keep kernel code freestanding and compatible with the existing strict
  `-Wall -Wextra -Werror` builds. Avoid libc/POSIX assumptions in `src/` and
  boot code.
- Keep the host and unikernel implementations behaviorally aligned: they
  must consume the same model/sample formats, use the same dispatch policy,
  and report equivalent predictions. Host-only code belongs in
  `src/host_bench.c`/`src/host_caps.c`; freestanding CPU and port-I/O logic
  belongs in the kernel-side files.
- Preserve per-translation-unit ISA isolation. The engine must remain
  portable; AVX2/FMA flags belong on the corresponding kernel source object
  only. The runtime order is CPUID eligibility, CR4/XCR0 AVX-state enablement,
  scalar correctness gate, timing, then selection.
- Treat the scalar kernel as the reference implementation. SIMD candidates
  must agree on argmax and remain within the shared floating-point error gate;
  never select a SIMD path from CPUID capability alone.
- Keep the machine-readable `@@BENCH_BEGIN`/`@@BENCH_END` transcript contract
  stable. Changes to benchmark output must update the parser/validation in
  `tools/make_report.py` and its tests together.
- Benchmark output is empirical: QEMU TCG and the native host are different
  environments, so latency ratios are not scheduler speedup claims. Preserve
  dispatch evidence, per-kernel timings, eligibility, and correctness results
  when changing reporting.
- Generated dashboard files (`dashboard/data/results.json` and
  `dashboard/data/results.js`) come from `make bench`; regenerate them through
  the report pipeline rather than editing values manually.
- The kernel uses identity-mapped memory only in the first 1 GiB, 2 MiB pages,
  a bitmap-backed physical frame allocator, a bump heap, COM1 serial output,
  disabled hardware interrupts, and PIT channel-2 calibration for TSC timing.
  Changes to boot order, AVX enablement, memory setup, or timing must preserve
  the stage transcript and explicit panic diagnostics.
- The LLM path uses a dedicated approximately 2 MiB arena for tokenizer
  tables, activations, KV cache, and logits; model/tokenizer bytes remain
  zero-copy GRUB modules. The framebuffer UI is optional at boot and falls
  back to serial when QEMU does not provide a usable 32-bit linear framebuffer.

## Relevant documentation

Use `README.md` for project scope, formats, and the main workflow;
`docs/quickstart.md` for target outputs and troubleshooting; and
`docs/lab-guide.md` for experiments and measurement methodology. The CI
workflow in `.github/workflows/ci.yml` shows the expected clean Linux
validation sequence.
