# Bare-Metal AI Inference Unikernel

A single-address-space x86-64 unikernel that boots through GRUB/Multiboot2 into
64-bit long mode and runs a 784-128-10 neural network entirely in ring 0.
There is no scheduler, no syscall interface, no libc, no filesystem, and no
general-purpose kernel underneath it: the weights arrive as a GRUB module, the
kernel casts them to `float` pointers, and inference starts within milliseconds
of power-on.

The kernel ships three mathematically equivalent forward passes — portable
scalar, AVX2, and AVX2+FMA — and an **adaptive execution engine** that probes
CPUID, enables the AVX state machine itself, times every eligible kernel at
boot, gates every SIMD path against the scalar reference, and selects by
measurement. The same engine is compiled as an ordinary Linux program, so the
dispatch policy and the two environments can be compared with identical code,
identical weights, and identical inputs. Those runs are compiled into a
static measurement report in `dashboard/`.

## Why the dispatch is measured, not assumed

The capability-only policy ("if CPUID says AVX2, use AVX2") was tested
empirically and **rejected**. The probe kernel (`probe/`) measured, on this
project's own benchmark environment:

| Environment | scalar | AVX2 | AVX2+FMA | correct policy |
| --- | --- | --- | --- | --- |
| QEMU TCG (default) | baseline | not advertised | not advertised | scalar |
| QEMU TCG (`-cpu max`) | baseline | 3.15× slower | 19.7× slower | scalar (despite CPUID advertising AVX2+FMA) |
| Native CPU (Intel Core 5 210H) | baseline | 1.6× faster | 1.2× faster | AVX2 |

A CPUID-based dispatcher would have made the unikernel **~3× slower on its
own benchmark** and meaningfully slower than necessary on silicon. The measured
dispatcher selects `scalar` under TCG and `avx2` on this silicon, and its full
evidence — per-kernel cycles, eligibility, correctness-gate results — is
printed at boot and carried into the report.

## What is actually measured

| Quantity | How it is obtained |
| --- | --- |
| Accuracy | The frozen `samples.bin` set is scored inside the kernel; the native run must score the same inputs and agree on every prediction. Trained MNIST data is optional; clean builds use labeled synthetic smoke fixtures |
| Per-pass latency | fenced `RDTSC` around every forward pass, converted to nanoseconds through a PIT channel-2 calibration (50 ms window) |
| Tails | nearest-rank p50 / p90 / p99 over 10,000 individually timed passes |
| Distribution | 16-bucket histogram emitted by the kernel itself |
| Boot cost | cycles from kernel entry through completion of the first forward pass |
| Dispatch evidence | per-kernel boot timing, CPUID snapshot, eligibility masks, and the scalar-reference correctness gate (`diff ≤ 500e-6`) for every candidate kernel |
| Footprint | linked kernel image, weights, and sample module sizes; allocator capacity and host process RSS are reported separately |

The kernel and the native binary both print a machine-readable
`@@BENCH_BEGIN ... @@BENCH_END` block. `tools/make_report.py` merges those
blocks with the training report and emits `dashboard/data/results.json` plus
`results.js`.

## Quick start

### Windows (Docker Desktop, no cross-compiler needed)

```powershell
.\scripts\build.ps1        # build the toolchain image and the ISO
.\scripts\benchmark.ps1    # boot in QEMU, run both benchmarks, build the report
.\scripts\run.ps1          # boot interactively and watch the serial console
```

Then open `dashboard/index.html` directly — it loads `data/results.js`, so no
web server is required.

### Debian / Ubuntu

```sh
sudo apt install gcc binutils make grub-pc-bin grub-common xorriso \
    qemu-system-x86 python3
make test       # native unit tests against the exported artifacts
make bench      # ISO + QEMU boot + native baseline + dashboard report
make serve      # optional: http://localhost:8000 for the same dashboard
make run        # interactive boot with serial on stdio
```

### Demo and MNIST training

A clean build works without PyTorch: missing model/sample artifacts are filled
with deterministic, explicitly labeled synthetic smoke fixtures. Those fixtures
exercise boot and inference plumbing and **must not be interpreted as MNIST
accuracy**. `make demo` deliberately replaces the artifacts with these fixtures.

To produce real MNIST artifacts, train on a machine with PyTorch:

```powershell
python -m pip install -r requirements-train.txt
python tools/train_mnist.py --output-dir build
```

The trainer needs a Python with PyTorch, so it runs on the host, not inside the
QEMU container. On Linux with those packages installed, `make train` runs the
same command. Training does not need to be repeated for a benchmark run.

`make train` writes three artifacts into `build/`:

- `model.bin` — 407,100 B flat weight file consumed by the kernel
- `samples.bin` — 500 real MNIST test images, frozen so both environments
  classify exactly the same inputs (16-byte header + 500 × 3,140 B)
- `training.json` — epochs, hyperparameters, confusion matrix, per-class
  accuracy, weights statistics, and preview digit bitmaps

Training is deterministic for the configured seed (`7`). The generated report
records the accuracy from that run; pre-generated dashboard data is a captured
benchmark and is not a guarantee for future training runs.

## Repository layout

| Path | Purpose |
| --- | --- |
| `boot/boot.S` | Multiboot2 header, 32-bit entry, page tables, long-mode jump |
| `boot/isr.S` | 256-entry interrupt stub table |
| `src/kernel.c` | Eight-stage boot transcript, module discovery, dispatch stage, inference driver |
| `src/memory.c` | Multiboot2 memory map, frame bitmap, bump heap, 2 MiB mapping |
| `src/interrupts.c` | IDT installation and CPU exception reporting |
| `src/timer.c` | `RDTSC`, PIT channel-2 calibration, QEMU debug exit |
| `src/serial.c` | COM1 driver, aligned status output, printf-style formatting |
| `src/ai.c` | Dispatch engine: kernel registry, measured policy, correctness gate (shared with the host build) |
| `src/kernels_scalar.c` | Portable reference forward pass (the gate's ground truth) |
| `src/kernels_avx2.c` | AVX2 forward pass, compiled with `-mavx2` |
| `src/kernels_fma.c` | AVX2+FMA forward pass, compiled with `-mavx2 -mfma` |
| `src/cpuid.c` | Unikernel CPUID snapshot + AVX state enablement (CR4.OSXSAVE, XCR0) |
| `src/host_caps.c` | Host-side CPUID snapshot for the native build |
| `src/bench.c` | Timing loop, percentiles, histogram, result block with dispatch evidence |
| `src/host_bench.c` | Native baseline: `--kernel` forcing, dispatch policy, CSV rows for factorial cells |
| `tests/test_ai.c` | Host tests: validation, determinism, artifact sanity |
| `tests/test_dispatch.c` | Host tests: every eligible kernel agrees with the scalar reference |
| `tools/train_mnist.py` | PyTorch training and artifact export |
| `tools/generate_demo_model.py`, `tools/generate_demo_samples.py` | Deterministic fixtures for smoke builds without training dependencies |
| `tools/run_qemu.py` | QEMU launcher: exit-status translation, timeout, transcript validation, extra QEMU args |
| `tools/make_report.py` | Merges transcripts into the dashboard dataset, including the dispatch record |
| `probe/` | Capability/speed probe kernel used to gather the dispatch-policy evidence |
| `dashboard/` | Static measurement report (no build step, no CDN) |
| `Dockerfile.toolchain` | Debian image with GCC, GRUB, xorriso, QEMU, Python |

## Model and sample formats

All fields are little-endian. `model.bin` starts with five `uint32` values —
magic `0x41494D4C`, version `1`, input `784`, hidden `128`, output `10` —
followed by `float32` arrays in this order: hidden weights `[128][784]`,
hidden biases `[128]`, output weights `[10][128]`, output biases `[10]`.
Total size 407,100 bytes.

`samples.bin` starts with four `uint32` fields: magic `0x504D4153`, version,
count, and reserved (zero). It is followed by `count` records of `uint32 label`
and 784 `float32` intensities. The kernel validates the header, labels, exact
file size, and pixel ranges before inference, and reports malformed inputs
explicitly.

## Scope and limits

Mapped memory is the first 1 GiB, using 2 MiB identity pages; the frame
allocator intentionally manages only that range. There are no hardware
interrupts, no APIC, no disk, and no GPU. The target is QEMU, not physical
hardware.

Two limits matter when reading the report:

1. **The guest CPU is emulated.** QEMU runs the kernel under TCG, while the
   native baseline runs on silicon. The per-pass latency difference therefore
   describes these two environments; it is not a scheduler speedup claim.
2. **One workload, three kernels.** One model shape, one sample set, float32
   math across scalar/AVX2/FMA code paths. No quantisation or batching. The
   methodology is the transferable result.

## The factorial experiment

`make dispatch-csv` produces the Linux-side cells of the AEGIS-X factorial
design: it runs the native binary with each kernel forced (E1 scalar, E2
AVX2/FMA), appending one CSV row per cell to `build/dispatch.csv`. The
unikernel covers E4/E5 automatically, because the same dispatch policy runs
in-kernel and reports its measurements in the boot transcript. Comparing the
cells separates the environment effect from the kernel effect — the question
a simple "unikernel vs Linux" bar chart cannot answer.

To reproduce the probe evidence for the policy itself:

```sh
make probe       # default TCG CPU: no AVX advertised, scalar-only
make probe-max   # -cpu max: AVX2/FMA advertised and executing, measured slower
```

See [docs/quickstart.md](docs/quickstart.md) for the 30-second "what does
this do, where is the input, what comes out" summary plus every run command.

See [docs/lab-guide.md](docs/lab-guide.md) for experiments, the measurement
method, and troubleshooting.
