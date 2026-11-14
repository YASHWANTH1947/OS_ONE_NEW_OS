# Lab guide

## Boot and inspect

```powershell
.\scripts\build.ps1
.\scripts\run.ps1
```

GRUB enters the Multiboot2 32-bit entry point, the stub builds 2 MiB identity
pages and switches to long mode, and the kernel prints an eight-stage status
board on COM1. Each stage must report `READY`:

1. Serial console (COM1, 115200 8N1)
2. IDT with 256 gates and 32 CPU exception vectors
3. Physical frame allocator over the Multiboot2 memory map
4. Paging: 1 GiB identity map with 2 MiB pages
5. GRUB modules: the MLP weight file and the frozen sample set
6. TSC calibration against PIT channel 2
7. CPU features: CPUID snapshot, including the AVX state the kernel enabled
8. Execution kernel, selected by measured dispatch

Stage 3 prints usable RAM and free frames, stage 4 prints the heap and mapped
range, stage 5 prints the model dimensions and module sizes, and stage 6 prints
the measured TSC frequency. A malformed or missing module panics with a specific
message instead of faulting later.

Between stages 7 and 8 the kernel prints the DISPATCH block: every eligible
kernel's boot-time cycles, the correctness-gate result for each SIMD candidate
(`diff` against the scalar reference, in 1e-6 units), and the selection. Under
the default QEMU CPU nothing but scalar is eligible; under `-cpu max` all
three are measured and scalar still wins on TCG — which is the point of
measuring instead of trusting CPUID.

## Measuring

```powershell
.\scripts\benchmark.ps1     # make bench inside the toolchain container
```

The recipe repackages the ISO, boots it with the serial port captured to
`build/unikernel.log`, runs the native baseline into `build/host.json`, and
regenerates `dashboard/data/results.json` and `results.js`.

The kernel times 10,000 passes after 32 warmup passes, with every pass
individually recorded. The host binary uses the same number of passes and the
same 500 samples. QEMU is started with `-device isa-debug-exit`, so the kernel
terminates the VM (port `0xF4`) as soon as the benchmark and footprint summary
are written — the run is unattended and repeatable.

### Reading the report

Open `dashboard/index.html` in a browser (no server needed). It shows:

- headline p50, p99, boot-to-first-pass, and sample accuracy;
- the boot stage list parsed from the transcript;
- the 16-bucket latency histogram, plus percentile bars for both environments;
- the adaptive-engine section: per-kernel boot timing and gate results in
  both environments, with the CPUID snapshot that made them eligible;
- the accuracy ring, the 10×10 confusion matrix, and six classified digits;
- training curves, hyperparameters, and SHA-256 hashes of every artifact;
- the verbatim serial transcript.

The raw numbers live in `dashboard/data/results.json`; every value there is
copied from a captured run, never typed by hand.

## Suggested experiments

1. Change `BENCH_PASSES` in `include/bench.h`, rerun `make bench`, and compare
   p50/p99 across runs. Explain why repeated p99 measurements vary more than
   p50 under emulation.
2. Raise the PIT calibration window in `src/timer.c` from 50 ms and check
   whether the reported TSC frequency stabilises.
3. Compare the kernel histogram against the native one. The shapes differ
   because one is TCG-emulated and the other runs on silicon — describe what
   that implies about using the ratio as a speedup claim.
4. Boot with `make probe` and `make probe-max` and compare the CPUID
   snapshots. Then boot the full kernel against `-cpu max`
   (`tools/run_qemu.py ... --qemu-args "-cpu max"`) and observe the
   dispatcher choosing scalar despite AVX2 being advertised. This is the
   evidence for the measured-dispatch policy; write down what a
   capability-only policy would have cost in each environment.
5. Run `make dispatch-csv` and read `build/dispatch.csv`: the same
   engine with each kernel forced on silicon. Compare the unikernel's
   in-kernel dispatch measurements (cycles, same policy) with the host
   ranking, and explain the flip from scalar-under-TCG to
   AVX2-on-silicon.
6. Retrain with a different hidden width (`tools/train_mnist.py`, then update
   `AI_HIDDEN` in `include/ai.h` and the model header check). Re-measure the
   effect on per-pass time and accuracy.
7. Allocate and release a PMM frame in a temporary kernel test and verify the
   free-frame count in the footprint section changes by one in each direction.
8. Trigger a harmless exception in a scratch build to inspect the IDT
   diagnostic, then remove the trigger.

## Troubleshooting

- **`grub-mkrescue` missing or `xorriso` not found**: rebuild the image with
  `.\scripts\build.ps1`; the container installs GRUB and xorriso.
- **No serial output**: `make run` attaches COM1 to stdio; `make bench` writes
  `build/unikernel.log` instead. Check the log file before assuming a hang.
- **Triple fault before the banner**: inspect the Multiboot2 header and the
  long-mode transition. The stub expects GRUB's 32-bit protected-mode entry
  contract and a CPU with long mode and SSE2.
- **Exceptions during dispatch under `-cpu max`**: the kernel enables
  CR4.OSXSAVE and XCR0 itself before any YMM instruction; if a scratch build
  reordered that sequence, the IDT diagnostic will point at vector 6/13 inside
  the AVX2 kernel. The ordering — CPUID gates, CR4, XSETBV, then any VEX
  instruction — is load-bearing (verified by the probe).
- **`report ... has no @@BENCH_BEGIN block`**: the kernel did not finish. Read
  the captured transcript; a panic or an early reset is printed there.
- **`ModuleNotFoundError: torch`**: only the training step needs PyTorch. Run
  `make train` on a machine with `requirements-train.txt` installed; the
  toolchain image deliberately omits it.
- **Model-format panic**: the artifacts must come from the same exporter. Rerun
  `make train` and rebuild the ISO.

The IDT installs CPU exception gates only; hardware interrupts stay disabled and
the guest TSC is the timing source. Everything is calibrated against PIT
channel 2 rather than an APIC timer.
