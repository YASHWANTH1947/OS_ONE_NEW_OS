# Quickstart — run, operate, and read the output

Everything runs inside the Docker toolchain; the only host requirements are
Docker Desktop (Windows) or Docker Engine (Linux) and, optionally, Python +
PyTorch on the host for training.

## 0. What this system does (30-second version)

The project is a **benchmark machine for AI inference**, not an interactive
app. Nobody types anything at runtime. Two binary inputs are baked into the
bootable ISO *before* the machine starts:

- `build/model.bin` — a trained 784-128-10 neural network (MNIST digit
  recognizer): the *brain*.
- `build/samples.bin` — 500 labeled handwritten digit images (each 28×28
  pixels flattened to 784 floats in [0,1], with its true label): the *exam
  paper + answer key*.

At boot, GRUB loads the kernel and both modules into RAM. The kernel then:

1. Probes the CPU and picks its fastest math kernel by measurement.
2. Runs each of the 500 images through the network: pixels → ×W1+bias → ReLU
   → ×W2+bias → 10 scores → highest score = predicted digit. Compares each
   prediction with the true label.
3. Repeats one forward pass 10,000 times under a cycle-accurate timer.
4. Prints everything (accuracy, latency percentiles, throughput, dispatch
   evidence) to the serial console as both a human-readable board and a
   machine-readable `@@BENCH` block, then opens a small ring-0 command shell.
   The shell shuts the VM down automatically after 15 seconds of inactivity.

```
YOU (before boot)                THE UNIKERNEL (runtime, no user input)      OUTPUT (~25 s)
─────────────────                ──────────────────────────────────────    ─────────────
train/pick model ──► model.bin ┐  boot → load brain + exam into RAM          accuracy 98.20%
pick 500 images ───► samples.bin├► 500 images  →  500 predicted digits        p50 / p99 latency
                        (baked into the ISO)     10,000 timed passes          chosen kernel
                                 kernel measures itself                       dashboard report
```

The latency comparison against a Linux host running the identical code is the
research result; the per-kernel factorial (`make dispatch-csv`) is what makes
it scientific rather than anecdotal.

## 1. One-time setup

```powershell
cd "C:\Users\ACER\Downloads\oss prject"
scripts\build.ps1        # builds the toolchain image + kernel + ISO
```

If PowerShell blocks the script (unsigned), run instead:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

Linux/macOS equivalent:

```sh
docker build -f Dockerfile.toolchain -t ai-os-lab:toolchain .
docker run --rm --user 1000:1000 -e HOME=/tmp \
    --mount type=bind,source="$(pwd)",target=/work \
    --workdir /work ai-os-lab:toolchain make all
```

All later commands use the same `docker run ... make <target>` wrapper. For
readability the commands below are written as the `make` target only.

| Target | Input | Output | What it does |
| --- | --- | --- | --- |
| `make all` | sources | `build/kernel.elf`, `build/ai-os-lab.iso`, `build/host-bench` | Compile the unikernel, link it, package the bootable ISO, build the native baseline |
| `make run` | `build/ai-os-lab.iso` | live serial console in your terminal | **Interactive boot** — watch all 8 stages, dispatch, inference, results, then the VM exits itself |
| `make bench` | ISO + weights | `build/unikernel.log`, `build/host.json`, `dashboard/data/results.json` + `results.js` | Full measured pipeline: boot → benchmark → native baseline → regenerate the dashboard |
| `make dispatch-csv` | ISO artifacts | `build/dispatch.csv` | Factorial experiment: native run with each kernel forced (scalar / avx2 / fma), one CSV row each |
| `make probe` | probe sources | `build/probe/probe.log` | Capability probe on the default QEMU CPU (expect: no AVX) |
| `make probe-max` | probe sources | `build/probe/probe.log` | Same probe with `-cpu max` (expect: AVX2/FMA advertised, measured ~3x slower than scalar under TCG) |
| `make serve` | dashboard files | http://localhost:8000 | Runs `make bench` first if needed, then serves the report |
| `make test` | build artifacts | test output in terminal | All correctness gates: model/sample validation, kernel equivalence, report tooling |
| `make train` | `data/MNIST/raw/` (host only) | `build/model.bin`, `build/samples.bin`, `build/training.json` | Trains the real MNIST MLP (needs host Python + PyTorch; deterministic seed 7) |
| `make demo` | — | same three artifacts, synthetic | Replaces artifacts with labeled smoke fixtures (no PyTorch needed) |
| `make clean` | — | — | Deletes `build/` entirely |

## 2. Interactive boot — what you will see (make run)

```
  [1/8] Serial console (COM1, 115200 8N1) ............................ READY
  [2/8] IDT: 32 CPU exception gates (vectors 0-31) ................... READY
  [3/8] Physical frame allocator (4 KiB bitmap) ...................... READY
        usable RAM 260992 KiB | free frames 64694
  [4/8] Paging: 1 GiB identity map, 2 MiB pages ...................... READY
        heap 65536 B (bump allocator) | mapped 1024 MiB
  [5/8] GRUB modules (initrd, no filesystem) ......................... READY
        mlp 784 -> 128 -> 10 | model 407100 B | samples 500 x 3140 B
  [6/8] TSC calibration (PIT channel 2, 50 ms) ....................... READY
        2689949540 cycles/s | 0.371 ns per cycle
  [7/8] CPU features (CPUID snapshot) ................................ READY
        QEMU Virtual CPU version 2.5+
        avx=0 avx2=0 fma=0 avx_state=0
  DISPATCH (measured, not assumed)
  selected ......... scalar
  eligible ......... 0x1
  scalar .......... 2241500 cycles
  [8/8] Execution kernel selected by measurement ..................... READY

  INFERENCE
  Accuracy .......................... 491 / 500  (98.20%)
  p50 latency    1176.66 us   p99 2822.19 us
  Throughput ........................ 780 inferences/s
```

After the result block, `make run` accepts commands over the serial console:
`help`, `status`, `run`, `clear`, and `shutdown` (also `exit`). The kernel
automatically closes QEMU after 15 idle seconds, so unattended runs still
finish without manual input.

## 3. Changing the input

The kernel consumes two binary inputs bundled into the ISO by GRUB:

- `build/model.bin` — 407,100 B: 5×u32 header (magic, version, 784, 128, 10)
  then w1[128][784], b1[128], w2[10][128], b2[10] as float32, little-endian.
- `build/samples.bin` — 16 B header (magic, version, count, reserved) then
  `count` records of (u32 label + 784 float32 pixels in [0,1]).

Swap in your own weights by writing files in this format to `build/` and
rerunning `make all` (the ISO rule picks them up) or `make bench`. Malformed
files are rejected with a specific panic message at stage 5 — never a mystery
triple fault.

## 4. Reading the report (make bench → dashboard/index.html)

Open `dashboard/index.html` directly in a browser (double-click works; the
page loads `data/results.js`, so no server needed). Sections:

1. **Boot pipeline** — the 8 stages parsed from the real captured transcript
2. **Latency distribution** — 16-bucket histogram the kernel itself emitted,
   plus percentile bars for both environments
3. **Adaptive execution engine** — per-kernel boot timing and correctness-gate
   results in both the unikernel and the native run
4. **Classification results** — accuracy ring, 10×10 confusion matrix,
   classified digit previews
5. **Training and artifacts** — loss/accuracy curves, hyperparameters,
   SHA-256 hashes of every artifact
6. **Method and honest caveats** — including why the latency ratio is not a
   speedup claim, and the verbatim serial transcript

The raw data behind every number: `dashboard/data/results.json`.

## 5. The three experiment commands worth running

```sh
make bench          # environment comparison (unikernel vs native)
make dispatch-csv   # kernel comparison on silicon (forced scalar/avx2/fma)
make probe-max      # why measured dispatch beats CPUID guessing
```

`build/dispatch.csv` after `make dispatch-csv` (real example, 10,000 passes
per cell, 500 samples):

```
variant,kernel,forced,passes,samples,correct,...,throughput_per_second,...
linux-host,scalar,1,10000,500,491,...,47358,...
linux-host,avx2,  1,10000,500,491,...,85970,...
linux-host,fma,   1,10000,500,491,...,93868,...
```

Same accuracy in every cell (prediction-level equivalence) — the only
difference is speed, which is the point of the factorial design.

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `scripts\*.ps1` blocked | `powershell -ExecutionPolicy Bypass -File scripts\...` |
| `docker ... Cannot connect to the Docker daemon` | Start Docker Desktop, wait for the engine, retry |
| `grub-mkrescue: not found` | Rerun `scripts\build.ps1` (the container image includes GRUB) |
| `No module named 'torch'` | Only training needs PyTorch; run it on the host: `python tools\train_mnist.py --output-dir build` |
| Report shows ~10% accuracy | You are on demo fixtures; rerun `make train` (host) then `make bench` |
| QEMU seems to hang | First TCG execution of GRUB takes a few seconds; `make bench` writes `build/unikernel.log` — check that file |
