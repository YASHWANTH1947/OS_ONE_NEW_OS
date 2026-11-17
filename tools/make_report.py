#!/usr/bin/env python3
"""Turn a completed benchmark run into the dashboard dataset.

Inputs are the artifacts a run actually produced:

  build/unikernel.log  QEMU serial transcript, including the @@BENCH block
  build/host.json      native Linux baseline written by host_bench
  build/training.json  optional MNIST training + evaluation report
  build/kernel.elf, build/model.bin, build/ai-os-lab.iso   measured binaries

Nothing here invents numbers: every value comes from one of those files, and
the run fails loudly if the unikernel never finished its benchmark.

Run:  python tools/make_report.py --unikernel-log build/unikernel.log \
          --host-json build/host.json [--training build/training.json] \
          --out-dir dashboard/data
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

BEGIN = "@@BENCH_BEGIN"
END = "@@BENCH_END"

REQUIRED_UNIKERNEL = (
    "passes",
    "samples",
    "correct",
    "min_ns",
    "p50_ns",
    "p90_ns",
    "p99_ns",
    "max_ns",
    "mean_ns",
    "model_bytes",
    "samples_bytes",
    "hist_ns",
    "dispatch_kernel",
)

DISPATCH_FIELDS = (
    "dispatch_eligible_mask",
    "dispatch_measured_mask",
    "dispatch_banned_mask",
    "cpuid_brand",
    "cpuid_avx",
    "cpuid_avx2",
    "cpuid_fma",
    "cpuid_avx_state",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_block(lines: list[str], origin: str) -> list[str]:
    start = None
    for index, line in enumerate(lines):
        if BEGIN in line:
            start = index
            break
    if start is None:
        raise SystemExit(
            f"error: {origin} has no {BEGIN} block; the benchmark did not run "
            "to completion"
        )
    for index in range(start + 1, len(lines)):
        if END in lines[index]:
            return lines[start + 1 : index]
    raise SystemExit(f"error: {origin} has an unterminated {BEGIN} block")


def parse_pairs(lines: list[str]) -> dict:
    values: dict = {}
    for line in lines:
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        key = key.strip()
        raw = raw.strip()
        if key == "hist_ns":
            values[key] = [int(item) for item in raw.split(",") if item.strip()]
            continue
        try:
            values[key] = int(raw)
        except ValueError:
            values[key] = raw
    return values


def validate_variant(raw: dict, origin: str) -> None:
    if not isinstance(raw, dict):
        raise SystemExit(f"error: {origin} benchmark data must be an object")

    required = (
        "passes",
        "samples",
        "correct",
        "min_ns",
        "p50_ns",
        "p90_ns",
        "p99_ns",
        "max_ns",
        "mean_ns",
        "model_bytes",
        "samples_bytes",
        "hist_ns",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise SystemExit(f"error: {origin} benchmark data is missing {missing}")
    try:
        passes = int(raw["passes"])
        samples = int(raw["samples"])
        correct = int(raw["correct"])
        percentiles = [
            int(raw[key]) for key in
            ("min_ns", "p50_ns", "p90_ns", "p99_ns", "max_ns")
        ]
        histogram = raw["hist_ns"]
        counts = [int(value) for value in histogram]
        model_bytes = int(raw["model_bytes"])
        samples_bytes = int(raw["samples_bytes"])
    except (TypeError, ValueError):
        raise SystemExit(f"error: {origin} benchmark data has invalid values")

    if not isinstance(histogram, list):
        raise SystemExit(f"error: {origin} latency histogram must be a list")
    if passes < 1 or samples < 1 or correct < 0 or correct > samples:
        raise SystemExit(f"error: {origin} benchmark counts are out of range")
    if any(value < 0 for value in percentiles):
        raise SystemExit(f"error: {origin} latency values cannot be negative")
    if any(left > right for left, right in zip(percentiles, percentiles[1:])):
        raise SystemExit(f"error: {origin} latency percentiles are out of order")
    if model_bytes < 1 or samples_bytes < 1:
        raise SystemExit(f"error: {origin} artifact sizes must be positive")
    if len(counts) != 16 or sum(counts) != passes or any(value < 0 for value in counts):
        raise SystemExit(
            f"error: {origin} latency histogram must have 16 buckets "
            f"summing to {passes}"
        )


def variant_view(raw: dict) -> dict:
    samples = int(raw.get("samples", 0))
    correct = int(raw.get("correct", 0))
    p50 = int(raw.get("p50_ns", 0))
    p99 = int(raw.get("p99_ns", 0))
    histogram = [int(value) for value in raw.get("hist_ns", [])]
    view = {
        "variant": raw.get("variant", "unknown"),
        "engine": raw.get("engine", ""),
        "kernel": raw.get("dispatch_kernel", raw.get("kernel", "unknown")),
        "passes": int(raw.get("passes", 0)),
        "samples": samples,
        "correct": correct,
        "accuracy": round(correct / samples, 5) if samples else 0.0,
        "min_ns": int(raw.get("min_ns", 0)),
        "p50_ns": p50,
        "p90_ns": int(raw.get("p90_ns", 0)),
        "p99_ns": p99,
        "max_ns": int(raw.get("max_ns", 0)),
        "mean_ns": int(raw.get("mean_ns", 0)),
        "throughput_per_second": int(raw.get("throughput_per_second", 0)),
        "boot_to_inference_ns": int(raw.get("boot_to_inference_ns", 0)),
        "kernel_bytes": int(raw.get("kernel_bytes", 0)),
        "model_bytes": int(raw.get("model_bytes", 0)),
        "samples_bytes": int(raw.get("samples_bytes", 0)),
        "process_rss_bytes": int(raw.get("process_rss_bytes", 0)),
        "hist_ns": histogram,
        "hist_min_ns": int(raw.get("hist_min_ns", 0)),
        "hist_max_ns": int(raw.get("hist_max_ns", 0)),
        "jitter_ratio": round(p99 / p50, 3) if p50 else 0.0,
        "spread_ns": int(raw.get("max_ns", 0)) - int(raw.get("min_ns", 0)),
        "cycles_per_second": int(raw.get("cycles_per_second", 0)),
        "total_resident_bytes": (
            int(raw.get("process_rss_bytes", 0))
            if raw.get("variant") == "linux-host"
            else int(raw.get("kernel_bytes", 0))
            + int(raw.get("model_bytes", 0))
            + int(raw.get("samples_bytes", 0))
        ),
    }
    return view


def dispatch_view(raw: dict) -> dict:
    """Normalise the dispatch record emitted by both environments.

    Handles the unikernel's flat keys (dispatch_cycles_scalar=...) and the
    host JSON's nested object identically. Accepts a missing record (older
    logs) by reporting kernel='unknown' with no measurements.
    """
    raw = raw or {}
    nested = raw.get("dispatch", {}) if isinstance(raw.get("dispatch"), dict) else {}
    flat = {
        "selected": raw.get("dispatch_kernel", nested.get(
            "selected", raw.get("kernel", "unknown"))),
        "eligible_mask": int(raw.get("dispatch_eligible_mask",
                                     nested.get("eligible_mask", 0)) or 0),
        "measured_mask": int(raw.get("dispatch_measured_mask",
                                     nested.get("measured_mask", 0)) or 0),
        "banned_mask": int(raw.get("dispatch_banned_mask",
                                   nested.get("banned_mask", 0)) or 0),
    }
    caps = raw.get("caps", {}) if isinstance(raw.get("caps"), dict) else {}
    flat["caps"] = {
        "brand": str(caps.get("brand", raw.get("cpuid_brand", "")) or ""),
        "avx": int(caps.get("avx", raw.get("cpuid_avx", 0)) or 0),
        "avx2": int(caps.get("avx2", raw.get("cpuid_avx2", 0)) or 0),
        "fma": int(caps.get("fma", raw.get("cpuid_fma", 0)) or 0),
        "avx_state": int(caps.get("avx_state",
                                   raw.get("cpuid_avx_state", 0)) or 0),
    }
    kernels = ("scalar", "avx2", "fma")
    measured = {}
    for index, name in enumerate(kernels):
        cycles = raw.get(f"dispatch_cycles_{name}")
        if cycles is None and isinstance(nested.get("cycles"), dict):
            cycles = nested.get("cycles", {}).get(name)
        if cycles is None:
            continue
        entry = {"cycles": int(cycles)}
        diff = raw.get(f"dispatch_diff_{name}")
        if diff is None and isinstance(nested.get("diff"), dict):
            diff = nested.get("diff", {}).get(name)
        if diff is not None:
            entry["diff"] = int(diff)
        measured[name] = entry
    flat["measured"] = measured
    flat["correctness_gate"] = AI_GATE_NOTE
    return flat


AI_GATE_NOTE = (
    "A SIMD kernel may only be selected after reproducing the scalar "
    "reference within 500e-6 on the calibration input; banned kernels are "
    "reported and never used."
)


def histogram_buckets(view: dict) -> list[dict]:
    histogram = view["hist_ns"]
    count = len(histogram)
    if count == 0:
        return []
    low = view["hist_min_ns"]
    high = view["hist_max_ns"]
    width = max(1, -(-(high - low) // count))  # ceil division
    buckets = []
    for index, value in enumerate(histogram):
        bucket_low = low + index * width
        buckets.append(
            {
                "from_ns": bucket_low,
                "to_ns": bucket_low + width,
                "count": value,
            }
        )
    return buckets


def artifact(path: Path, label: str, note: str) -> dict:
    if not path.exists():
        return {
            "label": label,
            "path": str(path),
            "bytes": 0,
            "sha256": "",
            "note": note,
            "present": False,
        }
    digest = sha256(path)
    return {
        "label": label,
        "path": str(path).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "sha256_short": digest[:12],
        "note": note,
        "present": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unikernel-log", type=Path, required=True)
    parser.add_argument("--host-json", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--demo-model", action="store_true")
    parser.add_argument("--demo-samples", action="store_true")
    parser.add_argument("--host-log", type=Path, default=Path("build/host.log"))
    parser.add_argument("--kernel", type=Path, default=Path("build/kernel.elf"))
    parser.add_argument("--model", type=Path, default=Path("build/model.bin"))
    parser.add_argument("--samples", type=Path, default=Path("build/samples.bin"))
    parser.add_argument("--iso", type=Path, default=Path("build/ai-os-lab.iso"))
    parser.add_argument("--out-dir", type=Path, default=Path("dashboard/data"))
    args = parser.parse_args()

    for path in (args.unikernel_log, args.host_json):
        if not path.exists():
            raise SystemExit(f"error: missing required input {path}")

    serial_text = args.unikernel_log.read_text(errors="replace").replace("\r", "")
    serial_lines = [line.rstrip() for line in serial_text.splitlines()]
    unikernel_raw = parse_pairs(extract_block(serial_lines, "unikernel log"))
    missing = [key for key in REQUIRED_UNIKERNEL if key not in unikernel_raw]
    if missing:
        raise SystemExit(f"error: unikernel result block is missing {missing}")

    host_raw = json.loads(args.host_json.read_text(encoding="utf-8"))
    if args.training is not None:
        if not args.training.exists():
            raise SystemExit(f"error: missing training report {args.training}")
        if args.demo_model or args.demo_samples:
            raise SystemExit(
                "error: demo fixture flags cannot be combined with a training report"
            )
        training = json.loads(args.training.read_text(encoding="utf-8"))
    else:
        dataset = (
            "Synthetic smoke fixtures (not MNIST)"
            if args.demo_samples
            else "Benchmark samples (training metadata unavailable)"
        )
        training = {
            "dataset": dataset,
            "architecture": {
                "input": 784,
                "hidden": 128,
                "output": 10,
                "activation": "ReLU",
                "parameters": 101770,
            },
            "training": {
                "epochs": 0,
                "batch_size": 0,
                "learning_rate": None,
                "optimizer": "not trained",
                "seed": 20250308,
                "seconds": 0,
                "history": [],
            },
            "evaluation": {
                "test_accuracy": None,
                "sample_accuracy": None,
                "sample_count": 0,
                "confusion_matrix": [],
                "per_class_accuracy": [],
            },
            "weights": {},
            "previews": [],
        }

    validate_variant(unikernel_raw, "unikernel")
    validate_variant(host_raw, "host")
    if int(unikernel_raw["samples"]) != int(host_raw["samples"]):
        raise SystemExit("error: host and unikernel scored different sample counts")
    if int(unikernel_raw.get("model_bytes", 0)) != int(host_raw.get("model_bytes", 0)):
        raise SystemExit("error: host and unikernel used different model sizes")
    if int(unikernel_raw.get("samples_bytes", 0)) != int(host_raw.get("samples_bytes", 0)):
        raise SystemExit("error: host and unikernel used different sample files")

    unikernel = variant_view(unikernel_raw)
    host = variant_view(host_raw)
    unikernel_dispatch = dispatch_view(unikernel_raw)
    host_dispatch = dispatch_view(host_raw)

    host_log_lines: list[str] = []
    if args.host_log.exists():
        host_log_lines = [
            line.rstrip()
            for line in args.host_log.read_text(errors="replace").replace(
                "\r", ""
            ).splitlines()
        ]

    comparison = {
        "accuracy_match": abs(unikernel["accuracy"] - host["accuracy"]) < 1e-9,
        "latency_ratio_p50": round(host["p50_ns"] / unikernel["p50_ns"], 3)
        if unikernel["p50_ns"]
        else 0.0,
        "latency_ratio_p99": round(host["p99_ns"] / unikernel["p99_ns"], 3)
        if unikernel["p99_ns"]
        else 0.0,
        "comparable": False,
        "note": (
            "Both environments ran the same ai.c over the same frozen weights, "
            "but the guest CPU is emulated by QEMU while the baseline runs on "
            "silicon, so the latency ratio is a record of these two "
            "environments rather than a like-for-like OS comparison."
        ),
    }

    dataset_name = training.get("dataset", "Synthetic smoke fixtures (not MNIST)")
    is_mnist = (
        isinstance(dataset_name, str)
        and dataset_name.strip().lower().startswith("mnist")
    )
    is_synthetic = args.demo_samples or (
        isinstance(dataset_name, str)
        and dataset_name.strip().lower().startswith("synthetic")
    )
    model_note = (
        "trained MNIST weights"
        if is_mnist
        else "untrained demo weights"
        if args.demo_model
        else "weights (training metadata unavailable)"
    )
    samples_note = (
        "frozen MNIST sample module"
        if is_mnist
        else "synthetic smoke fixtures (not MNIST)"
        if is_synthetic
        else "benchmark samples (dataset metadata unavailable)"
    )
    dataset = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "dashboard",
        "runtime_mode": "ai-os",
        "scenario": {
            "name": "MNIST MLP forward pass"
            if is_mnist
            else "Synthetic MLP smoke test"
            if is_synthetic
            else "Inference benchmark (metadata unavailable)",
            "architecture": "784 -> 128 (ReLU) -> 10",
            "passes": unikernel["passes"],
            "samples": unikernel["samples"],
            "timer": "RDTSC calibrated against PIT channel 2"
            if unikernel["cycles_per_second"]
            else "RDTSC (uncalibrated)",
        },
        "unikernel": unikernel,
        "host": host,
        "host_environment": host_raw.get("host", {}),
        "comparison": comparison,
        "dispatch": {
            "note": (
                "Kernels are selected by measured timing after a CPUID "
                "eligibility check and a correctness gate, never by a "
                "capability guess. The same policy runs in both "
                "environments; on emulated TCG CPUs it picks scalar, on "
                "silicon it picks the fastest measured SIMD kernel."
            ),
            "gate": AI_GATE_NOTE,
            "unikernel": unikernel_dispatch,
            "host": host_dispatch,
        },
        "histogram": {
            "unikernel": histogram_buckets(unikernel),
            "host": histogram_buckets(host),
        },
        "campaign": {
            "measured_at": host_raw.get("measured_at", ""),
            "container": host_raw.get("host", {}).get("container", False),
        },
        "artifacts": [
            artifact(args.kernel, "kernel.elf", "linked unikernel image"),
            artifact(
                args.model,
                "model.bin",
                model_note,
            ),
            artifact(
                args.samples,
                "samples.bin",
                samples_note,
            ),
            artifact(args.iso, "ai-os-lab.iso", "GRUB Multiboot2 boot image"),
        ],
        "logs": {
            "unikernel": serial_lines,
            "host": host_log_lines,
        },
        "model": training,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(dataset, indent=2)
    (args.out_dir / "results.json").write_text(json_text, encoding="utf-8")
    # A plain script file keeps the dashboard usable by double-clicking
    # index.html, where fetch() against file:// is blocked by the browser.
    (args.out_dir / "results.js").write_text(
        "/* Generated by tools/make_report.py - do not edit. */\n"
        "window.BENCH_DATA = " + json_text + ";\n",
        encoding="utf-8",
    )

    print(
        "Report written: "
        f"{unikernel['correct']}/{unikernel['samples']} correct "
        f"({unikernel['accuracy'] * 100:.2f}%), "
        f"p50 {unikernel['p50_ns'] / 1000:.2f} us, "
        f"p99 {unikernel['p99_ns'] / 1000:.2f} us"
    )
    print(f"  {args.out_dir / 'results.json'}")
    print(f"  {args.out_dir / 'results.js'}")
    if not comparison["accuracy_match"]:
        print(
            "warning: unikernel and host accuracy differ; check that both used "
            "the same model.bin and samples.bin"
        )


if __name__ == "__main__":
    main()
