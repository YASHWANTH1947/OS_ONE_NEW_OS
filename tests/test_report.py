import json
import sys
import tempfile
import unittest
from pathlib import Path

from tools import make_report


def valid_result():
    return {
        "passes": 16,
        "samples": 10,
        "correct": 8,
        "min_ns": 10,
        "p50_ns": 20,
        "p90_ns": 30,
        "p99_ns": 40,
        "max_ns": 50,
        "mean_ns": 25,
        "model_bytes": 407100,
        "samples_bytes": 31416,
        "hist_ns": [1] * 16,
        "dispatch_kernel": "scalar",
        "dispatch_eligible_mask": 1,
        "dispatch_measured_mask": 1,
        "dispatch_banned_mask": 0,
        "cpuid_brand": "Test CPU",
        "cpuid_avx": 1,
        "cpuid_avx2": 1,
        "cpuid_fma": 1,
        "cpuid_avx_state": 1,
        "dispatch_cycles_scalar": 5000,
        "dispatch_cycles_avx2": 4000,
        "dispatch_diff_avx2": 3,
    }


class BenchmarkReportTests(unittest.TestCase):
    def test_accepts_complete_histogram_and_ordered_percentiles(self):
        make_report.validate_variant(valid_result(), "test")

    def test_rejects_histogram_that_does_not_cover_every_pass(self):
        result = valid_result()
        result["hist_ns"] = [0] * 16
        with self.assertRaises(SystemExit):
            make_report.validate_variant(result, "test")

    def test_rejects_out_of_order_percentiles(self):
        result = valid_result()
        result["p90_ns"] = 15
        with self.assertRaises(SystemExit):
            make_report.validate_variant(result, "test")

    def test_rejects_invalid_sample_counts(self):
        result = valid_result()
        result["correct"] = 11
        with self.assertRaises(SystemExit):
            make_report.validate_variant(result, "test")

    def test_histogram_buckets_preserve_counts(self):
        view = {
            "hist_ns": [2, 0, 3, 1],
            "hist_min_ns": 0,
            "hist_max_ns": 8,
        }
        buckets = make_report.histogram_buckets(view)
        self.assertEqual(sum(bucket["count"] for bucket in buckets), 6)
        self.assertEqual(buckets[0]["from_ns"], 0)
        self.assertGreaterEqual(buckets[-1]["to_ns"], 8)

    def test_report_labels_missing_training_as_synthetic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            serial = root / "unikernel.log"
            host_json = root / "host.json"
            kernel = root / "kernel.elf"
            model = root / "model.bin"
            samples = root / "samples.bin"
            iso = root / "boot.iso"
            output = root / "report"
            result = valid_result()
            result.update(
                passes=16,
                samples=10,
                correct=5,
                hist_ns=[1] * 16,
                variant="linux-host",
            )
            serial.write_text(
                "@@BENCH_BEGIN\n"
                "variant=unikernel\n"
                "dispatch_kernel=avx2\n"
                "dispatch_eligible_mask=3\n"
                "dispatch_measured_mask=3\n"
                "dispatch_banned_mask=0\n"
                "cpuid_brand=Test CPU\n"
                "cpuid_avx=1\n"
                "cpuid_avx2=1\n"
                "cpuid_fma=1\n"
                "cpuid_avx_state=1\n"
                "dispatch_cycles_scalar=5000\n"
                "dispatch_cycles_avx2=4000\n"
                "dispatch_diff_avx2=3\n"
                "passes=16\n"
                "samples=10\n"
                "correct=5\n"
                "min_ns=10\n"
                "p50_ns=20\n"
                "p90_ns=30\n"
                "p99_ns=40\n"
                "max_ns=50\n"
                "mean_ns=25\n"
                "model_bytes=407100\n"
                "samples_bytes=31416\n"
                "hist_ns=" + ",".join(["1"] * 16) + "\n"
                "@@BENCH_END\n",
                encoding="utf-8",
            )
            host_json.write_text(json.dumps(result), encoding="utf-8")
            for path in (kernel, model, samples, iso):
                path.write_bytes(b"fixture")

            arguments = [
                "--unikernel-log", str(serial),
                "--host-json", str(host_json),
                "--kernel", str(kernel),
                "--model", str(model),
                "--samples", str(samples),
                "--iso", str(iso),
                "--out-dir", str(output),
            ]
            old_argv = sys.argv
            try:
                sys.argv = [
                    "make_report.py",
                    *arguments,
                    "--demo-model",
                    "--demo-samples",
                ]
                make_report.main()
                demo_report = json.loads(
                    (output / "results.json").read_text(encoding="utf-8")
                )
                sys.argv = ["make_report.py", *arguments]
                make_report.main()
                unknown_report = json.loads(
                    (output / "results.json").read_text(encoding="utf-8")
                )
            finally:
                sys.argv = old_argv

        self.assertEqual(
            demo_report["scenario"]["name"], "Synthetic MLP smoke test"
        )
        self.assertIn("not MNIST", demo_report["model"]["dataset"])
        self.assertIsNone(demo_report["model"]["evaluation"]["test_accuracy"])
        self.assertEqual(
            demo_report["artifacts"][2]["path"], str(samples).replace("\\", "/")
        )
        self.assertIn("not MNIST", demo_report["artifacts"][2]["note"])
        self.assertEqual(
            demo_report["artifacts"][1]["note"], "untrained demo weights"
        )
        self.assertEqual(
            unknown_report["scenario"]["name"],
            "Inference benchmark (metadata unavailable)",
        )
        self.assertIn(
            "metadata unavailable", unknown_report["model"]["dataset"]
        )

    def test_dispatch_view_prefers_fastest_measured_kernel(self):
        raw = {
            "dispatch_kernel": "avx2",
            "dispatch_eligible_mask": 3,
            "dispatch_measured_mask": 3,
            "dispatch_banned_mask": 0,
            "cpuid_brand": "Test CPU",
            "cpuid_avx": 1,
            "cpuid_avx2": 1,
            "cpuid_fma": 0,
            "cpuid_avx_state": 1,
            "dispatch_cycles_scalar": 5000,
            "dispatch_cycles_avx2": 4000,
            "dispatch_diff_avx2": 3,
        }
        view = make_report.dispatch_view(raw)
        self.assertEqual(view["selected"], "avx2")
        self.assertEqual(view["measured"]["scalar"]["cycles"], 5000)
        self.assertEqual(view["measured"]["avx2"]["cycles"], 4000)
        self.assertEqual(view["measured"]["avx2"]["diff"], 3)
        self.assertEqual(view["caps"]["avx2"], 1)
        self.assertEqual(view["caps"]["fma"], 0)
        self.assertNotIn("fma", view["measured"])

    def test_dispatch_view_accepts_nested_host_json(self):
        view = make_report.dispatch_view({
            "kernel": "fma",
            "dispatch": {
                "selected": "fma",
                "eligible_mask": 7,
                "measured_mask": 7,
                "banned_mask": 0,
                "cycles": {"scalar": 100, "avx2": 90, "fma": 80},
                "diff": {"avx2": 2, "fma": 1},
            },
            "caps": {
                "brand": "Host CPU",
                "avx": 1,
                "avx2": 1,
                "fma": 1,
                "avx_state": 1,
            },
        })
        self.assertEqual(view["selected"], "fma")
        self.assertEqual(view["measured"]["fma"]["cycles"], 80)
        self.assertEqual(view["measured"]["avx2"]["diff"], 2)
        self.assertEqual(view["caps"]["brand"], "Host CPU")
        self.assertEqual(view["eligible_mask"], 7)


if __name__ == "__main__":
    unittest.main()
