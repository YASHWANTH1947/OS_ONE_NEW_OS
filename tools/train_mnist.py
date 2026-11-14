#!/usr/bin/env python3
"""Train the lab's 784-128-10 ReLU MLP and export kernel-ready artifacts.

Outputs (all little-endian):
  model.bin     the weight module consumed by the unikernel and host benchmark
  samples.bin   a fixed set of real MNIST test images + labels, so both
                environments score the identical inputs on real data
  training.json accuracy, confusion matrix and loss curve for the report

Run:  python tools/train_mnist.py --epochs 4
"""

import argparse
import json
import struct
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MAGIC = 0x41494D4C
VERSION = 1
INPUTS = 784
HIDDEN = 128
OUTPUTS = 10

SAMPLE_MAGIC = 0x504D4153  # "SAMP"
SAMPLE_VERSION = 1


class MnistMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(INPUTS, HIDDEN)
        self.output = nn.Linear(HIDDEN, OUTPUTS)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(image)))


def write_model(model: MnistMlp, path: Path) -> None:
    layers = (
        model.hidden.weight.detach().cpu().contiguous().view(-1).tolist(),
        model.hidden.bias.detach().cpu().tolist(),
        model.output.weight.detach().cpu().contiguous().view(-1).tolist(),
        model.output.bias.detach().cpu().tolist(),
    )
    with path.open("wb") as model_file:
        model_file.write(
            struct.pack("<5I", MAGIC, VERSION, INPUTS, HIDDEN, OUTPUTS)
        )
        for layer in layers:
            model_file.write(struct.pack(f"<{len(layer)}f", *layer))


def write_samples(images: torch.Tensor, labels: torch.Tensor, path: Path) -> None:
    """Write u32 magic/version/count/reserved, then `count` label+pixel records."""
    flat = images.contiguous().cpu().numpy()
    tags = labels.contiguous().cpu().numpy()
    count = int(flat.shape[0])
    with path.open("wb") as sample_file:
        sample_file.write(
            struct.pack("<4I", SAMPLE_MAGIC, SAMPLE_VERSION, count, 0)
        )
        for index in range(count):
            sample_file.write(struct.pack("<I", int(tags[index])))
            sample_file.write(struct.pack(f"<{INPUTS}f", *flat[index].tolist()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda image: image.view(INPUTS)),
        ]
    )
    training = datasets.MNIST(
        args.data_dir, train=True, download=True, transform=transform
    )
    testing = datasets.MNIST(
        args.data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(testing, batch_size=512)

    model = MnistMlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    history = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1

        model.eval()
        correct = 0
        total = 0
        confusion = [[0] * OUTPUTS for _ in range(OUTPUTS)]
        with torch.no_grad():
            for images, labels in test_loader:
                predictions = model(images).argmax(dim=1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.numel())
                for truth, guess in zip(labels.tolist(), predictions.tolist()):
                    confusion[truth][guess] += 1

        accuracy = correct / total
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(running_loss / max(1, batches), 5),
                "test_accuracy": round(accuracy, 5),
            }
        )
        print(
            f"epoch {epoch + 1}/{args.epochs}: "
            f"loss {running_loss / max(1, batches):.4f} "
            f"test accuracy {accuracy:.2%}",
            flush=True,
        )

    model.eval()

    # Freeze the first `samples` test images so the kernel and the host binary
    # classify exactly the same inputs in exactly the same order.
    sample_images = torch.stack([testing[i][0] for i in range(args.samples)])
    sample_labels = torch.tensor(
        [testing[i][1] for i in range(args.samples)], dtype=torch.int64
    )

    with torch.no_grad():
        logits = model(sample_images)
        sample_predictions = logits.argmax(dim=1)
    sample_accuracy = float(
        (sample_predictions == sample_labels).sum().item()
    ) / max(1, args.samples)

    weights = model.hidden.weight.detach().cpu()
    trained_seconds = time.time() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.bin"
    samples_path = args.output_dir / "samples.bin"

    # Real trained artifacts replace any synthetic smoke fixtures: clear the
    # demo markers so the benchmark pipeline reports a trained run, not a
    # contradictory mix of both (the markers gate the report labeling).
    for marker in ("demo-model", "demo-samples"):
        marker_path = args.output_dir / marker
        if marker_path.exists():
            marker_path.unlink()

    write_model(model, model_path)
    write_samples(sample_images, sample_labels, samples_path)

    previews = []
    for index in range(min(6, args.samples)):
        pixels = [
            round(float(value) * 255)
            for value in sample_images[index].tolist()
        ]
        previews.append(
            {
                "label": int(sample_labels[index]),
                "prediction": int(sample_predictions[index]),
                "pixels": pixels,
            }
        )

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "MNIST (60,000 train / 10,000 test)",
        "architecture": {
            "input": INPUTS,
            "hidden": HIDDEN,
            "output": OUTPUTS,
            "activation": "ReLU",
            "parameters": int(sum(p.numel() for p in model.parameters())),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "Adam",
            "seed": args.seed,
            "seconds": round(trained_seconds, 2),
            "history": history,
        },
        "evaluation": {
            "test_accuracy": round(history[-1]["test_accuracy"], 5),
            "sample_accuracy": round(sample_accuracy, 5),
            "sample_count": args.samples,
            "confusion_matrix": confusion,
            "per_class_accuracy": [
                round(confusion[d][d] / max(1, sum(confusion[d])), 5)
                for d in range(OUTPUTS)
            ],
        },
        "weights": {
            "count": int(model.hidden.weight.numel() + model.output.weight.numel()),
            "min": round(float(weights.min()), 5),
            "max": round(float(weights.max()), 5),
            "mean_abs": round(float(weights.abs().mean()), 5),
        },
        "previews": previews,
        "artifacts": {
            "model_bytes": model_path.stat().st_size,
            "samples_bytes": samples_path.stat().st_size,
        },
    }
    (args.output_dir / "training.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(
        f"Exported {model_path} ({model_path.stat().st_size} bytes), "
        f"{samples_path} ({samples_path.stat().st_size} bytes), "
        f"training.json"
    )
    print(
        f"Sample-set accuracy: {sample_accuracy:.2%} over {args.samples} images"
    )


if __name__ == "__main__":
    main()
