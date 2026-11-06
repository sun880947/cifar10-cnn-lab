"""Prepare CIFAR-10 ``.npz`` arrays from torchvision or a HuggingFace parquet mirror.

Examples
--------
    # standard torchvision download (needs Internet)
    python -m scripts.prepare_data --source torchvision

    # offline HuggingFace parquet mirror (img.bytes + label columns)
    python -m scripts.prepare_data --source parquet --parquet-dir path/to/parquet
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def save_split(images: np.ndarray, labels: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, images=images.astype(np.uint8), labels=labels.astype(np.int64))
    print(f"saved {path}: images={images.shape} labels={labels.shape}")


def from_torchvision(output_dir: str) -> None:
    from torchvision.datasets import CIFAR10

    root = os.path.join(output_dir, "torchvision")
    train = CIFAR10(root=root, train=True, download=True)
    test = CIFAR10(root=root, train=False, download=True)
    save_split(train.data, np.asarray(train.targets), os.path.join(output_dir, "cifar10_train.npz"))
    save_split(test.data, np.asarray(test.targets), os.path.join(output_dir, "cifar10_test.npz"))


def _read_parquet(path: str, limit: int = 0):
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path, columns=["img", "label"])
    if limit:
        table = table.slice(0, limit)
    images = []
    labels = []
    for row in table.to_pylist():
        image = Image.open(io.BytesIO(row["img"]["bytes"])).convert("RGB")
        images.append(np.asarray(image, dtype=np.uint8))
        labels.append(int(row["label"]))
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def from_parquet(output_dir: str, parquet_dir: str, limit: int = 0) -> None:
    train_path = os.path.join(parquet_dir, "train-00000-of-00001.parquet")
    test_path = os.path.join(parquet_dir, "test-00000-of-00001.parquet")
    for path in (train_path, test_path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    x, y = _read_parquet(train_path, limit)
    save_split(x, y, os.path.join(output_dir, "cifar10_train.npz"))
    x, y = _read_parquet(test_path, limit)
    save_split(x, y, os.path.join(output_dir, "cifar10_test.npz"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["torchvision", "parquet"], default="torchvision")
    p.add_argument("--output-dir", default="data")
    p.add_argument("--parquet-dir", default="")
    p.add_argument("--limit", type=int, default=0, help="debug: use only the first N rows")
    args = p.parse_args()
    if args.source == "torchvision":
        from_torchvision(args.output_dir)
    else:
        if not args.parquet_dir:
            raise SystemExit("--parquet-dir is required for --source parquet")
        from_parquet(args.output_dir, args.parquet_dir, args.limit)


if __name__ == "__main__":
    main()