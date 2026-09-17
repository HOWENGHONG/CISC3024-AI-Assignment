"""
Obtain CIFAR-10 and cache it as compact NumPy arrays (data/cifar10_{train,test}.npz).

Two sources are supported:

1. The Hugging Face mirror `uoft-cs/cifar10` (Parquet files with PNG-encoded
   images).  This is the default because the original University of Toronto
   server was unreachable / extremely slow from the machine used for this
   assignment.
2. The original `cifar-10-python.tar.gz` (via torchvision) if it is already
   present in the data directory.

Both give exactly the same 50 000 / 10 000 images and the same class order
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

    python src/prepare_data.py --data-dir data
"""

from __future__ import annotations

import argparse
import io
import os
import urllib.request

import numpy as np

HF_BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
HF_FILES = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}


def npz_path(data_dir: str, split: str) -> str:
    return os.path.join(data_dir, f"cifar10_{split}.npz")


def parquet_path(data_dir: str, split: str) -> str:
    return os.path.join(data_dir, f"cifar10_{split}.parquet")


def _download(url: str, dst: str) -> None:
    print(f"downloading {url} -> {dst}")
    tmp = dst + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst)


def _from_parquet(path: str):
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path)
    imgs = table.column("img").to_pylist()
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
    x = np.empty((len(imgs), 32, 32, 3), dtype=np.uint8)
    for i, rec in enumerate(imgs):
        x[i] = np.asarray(Image.open(io.BytesIO(rec["bytes"])).convert("RGB"))
    return x, labels


def _from_torchvision(data_dir: str, split: str):
    from torchvision import datasets

    ds = datasets.CIFAR10(data_dir, train=(split == "train"), download=False)
    return np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64)


def prepare_split(data_dir: str, split: str) -> str:
    out = npz_path(data_dir, split)
    if os.path.exists(out):
        return out
    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(os.path.join(data_dir, "cifar-10-batches-py")) or os.path.exists(
        os.path.join(data_dir, "cifar-10-python.tar.gz")
    ):
        x, y = _from_torchvision(data_dir, split)
    else:
        pq_file = parquet_path(data_dir, split)
        if not os.path.exists(pq_file):
            _download(HF_BASE + HF_FILES[split], pq_file)
        x, y = _from_parquet(pq_file)
    assert x.shape[1:] == (32, 32, 3), x.shape
    np.savez_compressed(out, x=x, y=y)
    print(f"{split}: {x.shape[0]} images -> {out}  (class counts {np.bincount(y).tolist()})")
    return out


def load_split(data_dir: str, split: str):
    d = np.load(prepare_split(data_dir, split))
    return d["x"], d["y"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    a = p.parse_args()
    for s in ("train", "test"):
        prepare_split(a.data_dir, s)
