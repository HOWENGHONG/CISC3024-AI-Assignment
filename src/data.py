"""CIFAR-10 data pipeline (arrays are prepared/cached by prepare_data.py)."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import load_split  # noqa: E402

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


class CIFAR10Arrays(Dataset):
    """CIFAR-10 held in memory as a uint8 array of shape (N, 32, 32, 3)."""

    def __init__(self, data_dir: str, train: bool, transform=None):
        self.x, self.y = load_split(data_dir, "train" if train else "test")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        img = Image.fromarray(self.x[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[i])


def build_transforms(augment: str = "light"):
    """augment: "none" | "light" (random crop + horizontal flip) | "strong"
    (light + RandAugment + RandomErasing, the kind of recipe used by ConvNeXt)."""
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ops = []
    if augment in ("light", "strong"):
        ops += [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ]
    if augment == "strong":
        ops += [transforms.RandAugment(num_ops=2, magnitude=9)]
    ops += [transforms.ToTensor(), normalize]
    if augment == "strong":
        ops += [transforms.RandomErasing(p=0.25)]
    train_tf = transforms.Compose(ops)
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    return train_tf, test_tf


def _worker_init(_: int) -> None:
    # data workers only do light per-image ops; keep them from spawning a
    # full intra-op thread pool each, which would starve the training process
    torch.set_num_threads(1)


def build_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 3,
    augment: str = "light",
):
    train_tf, test_tf = build_transforms(augment)
    train_set = CIFAR10Arrays(data_dir, train=True, transform=train_tf)
    test_set = CIFAR10Arrays(data_dir, train=False, transform=test_tf)

    pin = torch.cuda.is_available()
    common = dict(num_workers=num_workers, pin_memory=pin,
                  persistent_workers=num_workers > 0, worker_init_fn=_worker_init)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, **common)
    return train_loader, test_loader
