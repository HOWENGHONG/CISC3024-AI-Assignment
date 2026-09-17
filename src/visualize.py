"""
Extra figures: architecture diagram of the ConvNeXt block vs. the ResNet block,
and a grid of test images with the predictions of the trained ConvNeXt.

    python src/visualize.py --checkpoint results/convnext_best.pt --fig-dir report/figures
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision import transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, CIFAR10Arrays  # noqa: E402
from models import build_model  # noqa: E402


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                        fc=color, ec="black", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))


def draw_block_diagram(fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2))
    for ax in axes:
        ax.set_xlim(0, 4); ax.set_ylim(0, 8); ax.axis("off")

    # --- ResNet bottleneck / basic block -----------------------------------------
    ax = axes[0]
    ax.set_title("ResNet block (2016)", fontsize=11)
    layers = [("1x1 conv, C -> C/4", "#f4cccc"), ("BatchNorm + ReLU", "#fff2cc"),
              ("3x3 conv, C/4 -> C/4", "#f4cccc"), ("BatchNorm + ReLU", "#fff2cc"),
              ("1x1 conv, C/4 -> C", "#f4cccc"), ("BatchNorm", "#fff2cc")]
    y = 6.6
    for text, color in layers:
        _box(ax, 1.0, y, 2.0, 0.6, text, color)
        _arrow(ax, 2.0, y, 2.0, y - 0.35)
        y -= 0.95
    _box(ax, 1.0, y, 2.0, 0.6, "+  (then ReLU)", "#d9ead3")
    ax.plot([0.6, 0.6], [y + 0.3, 7.6], color="black", lw=1.2)
    ax.plot([0.6, 2.0], [7.6, 7.6], color="black", lw=1.2)
    _arrow(ax, 0.6, y + 0.3, 1.0, y + 0.3)
    _arrow(ax, 2.0, 7.9, 2.0, 7.25)
    ax.text(2.1, 7.85, "x", fontsize=9)
    ax.text(0.15, 4.0, "identity", rotation=90, fontsize=8, va="center")

    # --- ConvNeXt block -------------------------------------------------------------
    ax = axes[1]
    ax.set_title("ConvNeXt block (2022)", fontsize=11)
    layers = [("depthwise 7x7 conv, C -> C", "#cfe2f3"), ("LayerNorm", "#fff2cc"),
              ("1x1 conv (Linear), C -> 4C", "#cfe2f3"), ("GELU", "#fff2cc"),
              ("1x1 conv (Linear), 4C -> C", "#cfe2f3"), ("Layer Scale (x gamma)", "#e6d0f5")]
    y = 6.6
    for text, color in layers:
        _box(ax, 1.0, y, 2.0, 0.6, text, color)
        _arrow(ax, 2.0, y, 2.0, y - 0.35)
        y -= 0.95
    _box(ax, 1.0, y, 2.0, 0.6, "+  (Stochastic Depth on branch)", "#d9ead3")
    ax.plot([0.6, 0.6], [y + 0.3, 7.6], color="black", lw=1.2)
    ax.plot([0.6, 2.0], [7.6, 7.6], color="black", lw=1.2)
    _arrow(ax, 0.6, y + 0.3, 1.0, y + 0.3)
    _arrow(ax, 2.0, 7.9, 2.0, 7.25)
    ax.text(2.1, 7.85, "x", fontsize=9)
    ax.text(0.15, 4.0, "identity", rotation=90, fontsize=8, va="center")

    fig.suptitle("Micro design: from the ResNet block to the ConvNeXt block\n"
                 "(large-kernel depthwise conv, inverted bottleneck, fewer norms/activations, LN + GELU)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "convnext_block.png"), dpi=170)
    plt.close(fig)


def draw_macro_diagram(fig_dir: str) -> None:
    """Overall ConvNeXt-CIFAR architecture as a horizontal pipeline."""
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(0, 23.5); ax.set_ylim(0, 4); ax.axis("off")
    stages = [
        ("Input\n3x32x32", "#ffffff"),
        ("Stem\n3x3 conv, 32\n+ LN", "#d9d9d9"),
        ("Stage 1\n1 x Block\nC=32, 32x32", "#cfe2f3"),
        ("Down\nLN + 2x2/s2", "#d9d9d9"),
        ("Stage 2\n1 x Block\nC=64, 16x16", "#cfe2f3"),
        ("Down\nLN + 2x2/s2", "#d9d9d9"),
        ("Stage 3\n3 x Block\nC=128, 8x8", "#9fc5e8"),
        ("Down\nLN + 2x2/s2", "#d9d9d9"),
        ("Stage 4\n1 x Block\nC=256, 4x4", "#cfe2f3"),
        ("GAP + LN\n+ Linear\n10 classes", "#d9ead3"),
    ]
    x = 0.3
    widths = [1.6, 1.8, 2.3, 1.7, 2.3, 1.7, 2.3, 1.7, 2.3, 2.2]
    for k, ((text, color), w) in enumerate(zip(stages, widths)):
        _box(ax, x, 1.0, w, 2.0, text, color)
        if k < len(stages) - 1:
            _arrow(ax, x + w + 0.03, 2.0, x + w + 0.33, 2.0)
        x += w + 0.35
    ax.set_title("ConvNeXt-CIFAR macro design (stage ratio 1:1:3:1, separate downsampling layers)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "convnext_macro.png"), dpi=170)
    plt.close(fig)


@torch.no_grad()
def sample_predictions(checkpoint: str, model_name: str, fig_dir: str, data_dir: str = "./data") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_set = CIFAR10Arrays(data_dir, train=False, transform=tf)
    g = torch.Generator().manual_seed(1)
    idx = torch.randperm(len(test_set), generator=g)[:24].tolist()
    xs = torch.stack([test_set[i][0] for i in idx]).to(device)
    ys = [test_set[i][1] for i in idx]
    probs = model(xs).softmax(1).cpu()
    preds = probs.argmax(1).tolist()

    mean = np.array(CIFAR10_MEAN)[:, None, None]; std = np.array(CIFAR10_STD)[:, None, None]
    fig, axes = plt.subplots(3, 8, figsize=(13, 5.6))
    for ax, i, x, y, p, pr in zip(axes.flat, idx, xs.cpu().numpy(), ys, preds, probs):
        img = np.clip(x * std + mean, 0, 1).transpose(1, 2, 0)
        ax.imshow(img); ax.axis("off")
        ok = p == y
        ax.set_title(f"pred: {CIFAR10_CLASSES[p]} ({pr[p]*100:.0f}%)\ntrue: {CIFAR10_CLASSES[y]}",
                     fontsize=7.5, color="green" if ok else "red")
    fig.suptitle("ConvNeXt-CIFAR predictions on random CIFAR-10 test images (green = correct, red = wrong)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "sample_predictions.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/convnext_best.pt")
    p.add_argument("--model", default="convnext")
    p.add_argument("--fig-dir", default="report/figures")
    p.add_argument("--data-dir", default="./data")
    args = p.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    draw_block_diagram(args.fig_dir)
    draw_macro_diagram(args.fig_dir)
    if os.path.exists(args.checkpoint):
        sample_predictions(args.checkpoint, args.model, args.fig_dir, args.data_dir)
    else:
        print(f"checkpoint {args.checkpoint} not found - skipping sample predictions")
    print(f"figures written to {args.fig_dir}")


if __name__ == "__main__":
    main()
