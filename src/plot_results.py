"""
Produce all figures for the report from the JSON files written by train.py.

    python src/plot_results.py --results-dir results --fig-dir report/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CIFAR10_CLASSES  # noqa: E402

PRETTY = {
    "convnext": "ConvNeXt-CIFAR (ours)",
    "convnext_k3": "ConvNeXt, 3x3 dwconv",
    "convnext_bnrelu": "ConvNeXt, BN + ReLU",
    "convnext_noreg": "ConvNeXt, no LayerScale / no StochDepth",
    "resnet20x2": "ResNet-20 x2 (baseline)",
    "resnet20": "ResNet-20 (baseline)",
}
ORDER = ["convnext", "convnext_k3", "convnext_bnrelu", "convnext_noreg", "resnet20x2", "resnet20"]


def load_results(results_dir: str) -> dict:
    runs = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        runs[r["run_name"]] = r
    return dict(sorted(runs.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def plot_curves(runs: dict, fig_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, r in runs.items():
        ep = [h["epoch"] for h in r["history"]]
        axes[0].plot(ep, [h["train_loss"] for h in r["history"]], "--", alpha=0.6)
        axes[0].plot(ep, [h["test_loss"] for h in r["history"]], label=PRETTY.get(name, name),
                     color=axes[0].lines[-1].get_color())
        axes[1].plot(ep, [h["test_acc"] for h in r["history"]], label=PRETTY.get(name, name))
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss (dashed = train, solid = test)")
    axes[0].set_title("Loss"); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("test accuracy (%)")
    axes[1].set_title("Test accuracy on CIFAR-10"); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(bottom=max(0, axes[1].get_ylim()[0]))
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "training_curves.png"), dpi=160)
    plt.close(fig)


def plot_bar(runs: dict, fig_dir: str) -> None:
    names = list(runs)
    accs = [runs[n]["best_test_acc"] for n in names]
    params = [runs[n]["num_parameters"] / 1e6 for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(names)), accs, color=["#1f77b4" if n == "convnext" else "#9ecae1" for n in names])
    for b, a, p in zip(bars, accs, params):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{a:.2f}%\n({p:.1f}M)",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([PRETTY.get(n, n).replace(", ", ",\n") for n in names], fontsize=8)
    ax.set_ylabel("best test accuracy (%)")
    ax.set_ylim(min(accs) - 3, max(accs) + 2)
    ax.set_title("CIFAR-10 accuracy of ConvNeXt variants and the ResNet-20 x2 baseline")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "accuracy_comparison.png"), dpi=160)
    plt.close(fig)


def plot_confusion(r: dict, fig_dir: str, fname: str = "confusion_matrix.png") -> None:
    cm = np.array(r["confusion_matrix"])
    cm_norm = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=6.5,
                    color="white" if cm_norm[i, j] > 50 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix - {PRETTY.get(r['run_name'], r['run_name'])}\n"
                 f"test accuracy {r['best_test_acc']:.2f}%")
    fig.colorbar(im, ax=ax, fraction=0.046, label="% of true class")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, fname), dpi=160)
    plt.close(fig)


def plot_per_class(runs: dict, fig_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(runs)
    x = np.arange(10)
    for i, (name, r) in enumerate(runs.items()):
        ax.bar(x + i * w - 0.4 + w / 2, [r["per_class_acc"][c] for c in CIFAR10_CLASSES], w,
               label=PRETTY.get(name, name))
    ax.set_xticks(x); ax.set_xticklabels(CIFAR10_CLASSES, rotation=30, ha="right")
    ax.set_ylabel("per-class accuracy (%)")
    ax.set_ylim(60, 101)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-class test accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "per_class_accuracy.png"), dpi=160)
    plt.close(fig)


def write_summary_table(runs: dict, fig_dir: str) -> None:
    rows = []
    for name, r in runs.items():
        rows.append(dict(
            run=name, model=PRETTY.get(name, name),
            params_M=round(r["num_parameters"] / 1e6, 2),
            best_acc=round(r["best_test_acc"], 2), best_epoch=r["best_epoch"],
            final_acc=round(r["final_test_acc"], 2),
            train_min=round(r["total_train_time_s"] / 60, 1),
            epoch_s=round(r["mean_epoch_time_s"], 1),
            throughput=round(r["inference_throughput_img_per_s"]),
        ))
    with open(os.path.join(fig_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"{'run':18s}{'params(M)':>10s}{'best acc':>10s}{'@ep':>5s}{'final':>8s}{'min':>7s}{'s/ep':>6s}{'img/s':>8s}")
    for r in rows:
        print(f"{r['run']:18s}{r['params_M']:10.2f}{r['best_acc']:10.2f}{r['best_epoch']:5d}"
              f"{r['final_acc']:8.2f}{r['train_min']:7.1f}{r['epoch_s']:6.1f}{r['throughput']:8d}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--fig-dir", default="report/figures")
    args = p.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    runs = load_results(args.results_dir)
    if not runs:
        raise SystemExit("no result files found")
    plot_curves(runs, args.fig_dir)
    plot_bar(runs, args.fig_dir)
    plot_per_class(runs, args.fig_dir)
    main_run = runs.get("convnext", next(iter(runs.values())))
    plot_confusion(main_run, args.fig_dir)
    for base in ("resnet20x2", "resnet20"):
        if base in runs:
            plot_confusion(runs[base], args.fig_dir, f"confusion_matrix_{base}.png")
    write_summary_table(runs, args.fig_dir)
    print(f"figures written to {args.fig_dir}")


if __name__ == "__main__":
    main()
