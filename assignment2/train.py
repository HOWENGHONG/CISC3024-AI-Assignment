"""
Train and compare three classifiers on a long-tailed Fashion-MNIST subset:

  * Softmax cross-entropy (implicit posterior, the paper's Eq. 1)
  * Logit adjustment (Menon et al., ICLR 2021; the paper's Eq. 3)
  * BAPE-mini (explicit vMF Bayes classifier + an LA head; Eqs. 6 and 13)

The test set is balanced. BAPE is scored twice on that same test set: once
with the long-tailed training prior, once after distribution adjustment to
the uniform test prior. That second number is the claim the algorithm makes.
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST

from bape import BAPEModel, SmallCNN, count_params, la_logits_adjusted, verify_map_and_adjustment

torch.manual_seed(0)
np.random.seed(0)

IMBALANCE = 100
N_MAX = 1000
EPOCHS = 20
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
ETA = 1.0
SEED = 0


def long_tail_counts(num_classes, n_max, imbalance):
    """n_i = n_max * imbalance^(-i / (K-1)), the CIFAR-LT exponential rule."""
    counts = []
    for i in range(num_classes):
        n_i = int(round(n_max * imbalance ** (-i / (num_classes - 1))))
        counts.append(max(n_i, 1))
    return counts


def shot_group(count):
    if count > 100:
        return "many"
    if count >= 20:
        return "medium"
    return "few"


def build_long_tail():
    train_set = FashionMNIST(root="data", train=True, download=True)
    test_set = FashionMNIST(root="data", train=False, download=True)
    images = train_set.data.numpy()
    labels = train_set.targets.numpy()
    counts = long_tail_counts(10, N_MAX, IMBALANCE)
    rng = np.random.default_rng(SEED)
    chosen = []
    for c, n_c in enumerate(counts):
        pool = np.where(labels == c)[0]
        chosen.append(rng.choice(pool, size=n_c, replace=False))
    index = np.concatenate(chosen)
    rng.shuffle(index)

    x = images[index].astype(np.float32) / 255.0
    y = labels[index].astype(np.int64)
    x_test = test_set.data.numpy().astype(np.float32) / 255.0
    y_test = test_set.targets.numpy().astype(np.int64)
    return x, y, x_test, y_test, counts


def make_loader(x, y, shuffle):
    xt = torch.from_numpy(x).unsqueeze(1)
    yt = torch.from_numpy(y)
    return DataLoader(TensorDataset(xt, yt), batch_size=BATCH, shuffle=shuffle)


def per_class_accuracy(pred, target, num_classes=10):
    acc = []
    for c in range(num_classes):
        mask = target == c
        acc.append(float((pred[mask] == c).mean()) if mask.any() else float("nan"))
    return acc


def group_accuracy(per_class, counts):
    buckets = {"many": [], "medium": [], "few": []}
    for acc, n in zip(per_class, counts):
        buckets[shot_group(n)].append(acc)
    return {name: (float(np.mean(vals)) if vals else float("nan")) for name, vals in buckets.items()}


def evaluate_logits(logits, y_test, counts):
    pred = logits.argmax(dim=1).cpu().numpy()
    target = y_test
    per_class = per_class_accuracy(pred, target)
    groups = group_accuracy(per_class, counts)
    return {
        "all": float((pred == target).mean()),
        "many": groups["many"],
        "medium": groups["medium"],
        "few": groups["few"],
        "per_class": per_class,
    }


class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SmallCNN()
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        return self.head(self.backbone(x))


def iter_batches(x, batch=256):
    for start in range(0, len(x), batch):
        yield torch.from_numpy(x[start:start + batch]).unsqueeze(1)


def collect_baseline_logits(model, x):
    chunks = []
    model.eval()
    with torch.no_grad():
        for xb in iter_batches(x):
            chunks.append(model(xb))
    return torch.cat(chunks, dim=0)


def collect_bape(model, x):
    la_chunks, z_chunks = [], []
    model.eval()
    with torch.no_grad():
        for xb in iter_batches(x):
            la, z = model(xb)
            la_chunks.append(la)
            z_chunks.append(z)
        la = torch.cat(la_chunks, dim=0)
        z = torch.cat(z_chunks, dim=0)
        return la, model.bayes_logits(z, model.pi_train), model.bayes_logits(z, model.pi_test)


def _train_baseline(kind, model, loader, test_x, y_test, pi_train, counts):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            if kind == "la":
                logits = la_logits_adjusted(logits, pi_train)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
        sched.step()
        logits = collect_baseline_logits(model, test_x)
        acc = float((logits.argmax(1).numpy() == y_test).mean())
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "test_acc": acc})
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  {kind:4s} epoch {epoch:2d}  loss {total_loss / seen:.4f}  test {acc:.4f}", flush=True)
    metrics = evaluate_logits(collect_baseline_logits(model, test_x), y_test, counts)
    metrics["history"] = history
    metrics["params"] = count_params(model)
    return metrics


def train_bape(loader, test_x, y_test, counts):
    torch.manual_seed(SEED)
    model = BAPEModel(num_classes=10, class_counts=counts)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            la_logits, z = model(xb, yb, update_stats=True)
            bape_logits = model.bayes_logits(z, model.pi_train)
            loss_la = loss_fn(la_logits_adjusted(la_logits, model.pi_train), yb)
            loss_bape = loss_fn(bape_logits, yb)
            loss = loss_bape + ETA * loss_la
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
        sched.step()
        _, _, adjusted = collect_bape(model, test_x)
        acc = float((adjusted.argmax(1).numpy() == y_test).mean())
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "test_acc": acc})
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  bape epoch {epoch:2d}  loss {total_loss / seen:.4f}  test(DA) {acc:.4f}", flush=True)

    la_logits, logits_train_prior, logits_test_prior = collect_bape(model, test_x)
    with torch.no_grad():
        # LA head, scored for a balanced test set (raw logits).
        p_la = torch.softmax(la_logits, dim=1)
        p_bape = torch.softmax(logits_test_prior, dim=1)
        ensemble = 0.5 * (p_la + p_bape)
        mu, kappa = model.map_parameters()

    results = {
        "train_prior": evaluate_logits(logits_train_prior, y_test, counts),
        "test_prior": evaluate_logits(logits_test_prior, y_test, counts),
        "la_head": evaluate_logits(la_logits, y_test, counts),
        "ensemble": evaluate_logits(ensemble, y_test, counts),
        "history": history,
        "params": count_params(model),
        "kappa": kappa.detach().cpu().tolist(),
        "mu_row_norm_max_err": float((mu.norm(dim=1) - 1).abs().max()),
    }
    return results


def plot(counts, ce, la, bape, path_curves, path_per_class, path_kappa):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, run, color in (
        ("softmax CE", ce, "tab:gray"),
        ("logit adjustment", la, "tab:orange"),
        ("BAPE + DA", bape, "tab:blue"),
    ):
        hist = run["history"]
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_loss"] for h in hist], color=color, label=name)
        axes[1].plot(epochs, [h["test_acc"] for h in hist], color=color, label=name)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_title("Balanced test accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path_curves, dpi=150)
    plt.close(fig)

    classes = np.arange(10)
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(classes - width, ce["per_class"], width, label="softmax CE", color="tab:gray")
    ax.bar(classes, la["per_class"], width, label="logit adjustment", color="tab:orange")
    ax.bar(classes + width, bape["test_prior"]["per_class"], width, label="BAPE + DA", color="tab:blue")
    ax.set_xticks(classes)
    ax.set_xticklabels([f"{c}\n(n={counts[c]})" for c in classes])
    ax.set_ylim(0, 1)
    ax.set_ylabel("per-class test accuracy")
    ax.set_title("Head classes are on the left, tail classes on the right")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path_per_class, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(counts, bape["kappa"], color="tab:blue")
    for c, (n, k) in enumerate(zip(counts, bape["kappa"])):
        ax.annotate(str(c), (n, k), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("training images in class (log scale)")
    ax.set_ylabel("estimated concentration kappa")
    ax.set_title("BAPE concentration vs class frequency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_kappa, dpi=150)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    checks = verify_map_and_adjustment()
    x, y, x_test, y_test, counts = build_long_tail()
    print("class counts:", counts)
    print("train images:", len(y), "test images:", len(y_test))
    loader = make_loader(x, y, shuffle=True)
    pi_train = torch.tensor(counts, dtype=torch.float32)
    pi_train = pi_train / pi_train.sum()

    print("\n=== softmax cross-entropy ===")
    ce = _train_baseline("ce", BaselineNet(), loader, x_test, y_test, pi_train, counts)
    print("\n=== logit adjustment ===")
    la = _train_baseline("la", BaselineNet(), loader, x_test, y_test, pi_train, counts)
    print("\n=== BAPE ===")
    bape = train_bape(loader, x_test, y_test, counts)

    plot(
        counts,
        ce,
        la,
        bape,
        "training_curves.png",
        "per_class_accuracy.png",
        "kappa_vs_frequency.png",
    )
    groups = {
        "many": [i for i, n in enumerate(counts) if shot_group(n) == "many"],
        "medium": [i for i, n in enumerate(counts) if shot_group(n) == "medium"],
        "few": [i for i, n in enumerate(counts) if shot_group(n) == "few"],
    }
    results = {
        "dataset": "Fashion-MNIST long-tailed subset",
        "imbalance_factor": IMBALANCE,
        "n_max": N_MAX,
        "class_counts": counts,
        "shot_groups": groups,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "eta": ETA,
        "alpha_hat": 4.0,
        "beta_hat": 0.8,
        "embed_dim": 32,
        "seed": SEED,
        "checks": checks,
        "softmax_ce": ce,
        "logit_adjustment": la,
        "bape": bape,
        "seconds": time.perf_counter() - t0,
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    def line(name, m):
        print(
            f"{name:22s}  all {m['all']:.4f}  many {m['many']:.4f}  "
            f"medium {m['medium']:.4f}  few {m['few']:.4f}"
        )

    print("\n=== balanced test accuracy ===")
    line("softmax CE", ce)
    line("logit adjustment", la)
    line("BAPE train prior", bape["train_prior"])
    line("BAPE test prior", bape["test_prior"])
    line("BAPE LA head", bape["la_head"])
    line("BAPE ensemble", bape["ensemble"])
    print(f"kappa by class: {['%.2f' % k for k in bape['kappa']]}")
    print(f"done in {results['seconds']:.1f}s")


if __name__ == "__main__":
    main()
