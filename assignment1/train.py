"""
Train & evaluate RepViT-mini on scikit-learn's `load_digits` dataset
(1797 8x8 grayscale images of handwritten digits 0-9), then measure the
practical effect of structural re-parameterization (parameter count and
CPU inference latency, before vs after fusing).
"""
import time
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repvit_mini import RepViTMini, RepVGGDW, count_params

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------- data ----
digits = load_digits()
X = digits.images.astype(np.float32) / 16.0        # scale to [0,1]
y = digits.target.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

X_train_t = torch.from_numpy(X_train).unsqueeze(1)   # (N,1,8,8)
y_train_t = torch.from_numpy(y_train)
X_test_t = torch.from_numpy(X_test).unsqueeze(1)
y_test_t = torch.from_numpy(y_test)

print(f"train: {X_train_t.shape}, test: {X_test_t.shape}")

# --------------------------------------------------------------- model ----
model = RepViTMini(num_classes=10, widths=(24, 48), depths=(2, 2))
print(f"RepViT-mini parameter count (train-time, multi-branch): {count_params(model):,}")

opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 60
BATCH = 32
n = X_train_t.shape[0]

history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(n)
    tot_loss, tot_correct = 0.0, 0
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = X_train_t[idx], y_train_t[idx]
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        tot_loss += loss.item() * xb.size(0)
        tot_correct += (out.argmax(1) == yb).sum().item()
    sched.step()

    train_loss = tot_loss / n
    train_acc = tot_correct / n

    model.eval()
    with torch.no_grad():
        test_out = model(X_test_t)
        test_acc = (test_out.argmax(1) == y_test_t).float().mean().item()

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)

    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
              f"train_acc {train_acc:.4f}  test_acc {test_acc:.4f}")

# ------------------------------------------------------- final metrics ----
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    preds = logits.argmax(1)
final_test_acc = (preds == y_test_t).float().mean().item()
print(f"\nFinal test accuracy (un-fused, train-time multi-branch model): {final_test_acc:.4f}")

# confusion matrix
conf = np.zeros((10, 10), dtype=int)
for t, p in zip(y_test_t.numpy(), preds.numpy()):
    conf[t, p] += 1

# ------------------------------------------ before/after fusion checks ----
with torch.no_grad():
    logits_before_fuse = model(X_test_t).clone()

params_before = count_params(model)
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(200):
        model(X_test_t)
t_before = (time.perf_counter() - t0) / 200

model.fuse()

with torch.no_grad():
    logits_after_fuse = model(X_test_t)
params_after = count_params(model)
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(200):
        model(X_test_t)
t_after = (time.perf_counter() - t0) / 200

max_diff = (logits_before_fuse - logits_after_fuse).abs().max().item()
acc_after_fuse = (logits_after_fuse.argmax(1) == y_test_t).float().mean().item()

print("\n=== Structural re-parameterization: before vs after fusing ===")
print(f"max |logits_before - logits_after| over test set : {max_diff:.3e}")
print(f"test accuracy before fuse : {final_test_acc:.4f}")
print(f"test accuracy after  fuse : {acc_after_fuse:.4f}")
print(f"parameters   before fuse  : {params_before:,}")
print(f"parameters   after  fuse  : {params_after:,}  "
      f"({100*(1-params_after/params_before):.1f}% fewer)")
print(f"CPU latency  before fuse  : {t_before*1000:.3f} ms / batch of {X_test_t.shape[0]}")
print(f"CPU latency  after  fuse  : {t_after*1000:.3f} ms / batch of {X_test_t.shape[0]}  "
      f"({(1-t_after/t_before)*100:.1f}% faster)")

results = {
    "final_test_acc": final_test_acc,
    "acc_after_fuse": acc_after_fuse,
    "max_logit_diff_after_fuse": max_diff,
    "params_before_fuse": params_before,
    "params_after_fuse": params_after,
    "latency_ms_before_fuse": t_before * 1000,
    "latency_ms_after_fuse": t_after * 1000,
    "history": history,
    "confusion_matrix": conf.tolist(),
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

# ------------------------------------------------------------- plots ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

axes[0].plot(history["epoch"], history["train_loss"], color="tab:red")
axes[0].set_title("Training loss")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("cross-entropy loss")
axes[0].grid(alpha=0.3)

axes[1].plot(history["epoch"], history["train_acc"], label="train acc")
axes[1].plot(history["epoch"], history["test_acc"], label="test acc")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("accuracy")
axes[1].legend()
axes[1].grid(alpha=0.3)

im = axes[2].imshow(conf, cmap="Blues")
axes[2].set_title(f"Confusion matrix (test acc={final_test_acc:.3f})")
axes[2].set_xlabel("predicted")
axes[2].set_ylabel("true")
axes[2].set_xticks(range(10))
axes[2].set_yticks(range(10))
for i in range(10):
    for j in range(10):
        if conf[i, j] > 0:
            axes[2].text(j, i, conf[i, j], ha="center", va="center",
                         color="white" if conf[i, j] > conf.max() / 2 else "black",
                         fontsize=8)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("\nSaved training_curves.png and results.json")

# a few sample predictions
fig2, axes2 = plt.subplots(2, 8, figsize=(12, 3.2))
for i, ax in enumerate(axes2.flat):
    img = X_test[i]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"T:{y_test[i]} P:{preds[i].item()}",
                 color="black" if y_test[i] == preds[i].item() else "red", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
print("Saved sample_predictions.png")
