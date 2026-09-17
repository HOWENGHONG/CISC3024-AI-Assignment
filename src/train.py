"""
Train / evaluate a model on CIFAR-10 and store everything needed for the report.

Example
-------
    python src/train.py --model convnext   --epochs 15 --out-dir results
    python src/train.py --model resnet20x2 --epochs 15 --out-dir results

The training recipe follows the spirit of the ConvNeXt paper (AdamW, cosine
schedule with linear warm-up, label smoothing, stochastic depth) scaled down
to a small CIFAR-10 setting that can be trained on a CPU.  bf16 mixed
precision is used both on CUDA GPUs and on CPUs that support it (AVX-512 BF16).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CIFAR10_CLASSES, build_loaders  # noqa: E402
from models import build_model, count_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-10 training for AI Assignment #1")
    p.add_argument("--model", default="convnext")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--threads", type=int, default=6, help="CPU threads for PyTorch ops")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./results")
    p.add_argument("--run-name", default=None, help="defaults to --model")
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    p.add_argument("--log-interval", type=int, default=100, help="print progress every N iterations (0 = off)")
    p.add_argument("--augment", default="light", choices=["none", "light", "strong"],
                   help="none | light (crop + flip) | strong (+ RandAugment + RandomErasing)")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, criterion, use_amp: bool):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
        loss = criterion(logits.float(), y)
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += y.size(0)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return total_loss / n, 100.0 * correct / n, preds, targets


@torch.no_grad()
def measure_throughput(model: nn.Module, device, use_amp: bool, batch_size: int = 256, iters: int = 20):
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return batch_size * iters / (time.perf_counter() - t0)


def main() -> None:
    args = parse_args()
    run_name = args.run_name or args.model
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp  # bf16 autocast works on CUDA and on modern CPUs
    torch.backends.cudnn.benchmark = True
    if device.type == "cpu":
        torch.set_num_threads(args.threads)

    train_loader, test_loader = build_loaders(
        args.data_dir, args.batch_size, args.workers, augment=args.augment
    )
    model = build_model(args.model).to(device).to(memory_format=torch.channels_last)
    n_params = count_parameters(model)

    # AdamW with no weight decay on norm / bias / layer-scale parameters
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".gamma") else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.999),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor()
    print(f"device={device} ({device_name}, threads={torch.get_num_threads()})")
    print(f"model={args.model} params={n_params/1e6:.2f}M amp(bf16)={use_amp} epochs={args.epochs} "
          f"steps/epoch={steps_per_epoch} augment={args.augment}")

    history = []
    best_acc, best_state = 0.0, None
    step = 0
    t_train0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_loss, running_correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            lr = cosine_lr(step, total_steps, warmup_steps, args.lr, args.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
            it = step - (epoch - 1) * steps_per_epoch
            if args.log_interval and it % args.log_interval == 0 and it < steps_per_epoch:
                print(f"  [{run_name}] ep {epoch} it {it:4d}/{steps_per_epoch} "
                      f"loss {running_loss/seen:.4f} acc {100*running_correct/seen:5.1f}% "
                      f"{(time.perf_counter()-t0)/it*1000:.0f} ms/it", flush=True)

        train_loss = running_loss / seen
        train_acc = 100.0 * running_correct / seen
        test_loss, test_acc, _, _ = evaluate(model, test_loader, device, criterion, use_amp)
        epoch_time = time.perf_counter() - t0
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(
            dict(epoch=epoch, lr=lr, train_loss=train_loss, train_acc=train_acc,
                 test_loss=test_loss, test_acc=test_acc, epoch_time_s=epoch_time)
        )
        print(
            f"[{run_name}] epoch {epoch:3d}/{args.epochs} lr {lr:.2e} "
            f"train loss {train_loss:.4f} acc {train_acc:6.2f}% | "
            f"test loss {test_loss:.4f} acc {test_acc:6.2f}% (best {best_acc:6.2f}%) | {epoch_time:5.1f}s",
            flush=True,
        )
    total_train_time = time.perf_counter() - t_train0

    # ---- final evaluation with the best checkpoint ------------------------------------------
    model.load_state_dict(best_state)
    test_loss, test_acc, preds, targets = evaluate(model, test_loader, device, criterion, use_amp)
    num_classes = len(CIFAR10_CLASSES)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        confusion[t, p] += 1
    per_class_acc = (confusion.diagonal() / confusion.sum(1) * 100.0).tolist()
    throughput = measure_throughput(model, device, use_amp)

    results = dict(
        run_name=run_name,
        model=args.model,
        config=vars(args),
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            device=f"{device.type}: {device_name}",
            threads=torch.get_num_threads(),
            amp_bf16=use_amp,
            os=platform.platform(),
        ),
        num_parameters=n_params,
        final_test_acc=history[-1]["test_acc"],
        best_test_acc=best_acc,
        best_epoch=int(np.argmax([h["test_acc"] for h in history]) + 1),
        test_loss_best=test_loss,
        per_class_acc=dict(zip(CIFAR10_CLASSES, per_class_acc)),
        confusion_matrix=confusion.tolist(),
        total_train_time_s=total_train_time,
        mean_epoch_time_s=float(np.mean([h["epoch_time_s"] for h in history])),
        inference_throughput_img_per_s=throughput,
        history=history,
    )
    out_json = os.path.join(args.out_dir, f"{run_name}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    torch.save(best_state, os.path.join(args.out_dir, f"{run_name}_best.pt"))
    print(f"\n[{run_name}] best test acc {best_acc:.2f}% (epoch {results['best_epoch']}), "
          f"final {results['final_test_acc']:.2f}%, params {n_params/1e6:.2f}M, "
          f"train time {total_train_time/60:.1f} min, throughput {throughput:.0f} img/s")
    print(f"results written to {out_json}")


if __name__ == "__main__":
    main()
