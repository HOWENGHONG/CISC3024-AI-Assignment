"""
Run the complete experiment suite sequentially (main model, ablations, baseline).

    python src/run_experiments.py --epochs 15
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

RUNS = ["convnext", "convnext_k3", "convnext_bnrelu", "convnext_noreg", "resnet20x2"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--runs", nargs="*", default=RUNS)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.out_dir, exist_ok=True)
    for run in args.runs:
        if args.skip_existing and os.path.exists(os.path.join(args.out_dir, f"{run}.json")):
            print(f"skipping {run} (results exist)")
            continue
        cmd = [sys.executable, "-u", os.path.join(here, "train.py"), "--model", run,
               "--epochs", str(args.epochs), "--out-dir", args.out_dir]
        log_path = os.path.join(args.out_dir, f"log_{run}.txt")
        print(f"\n>>> {' '.join(cmd)}\n    (log: {log_path})", flush=True)
        with open(log_path, "w", encoding="utf-8") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
