# CISC3024 Pattern Recognition — AI Assignment #1

**Student:** Ho Weng Hong (uc325107)
**AI tool:** Cursor agent mode (Claude) — the agent did the literature search, wrote all code, ran the experiments and generated the report.

## Algorithm

**ConvNeXt** — Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie,
*"A ConvNet for the 2020s"*, CVPR 2022 ([arXiv:2201.03545](https://arxiv.org/abs/2201.03545)).

A pure deep CNN that "modernises" ResNet with ideas from Vision Transformers
(stage ratio 1:1:3:1, patchify stem, depthwise 7×7 convolution, inverted
bottleneck, LayerNorm, GELU, one norm / one activation per block, separate
down-sampling layers, layer scale, stochastic depth).
Here it is re-implemented from scratch in PyTorch, adapted to 32×32 inputs and
evaluated on **CIFAR-10 image classification**, together with three ablations
that undo steps of the paper's "modernisation roadmap" (3×3 kernel,
BatchNorm+ReLU, no layer-scale/stochastic-depth) and a parameter-matched
ResNet-20×2 baseline. Everything was trained on a 6-core CPU (see the report
for why no GPU could be used).

## Repository layout

| Path | Content |
| --- | --- |
| `src/models.py` | `ConvNeXtBlock`, `ConvNeXtCIFAR` (with ablation switches), CIFAR ResNet baseline |
| `src/prepare_data.py` | downloads CIFAR-10 (Hugging Face Parquet mirror or original tarball) and caches it as `.npz` |
| `src/data.py` | dataset / augmentation / data loaders |
| `src/train.py` | training & evaluation; writes `results/<run>.json` with history, confusion matrix, timing |
| `src/run_experiments.py` | runs the whole suite (`convnext`, `convnext_k3`, `convnext_bnrelu`, `convnext_noreg`, `resnet20x2`) |
| `src/plot_results.py`, `src/visualize.py` | all figures in `report/figures/` |
| `report/make_report.py` | builds the Word report from the result files |
| `report/CISC3024_AI_Assignment1_HoWengHong_uc325107.docx` / `.pdf` | **the report** |
| `results/` | JSON results and training logs of the runs reported |
| `prompts/PROMPTS.md` | log of the prompts given to the AI agent |

## Reproduce

```bash
pip install -r requirements.txt
python src/prepare_data.py                 # CIFAR-10 -> data/cifar10_{train,test}.npz
python src/run_experiments.py --epochs 15  # ~2 h on a 6-core CPU; much faster on a GPU
python src/plot_results.py
python src/visualize.py
python report/make_report.py
```

Single run, e.g. the main model: `python src/train.py --model convnext --epochs 15`.
Available models: `convnext`, `convnext_k3`, `convnext_bnrelu`, `convnext_noreg`, `resnet20`, `resnet20x2`.

## Results (CIFAR-10 test accuracy, 15 epochs, CPU bf16, seed 0)

Numbers are filled in automatically from `results/*.json` by
`report/make_report.py`; the report discusses why the BatchNorm ResNet wins
in this short-schedule regime and what each ablation shows.

<!-- results:start -->
| Model | Params | Best test acc. | Final test acc. | Train time (CPU) | Inference img/s |
| --- | --- | --- | --- | --- | --- |
| ConvNeXt-CIFAR (ours) | 1.18 M | **80.27%** | 80.27% | 28.8 min | 1838 |
| ConvNeXt, 3x3 dwconv | 1.15 M | **80.16%** | 80.13% | 24.4 min | 1911 |
| ConvNeXt, BN + ReLU | 1.18 M | **83.42%** | 83.42% | 21.5 min | 3017 |
| ConvNeXt, no LayerScale / no StochDepth | 1.18 M | **82.86%** | 82.86% | 28.0 min | 2024 |
| ResNet-20 x2 (baseline) | 1.08 M | **90.67%** | 90.67% | 20.0 min | 2374 |
<!-- results:end -->
