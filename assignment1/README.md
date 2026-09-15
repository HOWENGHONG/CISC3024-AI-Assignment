# AI Assignment #1 — RepViT-mini

A small but mechanically faithful reproduction of the core ideas of

> A. Wang, H. Chen, Z. Lin, J. Han, G. Ding.
> **RepViT: Revisiting Mobile CNN From ViT Perspective.** CVPR 2024, pp. 15909–15920.
> [arXiv:2307.09283](https://arxiv.org/abs/2307.09283)

## Files

| File | Content |
| --- | --- |
| `repvit_mini.py` | Model definition: `RepVGGDW` re-parameterizable token mixer, `SqueezeExcite`, `ChannelMixer`, `RepViTBlock`, `RepViTMini`, plus a numerical self-check of the fusion algebra |
| `train.py` | Trains and evaluates RepViT-mini on scikit-learn's `load_digits`, then measures the effect of structural re-parameterization (accuracy / parameter count / CPU latency, before vs. after fusing) and writes `results.json`, `training_curves.png`, `sample_predictions.png` |

## Reproducing the reported results

```bash
pip install torch scikit-learn matplotlib numpy
python repvit_mini.py   # unit-test of the re-parameterization fusion
python train.py         # training, evaluation, fusion benchmark, plots
```

Everything runs on CPU in a couple of minutes; all random seeds are fixed to 0.

## Results obtained

| Metric | Before fusing | After fusing |
| --- | --- | --- |
| Test accuracy | 99.44% | 99.44% |
| Parameters | 38,956 | 38,092 (−2.2%) |
| CPU latency / 360-image batch | 36.6 ms | 25.5 ms (−30.4%) |

Fusing the multi-branch token mixer into a single depthwise convolution leaves the outputs
unchanged up to float32 rounding error (max logit difference 6.7 × 10⁻⁶) while making the network
smaller and faster — the trade-off that RepViT's structural re-parameterization is designed to
deliver.
