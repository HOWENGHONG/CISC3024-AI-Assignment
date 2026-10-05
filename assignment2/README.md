# AI Assignment #2 — BAPE-mini

A small but mechanically faithful reproduction of the core of

> Chaoqun Du, Yulin Wang, Shiji Song, Gao Huang.
> **BAPE: Learning an Explicit Bayes Classifier for Long-tailed Visual Recognition.**
> arXiv:2506.23280, 2025.

## Files

| File | Content |
| --- | --- |
| `bape.py` | vMF Bayes classifier: ETF prior, online MAP estimation of `(mu, kappa)`, Bayes posterior, distribution adjustment, and a numerical self-check |
| `train.py` | Long-tailed Fashion-MNIST experiment comparing softmax cross-entropy, logit adjustment, and BAPE. Writes `results.json` and the three figures used in the report |

## Reproducing the reported results

```bash
pip install torch torchvision scipy matplotlib numpy
python bape.py    # MAP and prior-swap checks, no dataset needed
python train.py   # about 12 minutes on CPU; all seeds fixed to 0
```

Fashion-MNIST is downloaded into `data/` on the first run.

## Results obtained

Balanced Fashion-MNIST test set, imbalance factor 100, 20 epochs, CPU:

| Method | All | Many | Medium | Few |
| --- | --- | --- | --- | --- |
| Softmax cross-entropy | 80.70% | 88.04% | 68.33% | 80.90% |
| Logit adjustment | 84.17% | 87.70% | 74.33% | 90.10% |
| BAPE, training prior | 80.75% | 87.08% | 69.97% | 81.10% |
| BAPE, test prior (distribution adjustment) | 82.08% | 86.86% | 72.07% | 85.15% |

Replacing the long-tailed training prior with the uniform test prior changes no network weight. It raises overall accuracy by 1.33 points and few-shot accuracy by 4.05 points. On this small model the logit-adjustment head is still the strongest single predictor (84.34% when trained inside the BAPE network). That comparison, and the checks `cos(mu_hat, mu_true) = 0.999911` and `kappa_hat = 31.50` against a true concentration of 30, are what the run is for.
