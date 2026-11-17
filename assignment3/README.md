# AI Assignment #3 — 2DPCA, LDA, and the 2024 cascades

A from-scratch implementation of the PCA / LDA pipelines in

> Kun Jiang, Yu Song, Shuzhan Li, Kejin Chen.
> **Feature extraction and recognition of face image based on 2DPCA with LDA algorithm.**
> IEEE IAEAC, 2024. doi:10.1109/IAEAC59436.2024.10503906

The component algorithms are Yang et al. (2DPCA, TPAMI 2004), Belhumeur et al. (Fisherfaces, TPAMI 1997), and Ye et al. (2DLDA, NIPS 2004).

## Files

| File | Content |
| --- | --- |
| `subspace.py` | PCA, 2DPCA, LDA, 2DLDA, nearest class mean, and the reconstruction / Fisher-direction checks |
| `train.py` | Small-sample Fashion-MNIST comparison. Writes `results.json` and the figures used in the report |

## Reproducing the reported results

```bash
pip install numpy scipy matplotlib
python subspace.py   # reconstruction identity and the LDA direction, no dataset
python train.py      # a few seconds on CPU; seed 0
```

`train.py` reads the Fashion-MNIST files already stored in `../assignment2/data/FashionMNIST/raw/`.

## Results obtained

20 training images per class, 200 test images per class. The pixel within-class scatter is 784 by 784 and has rank 190, which is exactly the small-sample-size singularity.

| Method | Test accuracy |
| --- | --- |
| PCA, 40 components | 67.35% |
| PCA-40 then LDA | 75.50% |
| Fisherfaces (PCA down to n − C, then LDA) | 54.95% |
| 2DPCA, 5 components | 64.55% |
| 2DPCA then LDA | 51.10% |
| 2DPCA then PCA then LDA | 71.40% |
| 2DLDA | 70.85% |
| 2DPCA then 2DLDA | 65.05% |

The identity checks: a full 2DPCA basis reconstructs to MSE 5.4 × 10⁻³¹, and the discarded energy matches the leftover eigenvalues to 3.6 × 10⁻¹⁵. When the within-class scatter is the identity, the LDA direction lines up with the class-mean gap (cosine 1).
