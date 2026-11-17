---
title: |
  AI Assignment \#3 \
  Finding, Implementing, and Evaluating a Recent PCA/LDA Algorithm for Image Recognition
author: |
  HO WENG HONG (UC325107) \
  CISC3024 Pattern Recognition
date: November 2026
fontsize: 11pt
geometry: margin=2.4cm
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes: |
  \usepackage{fancyhdr}
  \pagestyle{fancy}
  \fancyhf{}
  \fancyhead[L]{\small CISC3024 Pattern Recognition}
  \fancyhead[R]{\small AI Assignment \#3}
  \fancyfoot[C]{\small\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
  \usepackage{listings}
  \lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,breakatwhitespace=false,columns=fullflexible,frame=single,framerule=0.3pt,showstringspaces=false,numberstyle=\tiny\color{gray},xleftmargin=1.2em}
---

# Overview of the Assignment and of This Report

The assignment asks for a **recent PCA or LDA algorithm with an application in computer vision and pattern recognition**. Searching, programming, and the writing of this report were done by an AI agent. No code was written by hand.

The algorithm selected is the **2DPCA-then-LDA family** used for face recognition by Jiang, Song, Li, and Chen (IEEE IAEAC, 2024). Classical LDA vectorises an image and then builds a within-class scatter whose size is the number of pixels. With few images that matrix is singular: the small-sample-size problem. Two-dimensional PCA and two-dimensional LDA keep the image as a matrix, so the scatter is the size of one side of the image. The 2024 paper then stacks these pieces into three pipelines: 2DPCA followed by LDA, 2DPCA followed by 2DLDA, and 2DPCA followed by an ordinary PCA stage and then LDA.

The experiment is a complete run on one CPU. It takes about two seconds. A GPU would not change the eigenvalues or the accuracies, so the run was not moved to a faster machine. Twenty Fashion-MNIST images per class are used for training, which is few enough that the 784-dimensional within-class scatter has rank 190 and is singular. Section 4.2 says what was changed relative to the paper's face databases, and Section 5.5 says what the numbers are and are not evidence for.

| # | Required content | Where it is in this report |
| --- | --- | --- |
| 1 | How AI finds the algorithm | Section 2 |
| 2 | Algorithm description | Section 3 |
| 3 | How AI implements the algorithm | Section 4 |
| 4 | Experiment settings and results | Section 5 |
| 5 | What I have learnt from this AI assignment | Section 6 |
| 6 | A webpage link of my source codes | Section 7 (<https://github.com/HOWENGHONG/CISC3024-AI-Assignment/tree/main/assignment3>) |

# How the AI Found the Algorithm

## The instruction

I gave the agent the assignment sheet and told it to do the assignment end to end, in the same form as AI Assignment 2: find the algorithm, implement it, run it, and write the report. The algorithm had to be recent, it had to be PCA or LDA, and it had to be used for images. It also had to be something whose equations could be checked, because a report that only quotes a paper's table is not an experiment.

## The search

The agent searched for `recent PCA LDA algorithm computer vision pattern recognition 2024 2025 paper`, then opened the hits rather than trusting the snippets. Five candidates survived.

| Candidate | Venue | Why it was a candidate | Why it was not selected |
| --- | --- | --- | --- |
| Robust and sparse LDA for image classification | Pattern Recognition, 2024 | A new LDA objective, robust to pixel noise, on image datasets | The PDF is paywalled. Implementing it from the abstract would mean inventing the objective |
| ERSLDA, efficient and robust sparse LDA | IEEE TETCI, 2024 | $L_p$ and $L_{2,p}$ penalties, solved by ADMM, tested on image datasets | Same authors, same problem: no open equation sheet |
| 2DPCA combined with LDA and with 2DLDA | IEEE IAEAC, 2024 | Explicitly a PCA/LDA face-recognition pipeline, aimed at the small-sample-size singularity | Selected |
| Bilateral 2DLDA | Journal of Supercomputing, 2024 | A two-sided 2DLDA for faces | Paywalled. The iteration is not recoverable from the abstract |
| PCA and LDA spectral compression | Optica Open preprint, 2025 | Both methods, used for hyperspectral classification | Needs a hyperspectral cube this machine does not have |

## Why this pipeline

Jiang, Song, Li, and Chen (China University of Petroleum, Beijing) do not invent a new scatter formula. They put three published algorithms into pipelines whose purpose is the failure mode this course has already met: LDA on vectorised images when there are fewer images than pixels. The components have equations I can implement and test without the conference PDF:

1. 2DPCA (Yang, Zhang, Frangi, and Yang, IEEE TPAMI 2004). The reconstruction error of a truncated basis equals the sum of the discarded eigenvalues. That identity does not depend on the dataset.
2. Fisherfaces, PCA followed by LDA (Belhumeur, Hespanha, and Kriegman, IEEE TPAMI 1997). When the within-class scatter is the identity, the LDA direction is exactly the gap between the class means.
3. 2DLDA (Ye, Janardan, and Li, NIPS 2004). The scatter matrices are the size of one side of the image, so they stay invertible when the pixel-space scatter does not.

The 2024 paper's own claim, stated in the abstract, is that 2DPCA+LDA and 2DPCA+2DLDA avoid the small-sample-size problem, and that inserting a PCA stage (2DPCA+PCA+LDA) makes the pipeline faster without giving that up. Those three pipelines are what Section 5 measures. The citation was checked against the DOI 10.1109/IAEAC59436.2024.10503906: Kun Jiang, Yu Song, Shuzhan Li, Kejin Chen, IEEE IAEAC, 15 March 2024.

# Algorithm Description

## The small-sample-size problem

An image of height $m$ and width $n$ is usually flattened into a vector in $\mathbb{R}^{mn}$. PCA and LDA then work in that vector space. For $N$ centred training vectors the total scatter has rank at most $N-1$. The within-class scatter $S_w$ has rank at most $N-C$, where $C$ is the number of classes. A 28 by 28 image has 784 pixels. With 20 images in each of 10 classes, $N-C=190$, so $S_w$ is a 784 by 784 matrix of rank 190. It is not invertible, and the Fisher generalised eigenproblem $S_b w=\lambda S_w w$ is not defined in the usual way. That is the small-sample-size problem.

## PCA and Fisherfaces

PCA finds orthogonal directions that maximise variance. For centred samples $x_i$,
$$
C=\frac{1}{N}\sum_{i=1}^{N} x_i x_i^\top,
$$
and the basis is the eigenvectors of $C$ with the largest eigenvalues. Truncating to $k$ components, the squared reconstruction error, averaged over the training set, equals the sum of the discarded eigenvalues. That is the identity checked in Section 4.3.

Fisher LDA looks for directions that maximise the ratio of between-class scatter to within-class scatter,
$$
J(W)=\frac{\operatorname{tr}(W^\top S_b W)}{\operatorname{tr}(W^\top S_w W)}.
$$
The columns of $W$ are the leading generalised eigenvectors of $(S_b, S_w)$. If $S_w=I$ and the two class means differ only along the first coordinate, that eigenvector is exactly the first coordinate. Fisherfaces (Belhumeur et al.) first project by PCA down to $N-C$ dimensions, where $S_w$ becomes invertible, and only then run LDA. The largest safe PCA dimension is $N-C$. Using a smaller one is a regulariser, not a different algorithm.

## 2DPCA

2DPCA never flattens the image. For images $A_i\in\mathbb{R}^{m\times n}$ with mean $\bar A$, Yang et al. form the $n$ by $n$ matrix
$$
G=\frac{1}{N}\sum_{i=1}^{N}(A_i-\bar A)^\top(A_i-\bar A).
$$
The projection axes $X\in\mathbb{R}^{n\times k}$ are the $k$ leading eigenvectors of $G$. The feature of an image is the $m$ by $k$ matrix $Y=(A-\bar A)X$, which is then read as a vector of length $mk$. Because $G$ is $n$ by $n$, here 28 by 28, it does not become singular just because $mn>N$. Reconstruction is $\hat A=YX^\top+\bar A$. With all $n$ axes this is exact, and the discarded energy again equals the tail of the spectrum of $G$.

## 2DLDA

2DLDA is the supervised version of the same idea. With class means $M_c$ and grand mean $M$, the right-multiplication form used here is
$$
S_w=\sum_c\sum_{A_i\in c}(A_i-M_c)^\top(A_i-M_c),
\qquad
S_b=\sum_c n_c(M_c-M)^\top(M_c-M).
$$
Both matrices are $n$ by $n$. The axes solve $S_b x=\lambda S_w x$, and an image is projected as $AX$. The feature is again an $m$ by $k$ matrix. Nothing of size $mn$ is ever inverted.

## The three cascades in the 2024 paper

The abstract names three combinations. All three are implemented.

1. **2DPCA then LDA.** Project with 2DPCA, vectorise, and run ordinary LDA in that shorter vector space. The LDA step is non-singular only while $mk<N-C$.
2. **2DPCA then 2DLDA.** Project with 2DPCA, keep the result as a matrix, and run 2DLDA on those smaller matrices.
3. **2DPCA then PCA then LDA.** After 2DPCA, an ordinary PCA stage shortens the vector before LDA. The paper introduces this because 2DPCA alone leaves a feature that is still long ($m$ times the number of kept axes), so the following LDA is both slower and, as Section 5 shows, easy to overfit.

Every method is scored by the nearest class mean in its feature space. No neural network is trained. The comparison is the subspace, not a classifier stacked on top of it.

# How the AI Implemented the Algorithm

## What was implemented

`subspace.py` contains the linear algebra. `train.py` builds the Fashion-MNIST split, fits every pipeline, and writes the figures.

| Object | Code |
| --- | --- |
| PCA by a thin SVD | `pca_basis`, `project_pca`, `reconstruct_pca` |
| 2DPCA, Yang et al. | `twodpca_basis`, `project_twodpca`, `reconstruct_twodpca` |
| Within-class and between-class scatter | `scatter_matrices` |
| Fisher LDA, with a ridge so a singular $S_w$ still returns a basis | `lda_basis` |
| Fisher criterion $J(W)$ | `fisher_criterion` |
| Right-side 2DLDA | `twodlda_basis`, `project_twodlda` |
| Nearest class mean | `nearest_class_mean` |
| Rank of the pixel-space $S_w$ | `pixel_within_rank` |

The ridge added to $S_w$ is $10^{-4}$ times $\operatorname{tr}(S_w)/d$. It is there so that Fisherfaces, which push PCA all the way to $N-C$, do not crash on a numerically singular matrix. The rank reported in Section 5.2 is computed before that ridge.

## What was scaled down

The 2024 paper evaluates face databases. This run uses Fashion-MNIST, which is already on disk from Assignment 2, and uses a split that creates the same singularity the paper is about.

- Twenty training images per class, 200 in total, and 200 test images per class, 2,000 in total. Seed 0.
- Images are 28 by 28, pixels divided by 255. No augmentation.
- The pixel scatter is 784 by 784. The 2DPCA scatter is 28 by 28.
- Final LDA dimension is $C-1=9$. The 2DPCA stage that feeds LDA keeps 5 axes, so the intermediate feature has length $28\times 5=140$, which is below $N-C=190$.
- Fisherfaces, for comparison, follow Belhumeur et al. and reduce by PCA to $N-C=190$ before LDA. A second vector pipeline stops PCA at 40 components and then runs LDA. That second one is the regularised operating point; it is not a different paper.

The classifier is the nearest class mean. The paper's face-recognition rates are not claimed.

## Correctness checks, before any Fashion-MNIST image

Two checks in `verify_identities` do not use the dataset.

Thirty random 8 by 6 images were projected by a full 2DPCA basis and reconstructed. The mean squared error was $5.4\times 10^{-31}$. Keeping only two axes, the residual energy and the sum of the discarded eigenvalues differed by $3.6\times 10^{-15}$.

For LDA, the population case $S_w=I$ with class means separated only along the first coordinate produced a direction whose cosine with that coordinate was 1. On a finite sample from the same layout, the Fisher criterion of the LDA direction was 335, against 0.0001 for the trailing PCA axis. The direction that maximises $J(W)$ is not the direction of leftover variance.

# Experiment Settings and Results

## Settings

| Item | Setting |
| --- | --- |
| Dataset | Fashion-MNIST, small-sample split |
| Train / test | 20 images per class / 200 images per class |
| Pixels | divided by 255, no augmentation |
| Pixel dimension | 784 |
| 2DPCA scatter | 28 by 28 |
| Final LDA dimension | 9, one less than the number of classes |
| Decision rule | nearest class mean |
| Seed | 0 |
| Hardware | one CPU; the whole comparison took 2.1 seconds |
| Ridge on $S_w$ | $10^{-4}\operatorname{tr}(S_w)/d$, used only inside LDA |

## The within-class scatter is singular

![Left: eigenvalues of the 784-dimensional within-class scatter. Right: eigenvalues of the 28 by 28 2DPCA covariance.](spectra.png)

The pixel-space within-class scatter has rank **190**, which is exactly $N-C$, out of 784. Its smallest eigenvalue is $-6.2\times 10^{-13}$, a numerical zero. The plot falls off a cliff at index 190. Classical LDA on the raw pixels is not a defined eigenproblem on this split. The 2DPCA covariance, by contrast, is 28 by 28 and its spectrum decays smoothly. The first five components already hold most of the energy, which is why the cascades below keep five 2DPCA axes.

## Accuracy

![Left: nearest-class-mean accuracy against feature dimension for PCA and 2DPCA. Right: the pipelines compared in Section 5.4.](accuracy_comparison.png)

| Method | Feature used at the decision | Test accuracy |
| --- | --- | --- |
| PCA, 40 components | 40 | 67.35% |
| PCA-40, then LDA | 9 | **75.50%** |
| Fisherfaces, PCA to $N-C=190$, then LDA | 9 | 54.95% |
| 2DPCA, 5 components | 140 | 64.55% |
| 2DPCA, then LDA | 9 | 51.10% |
| 2DPCA, then PCA, then LDA | 9 | 71.40% |
| 2DLDA, 9 axes | $28\times 9=252$ | 70.85% |
| 2DPCA, then 2DLDA | $28\times 5=140$ | 65.05% |

Unsupervised PCA levels off: 5 components give 60.70%, 40 give 67.35%, and 80 give 67.45%. 2DPCA needs more coordinates to catch up, because each axis still leaves a vector of length 28. At 8 axes (224 coordinates) it reaches 67.50%, the same plateau as PCA.

## Where LDA helps, and where it overfits

The best number in the table is 75.50%, from PCA down to 40 followed by LDA. That is 8.15 points above PCA-40 alone. The gain is not spread evenly. Pullover goes from 34.5% to 71.0%. Shirt, the class that looks like T-shirt, Pullover, and Coat, goes from 34.5% to 47.0%. Trouser and Ankle boot were already easy under PCA (86.5% and 87.0%) and stay easy (88.5% and 89.0%).

Fisherfaces, which follow the textbook and reduce only to $N-C=190$, score 54.95%. Their training Fisher criterion is 2294, two orders of magnitude above the criterion of PCA-40+LDA, which is 4.2. The training scatter says the 190-dimensional Fisherfaces basis is a much purer separation. The test set says it has fitted the 20 images per class. A huge $J(W)$ on the training split is not evidence.

The same thing happens to the first 2024 cascade. 2DPCA into 140 dimensions and then straight LDA scores 51.10%, worse than 2DPCA alone at 64.55%. Inserting the PCA stage the paper adds for speed, 2DPCA then PCA then LDA, raises that to 71.40%. On this split the inserted stage is not only faster. It is what keeps LDA from memorising the sample. 2DLDA, which never builds the 784-dimensional scatter, scores 70.85% with a feature of length 252. Stacking 2DPCA in front of it lowers that to 65.05%. On Fashion-MNIST the matrix LDA does not need the 2DPCA prefix.

![Eight test images, then the same images rebuilt from 20 PCA components and from 5 2DPCA components.](reconstructions.png)

The reconstructions are a reminder that PCA and 2DPCA are compression methods as well as features. Twenty PCA components and five 2DPCA components both keep the garment shape. They are not the same representation: the accuracy curves in the left panel of the first figure do not overlap.

## What these results do and do not show

They show four things I will stand behind.

The pixel within-class scatter on this split is singular, with rank exactly $N-C=190$. That is the problem the 2024 pipelines are for, and it is measured rather than asserted.

2DPCA's reconstruction identity holds. Full reconstruction error is $5.4\times 10^{-31}$. Truncated reconstruction error matches the discarded eigenvalues to $3.6\times 10^{-15}$.

LDA does what the Fisher criterion says when the geometry is clean. With $S_w=I$, the direction is the class-mean gap, cosine 1. On the images, LDA after a moderate PCA raises test accuracy by 8.15 points, and the class that moves is Pullover, not the classes PCA had already separated.

Pushing the supervised projection to the largest legal dimension overfits. Fisherfaces at $N-C$ and 2DPCA+LDA both lose to their unsupervised prefixes. The 2024 cascade that puts PCA between 2DPCA and LDA repairs that, from 51.10% to 71.40%. It does not, on this dataset, beat ordinary PCA-40+LDA at 75.50%. The paper reports gains on face databases. I did not run those databases, and I do not claim those gains.

# What I Have Learnt from This AI Assignment

## About PCA and LDA

PCA and LDA answer different questions. PCA keeps whatever varies, including within a class. LDA keeps whatever separates classes, and it will discard a high-variance direction if that direction is shared by every class. The population check makes this sharp: when $S_w$ is the identity, no amount of within-class variance is available to confuse the direction, and LDA returns the mean gap with cosine 1. The trailing PCA axis, which is where the leftover variance lives, has a Fisher criterion of essentially zero.

The small-sample-size rank is not a technicality. It is an integer I can compute. Two hundred images, ten classes, rank 190. Every implementation that inverts $S_w$ in pixel space is inverting a matrix that does not have an inverse. 2DPCA and 2DLDA avoid that by changing the size of the matrix, from 784 to 28. Fisherfaces avoid it by throwing away directions before LDA. Both are real fixes. They are not the same fix, and Section 5 shows they do not have the same test accuracy.

The training Fisher criterion is a bad leaderboard. Fisherfaces scored 2294 on the training scatter and 54.95% on the test set. PCA-40+LDA scored 4.2 and 75.50%. A criterion that the same 20 images both define and evaluate will reward a basis that has memorised them. The test images are the only place the criterion is allowed to count.

## About using an AI agent on this assignment

The search was useful because two of the most "recent" papers, the Pattern Recognition 2024 robust LDA and ERSLDA, could not be implemented honestly. Their PDFs are paywalled, and an objective assembled from an abstract is a guess. The agent discarded them. The pipeline that was implemented is one whose equations are in papers I can read, and whose 2024 contribution is the order of the stages, which is something a table of accuracies can test.

That order mattered more than the novelty of any one stage. 2DPCA then LDA, which is the combination the abstract leads with, was the worst supervised number in the table. 2DPCA then PCA then LDA, which the abstract presents as a speed trick, was the one that worked. I would have reported the wrong conclusion if I had implemented only the first cascade and treated a higher training criterion as success.

A GPU would not have improved this assignment. The run is a few eigenvalue decompositions of a 28 by 28 matrix and a 784 by 784 scatter, and it finished in 2.1 seconds. The accuracies are properties of the subspaces. Moving the same 2,200 images onto an RTX 4060 does not change an eigenvector. The machine with the 7600X and the 4060 is the right place for the generative course project, which actually trains a model. It is the wrong reason to rerun this one.

# Webpage Link of Source Codes

## <https://github.com/HOWENGHONG/CISC3024-AI-Assignment/tree/main/assignment3>

| File | Content |
| --- | --- |
| `assignment3/subspace.py` | PCA, 2DPCA, LDA, 2DLDA, and the identity checks |
| `assignment3/train.py` | The Fashion-MNIST comparison and the figures in Section 5 |
| `assignment3/README.md` | How to rerun the numbers |

```bash
git clone https://github.com/HOWENGHONG/CISC3024-AI-Assignment.git
cd CISC3024-AI-Assignment/assignment3
pip install numpy scipy matplotlib
python subspace.py
python train.py
```

`python subspace.py` needs no download. `python train.py` reads Fashion-MNIST from `assignment2/data/FashionMNIST/raw/` and takes a few seconds on CPU.

# References

1. Jiang, K., Song, Y., Li, S., Chen, K. (2024). *Feature extraction and recognition of face image based on 2DPCA with LDA algorithm.* IEEE IAEAC 2024. doi:10.1109/IAEAC59436.2024.10503906.
2. Yang, J., Zhang, D., Frangi, A. F., Yang, J. (2004). *Two-Dimensional PCA: A New Approach to Appearance-Based Face Representation and Recognition.* IEEE TPAMI, 26(1), 131--137.
3. Belhumeur, P. N., Hespanha, J. P., Kriegman, D. J. (1997). *Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection.* IEEE TPAMI, 19(7), 711--720.
4. Ye, J., Janardan, R., Li, Q. (2004). *Two-Dimensional Linear Discriminant Analysis.* Advances in Neural Information Processing Systems 17.
5. Xiao, H., Rasul, K., Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747.

# Appendix A: `subspace.py`

```{.python .numberLines}
<<<SUBSPACE>>>
```

# Appendix B: `train.py`

```{.python .numberLines}
<<<TRAIN>>>
```
