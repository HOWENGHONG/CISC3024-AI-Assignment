---
title: |
  AI Assignment \#2 \
  Finding, Implementing, and Evaluating a Recent Bayes Classifier for Visual Recognition
author: |
  HO WENG HONG (UC325107) \
  CISC3024 Pattern Recognition
date: October 2026
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
  \fancyhead[R]{\small AI Assignment \#2}
  \fancyfoot[C]{\small\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
  \usepackage{listings}
  \lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,breakatwhitespace=false,columns=fullflexible,frame=single,framerule=0.3pt,showstringspaces=false,numberstyle=\tiny\color{gray},xleftmargin=1.2em}
---

# Overview of the Assignment and of This Report

The assignment asks for a **recent classification or recognition algorithm based on probability or Bayes theory**, applied to computer vision and pattern recognition. Searching, programming, and the writing of this report were done by an AI agent. No code was written by hand.

The algorithm selected is **BAPE** (Bayes classifier by Point Estimation), arXiv:2506.23280, 2025. It replaces the usual softmax classifier with an explicit Bayes classifier: L2-normalised image features are modelled as a von Mises–Fisher distribution, the distribution's parameters are estimated by maximum a posteriori point estimation rather than by gradient descent, and at test time the class prior is swapped from the long-tailed training frequencies to the test frequencies.

The experiment is a scaled-down but complete run on one CPU. A small convolutional network is trained for 20 epochs on a long-tailed subset of Fashion-MNIST (imbalance factor 100, 2,480 training images) and scored on the full balanced test set of 10,000 images. The paper's CIFAR-LT / ImageNet-LT / ResNet setting was not rerun. Section 4.2 says exactly what was changed, and Section 5.5 says what the numbers are and are not evidence for.

| # | Required content | Where it is in this report |
| --- | --- | --- |
| 1 | How AI finds the algorithm | Section 2 |
| 2 | Algorithm description | Section 3 |
| 3 | How AI implements the algorithm | Section 4 |
| 4 | Experiment settings and results | Section 5 |
| 5 | What I have learnt from this AI assignment | Section 6 |
| 6 | A webpage link of my source codes | Section 7 (<https://github.com/HOWENGHONG/CISC3024-AI-Assignment/tree/main/assignment2>) |

# How the AI Found the Algorithm

## The instruction

I gave the agent the assignment sheet and told it to do the assignment end to end: find the algorithm, implement it, run it, and write the report. The constraint I cared about was the one in the sheet. The algorithm had to be recent, it had to be a classifier or recogniser, and it had to be based on probability or Bayes theory, in computer vision. It also had to be something the agent could actually run here, because a report that only quotes a paper's tables is not an experiment.

## The search

The agent searched the web for `recent Bayesian classification algorithm computer vision 2024 2025 paper`, then opened the papers rather than trusting the search snippets. Four candidates survived that check.

| Candidate | Venue | Why it was a candidate | Why it was not selected |
| --- | --- | --- | --- |
| BAPE: an explicit Bayes classifier for long-tailed recognition | arXiv:2506.23280, 2025 | The decision rule is Bayes' theorem. Parameters are point-estimated. The prior can be changed at test time | Selected |
| Bayesian Class Adaptation for vision-language models | CVPR 2025 | Updates both the likelihood and the class prior of a CLIP classifier at test time | Needs a pretrained CLIP model. The Bayes step is an adaptation trick on top of CLIP, not a classifier built from the likelihood |
| Bayesian exploration of pretrained models for low-shot classification | CVPR 2024 | A Gaussian-process ensemble whose mean is CLIP | Same dependency on large pretrained models, which this CPU cannot host |
| Make Me a BNN (ABNN) | CVPR 2024 | Turns a pretrained network into a Bayesian neural net with small adaptation layers | The interesting object is post-hoc uncertainty, and the method assumes a large pretrained model |

## Why BAPE

BAPE is the candidate that is literally the Bayes classifier from this course, applied to a current vision problem (long-tailed recognition), and whose two claims can be checked without ImageNet:

1. The MAP estimate of a von Mises–Fisher distribution recovers the mean direction and the concentration. That is a numerical identity. It does not depend on the dataset.
2. Changing only the class prior $\pi_y$ inside the posterior moves the logits by exactly $\log(\pi^{\text{new}}/\pi^{\text{old}})$. That is distribution adjustment, and it is the quantity the long-tailed experiment is then used to measure.

The citation was checked against the paper itself: Chaoqun Du, Yulin Wang, Shiji Song, Gao Huang, Department of Automation, Tsinghua University, arXiv:2506.23280. There is no official code release. The implementation below was written from the equations in Section 3 of the paper, not from a copied repository.

# Algorithm Description: BAPE

## The problem

A classifier that minimises the probability of error is the Bayes classifier
$$
\hat y(\mathbf x)=\arg\max_y p(y\mid\mathbf x)
=\arg\max_y \frac{p(y)\,p(\mathbf x\mid y)}{\sum_{y'} p(y')\,p(\mathbf x\mid y')}.
$$
A standard deep network never writes this down. It trains a linear layer with softmax cross-entropy,
$$
p(y\mid\mathbf x)=\frac{\exp(\mathbf w_y^\top \mathbf z+b_y)}{\sum_{y'}\exp(\mathbf w_{y'}^\top \mathbf z+b_{y'})},
$$
and hopes the output matches the posterior. On a balanced dataset that works. On a long-tailed dataset it does not, for two reasons the paper separates. During training, head classes dominate the gradient, and the decision rule that comes out has absorbed the *training* prior $p(y)$. The test set of a long-tailed benchmark is usually balanced, so that prior is the wrong one.

Logit adjustment (Menon et al., ICLR 2021) is the immediate predecessor. It adds $\log\pi_y$ to the logits, with $\pi_y$ the training frequency, so the linear layer is pushed toward the class-conditional rather than the training posterior. BAPE's objection is that this is still an implicit estimate: $\pi_y$, the class mean, and the concentration are all mixed into the same learned weights. BAPE estimates those pieces separately, and only the feature extractor is trained by gradient descent.

## The feature model

Real images are too complicated to give $p(\mathbf x\mid y)$ a formula. BAPE models the *features*. Let $\mathbf z=F(\mathbf x)/\|F(\mathbf x)\|$ be an L2-normalised embedding on the unit sphere. The paper's reason for a spherical model is the neural-collapse observation that features from one class gather around a single direction. The matching distribution is the von Mises–Fisher (vMF) law,
$$
p(\mathbf z\mid y)=\frac{1}{C_p(\kappa_y)}\exp\!\big(\kappa_y\,\boldsymbol\mu_y^\top\mathbf z\big),
\qquad
C_p(\kappa)=\frac{(2\pi)^{p/2} I_{p/2-1}(\kappa)}{\kappa^{p/2-1}},
$$
where $\boldsymbol\mu_y$ is a unit mean direction, $\kappa_y\ge 0$ is a concentration, $p$ is the embedding dimension, and $I_\nu$ is the modified Bessel function of the first kind. Large $\kappa_y$ means the class forms a tight cluster. $\kappa_y=0$ is the uniform distribution on the sphere.

Substituting the vMF density into Bayes' theorem gives the BAPE posterior. Up to a constant that does not depend on $y$,
$$
\log p(y\mid\mathbf z)
=\log\pi_y-\log C_p(\kappa_y)+\kappa_y\,\boldsymbol\mu_y^\top\mathbf z.
$$
This is still a linear classifier in $\mathbf z$. The difference from softmax is that $\boldsymbol\mu_y$ and $\kappa_y$ are computed from data, and $\pi_y$ is an input you choose.

## Point estimation

For $n_y$ unit vectors from class $y$, the sufficient statistic is their sum. With a conjugate prior that the paper interprets as $\alpha_0$ pseudo-observations of total length $\beta_0$ in a direction $\mathbf m_0$, the MAP estimates are
$$
\boldsymbol\mu=\mathbf m,
\qquad
\frac{I_{p/2}(\kappa)}{I_{p/2-1}(\kappa)}=\frac{\beta}{\alpha},
$$
where $\alpha=\alpha_0+n_y$, $\beta=\|\beta_0\mathbf m_0+\sum_i\mathbf z_i\|$, and $\mathbf m$ is that vector normalised. The paper uses Sra's closed-form approximation
$$
\hat\kappa=\frac{p\,\beta\,\alpha}{\alpha^2-\beta^2}
$$
instead of solving the Bessel ratio. The running sum is maintained online: each mini-batch updates the class sum and the class count. No classifier weight is stepped by the optimiser.

The prior direction $\mathbf m_0$ is initialised as a simplex equiangular tight frame, so the classes start maximally separated, and the paper then allows gradient updates of $\mathbf m_0$ only. $\alpha_0$ and $\beta_0$ are set proportional to the class size $N_y$. A strong prior keeps the estimate stable while the features are still random; as real samples accumulate, the prior's relative weight falls.

## The training loss, and distribution adjustment

Early in training the point estimate is noisy, so BAPE does not rely on it alone. A logit-adjustment head shares the backbone, and the loss is
$$
\mathcal L=\mathcal L_{\text{BAPE}}+\eta\,\mathcal L_{\text{LA}},
$$
with $\eta=1$ in the paper and in this reproduction. $\mathcal L_{\text{BAPE}}$ is cross-entropy on the Bayes logits above, using the training prior. $\mathcal L_{\text{LA}}$ is cross-entropy on the linear head after $\log\pi^{\text{train}}$ has been added.

Distribution adjustment uses the fact that $\pi_y$ was never baked into $\boldsymbol\mu$ and $\kappa$. The same estimated likelihood can be paired with any test prior. For a balanced test set, $\pi^{\text{test}}_y=1/K$. The logits move by
$$
\Delta_y=\log\pi^{\text{test}}_y-\log\pi^{\text{train}}_y
$$
and by nothing else. Head classes, which had a large training prior, are pushed down. Tail classes are pushed up. No weight changes, and the cost is one subtraction per class.

# How the AI Implemented the Algorithm

## What was implemented

`bape.py` contains the pieces that the equations specify.

| Paper object | Code |
| --- | --- |
| Unit-sphere embedding | `F.normalize` on the projection head |
| ETF prior direction $\mathbf m_0$ | `etf_prior`, stored as the parameter `m0` |
| Online class sum, Eq. 11 | `BAPEModel.update_statistics` |
| $\alpha,\beta$, $\boldsymbol\mu$, Sra's $\hat\kappa$ | `map_parameters`, `kappa_from_resultant` |
| $\log C_p(\kappa)$ | `log_vmf_normalizer`, via `scipy.special.ive` so the Bessel function does not overflow |
| Bayes logits | `bayes_logits(z, pi)` |
| Logit-adjustment head | `la_head` plus `la_logits_adjusted` |
| Distribution adjustment | call `bayes_logits` with `pi_test` instead of `pi_train` |

The sample sum is detached from the autograd graph. Gradients update the backbone, the two heads, and $\mathbf m_0$. They do not update $\boldsymbol\mu$ and $\kappa$ as if those were ordinary weights. That is the point of point estimation.

## What was scaled down, and why

The paper trains ResNet-32 for 200 epochs on CIFAR-10/100-LT, and ResNet-50 on ImageNet-LT and iNaturalist, with AutoAugment and a second augmented view for the BAPE head. This machine is a CPU with no GPU. The following changes keep the decision rule intact and make the run finish:

- **Data.** Fashion-MNIST, 28 by 28 grayscale, 10 classes. The training set is subsampled with the same exponential rule the paper uses for CIFAR-LT, $n_i=n_{\max}\gamma^{-i/(K-1)}$, with $n_{\max}=1000$ and imbalance factor $\gamma=100$. That gives the counts 1000, 599, 359, 215, 129, 77, 46, 28, 17, 10 (2,480 images). The test set is the original balanced set, 1,000 images per class.
- **Network.** A four-convolution CNN, width 32 then 64, then a 128-d fully connected layer. The BAPE projection head maps 128 to 32. The paper's head is 512 to 128 on top of ResNet-32.
- **Optimiser and schedule.** AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, cosine schedule, 20 epochs, batch 64. The paper uses SGD, momentum 0.9, batch 256, 200 epochs.
- **Prior strength.** The paper sets $\hat\alpha_0=40$ and $\hat\beta_0=8$ because after 200 epochs the data term is about $200 N_y$. Here the data term after 20 epochs is about $20 N_y$, so $\hat\alpha_0=4$ and $\hat\beta_0=0.8$ keep the same end-of-training prior fraction, about one sixth, and the same ratio $\beta_0/\alpha_0=0.2$.
- **One view, no augmentation.** Both heads see the same image. The paper feeds the LA head one augmented view and the BAPE head two SimCLR-style views.

$\eta=1$ is unchanged. The test prior is uniform. Shot groups follow the paper: many-shot means more than 100 training images (classes 0–4), medium-shot means 20 to 100 (classes 5–7), few-shot means fewer than 20 (classes 8–9).

## Correctness checks, before any images

Two checks in `verify_map_and_adjustment` do not use Fashion-MNIST.

A Wood (1994) sampler drew 4,000 unit vectors in 16 dimensions from a known vMF. With the prior turned off, the MAP mean had cosine 0.999911 with the true mean. Sra's formula returned $\hat\kappa=31.50$ when the true concentration was 30, and $\hat\kappa=5.06$ when the true concentration was 5. The estimate moves in the right direction and is close in absolute value.

The second check builds a two-class posterior and replaces $\pi=(0.9, 0.1)$ with $\pi=(0.5, 0.5)$. The change in every logit matched $\log(\pi^{\text{test}}/\pi^{\text{train}})$ with maximum absolute error $6.0\times 10^{-8}$. Distribution adjustment, as implemented, is an exact prior swap. It cannot silently rescale $\kappa$ or rotate $\boldsymbol\mu$.

After training, every estimated $\boldsymbol\mu_y$ had unit norm to within $1.2\times 10^{-7}$.

# Experiment Settings and Results

## Settings

| Item | Setting |
| --- | --- |
| Dataset | Fashion-MNIST, long-tailed training subset, full balanced test set |
| Imbalance factor | 100 |
| Class counts | 1000, 599, 359, 215, 129, 77, 46, 28, 17, 10 |
| Split of classes | many: 0–4; medium: 5–7; few: 8–9 |
| Pixels | divided by 255, no augmentation |
| Models | softmax CE; logit adjustment; BAPE (LA head + vMF head) |
| Parameters | 431,018 (baselines), 451,978 (BAPE, the extra is the projection head and $\mathbf m_0$) |
| Loss | cross-entropy; LA adds $\log\pi^{\text{train}}$; BAPE adds the two losses with $\eta=1$ |
| Optimiser | AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, cosine schedule |
| Epochs / batch | 20 / 64 |
| Seed | 0 for PyTorch, NumPy, and the subsample |
| Hardware | one CPU, no GPU; the whole comparison took 713 seconds |
| Test scoring | CE and LA: raw logits, which is the balanced-test correction. BAPE: reported under both priors |

The three runs share the CNN, the optimiser, the epoch count, and the seed. BAPE has one extra head, which is part of the method, not a larger backbone.

## Overall accuracy

![Left: training loss. Right: accuracy on the balanced test set. BAPE is scored after distribution adjustment. Its loss is the sum of two terms, so the left-hand curve starts higher.](training_curves.png)

| Method | All | Many | Medium | Few |
| --- | --- | --- | --- | --- |
| Softmax cross-entropy | 80.70% | 88.04% | 68.33% | 80.90% |
| Logit adjustment | 84.17% | 87.70% | 74.33% | 90.10% |
| BAPE, training prior | 80.75% | 87.08% | 69.97% | 81.10% |
| BAPE, test prior (distribution adjustment) | 82.08% | 86.86% | 72.07% | 85.15% |
| BAPE's LA head alone | 84.34% | 87.50% | 73.80% | 92.25% |
| Ensemble, average of the LA head and the adjusted BAPE posterior | 82.89% | 87.18% | 72.27% | 88.10% |

Logit adjustment is the strongest standalone method on this setup, 3.47 points above softmax. BAPE with the training prior is essentially tied with softmax (80.75% against 80.70%). Switching to the test prior, which changes no weight, raises BAPE to 82.08%. The LA head that was trained inside the BAPE network reaches 84.34%, a hair above the separately trained LA model. Averaging it with the vMF posterior pulls that number back down to 82.89%, so on this run the ensemble does not help.

## Where the prior swap actually moves the accuracy

![Per-class test accuracy. Classes are ordered from the most frequent (left) to the least frequent (right).](per_class_accuracy.png)

The class-wise comparison between the two BAPE priors is the measurement that matches the algebra in Section 4.4.

| Class | Name | $n$ | Train prior | Test prior | Change |
| --- | --- | --- | --- | --- | --- |
| 0 | T-shirt | 1000 | 95.4% | 94.4% | $-1.0$ |
| 1 | Trouser | 599 | 98.7% | 98.6% | $-0.1$ |
| 2 | Pullover | 359 | 88.6% | 87.6% | $-1.0$ |
| 3 | Dress | 215 | 82.5% | 83.2% | $+0.7$ |
| 4 | Coat | 129 | 70.2% | 70.5% | $+0.3$ |
| 5 | Sandal | 77 | 97.1% | 97.1% | $0.0$ |
| 6 | Shirt | 46 | 25.8% | 32.2% | $+6.4$ |
| 7 | Sneaker | 28 | 87.0% | 86.9% | $-0.1$ |
| 8 | Bag | 17 | 78.5% | 84.0% | $+5.5$ |
| 9 | Ankle boot | 10 | 83.7% | 86.3% | $+2.6$ |

The three classes with a clear gain are the three scarcest that are also confusable or rare: Shirt ($+6.4$), Bag ($+5.5$), Ankle boot ($+2.6$). The two largest classes drop by about one point. That is what a prior correction is supposed to do. Summed over the test set, the net change is $+1.33$ points (80.75% to 82.08%), and the few-shot group moves by $+4.05$ points (81.10% to 85.15%).

Shirt is the class softmax fails. With 46 training images it is only "medium" by the paper's cutoff, but it is visually close to T-shirt, Pullover, and Coat, which between them have 1,588 images. Softmax scores 20.6% on Shirt. Logit adjustment scores 38.5%. BAPE after the prior swap scores 32.2%, up from 25.8% with the training prior. The explicit classifier recovers most of the way from softmax toward logit adjustment, and does it by editing $\pi$, not by retraining.

Few-shot accuracy looks high next to medium-shot accuracy (80.90% against 68.33% for softmax) only because of which classes the exponential schedule put in the tail. Bag and Ankle boot are easy shapes. Shirt, which sits in the medium group, is the hard one. Group averages hide that. The per-class table is the one to read.

## Concentration is not a copy of the class count

![Estimated vMF concentration against the number of training images. Class 1, not class 0, is the tightest cluster.](kappa_vs_frequency.png)

$\hat\kappa$ runs from 54.3 (Bag, 17 images) to 76.9 (Trouser, 599 images). The largest class, T-shirt, has $\hat\kappa=64.2$, close to the smallest class, Ankle boot, at 61.1. The correlation of $\hat\kappa$ with $\log n_y$ is 0.49: there is a mild trend, not the proportionality you would see if the classifier were just memorising frequency. Trousers form a tight cluster because they look alike, not because they are the most frequent class. This is the qualitative claim of the paper's Figure 1, at this scale. It is not a reproduction of that figure's CIFAR-100 plot.

## What these results do and do not show

They show three things I am willing to stand behind.

The MAP algebra is implemented correctly. On synthetic vMF data, the mean direction is recovered to cosine 0.999911 and the concentration to about 31.5 against a truth of 30.

Distribution adjustment is an exact prior swap, error $6.0\times 10^{-8}$ on the logit identity, and on the real test set it does the Bayes thing: tail classes gain, the largest classes give a little back, and the net accuracy rises by 1.33 points with no retraining.

On this CPU-scale problem, an explicit vMF classifier does not beat logit adjustment. The best number in the table is 84.34%, from the linear LA head. The paper reports BAPE ahead of logit adjustment by about two points on CIFAR-100-LT (52.5% against 50.5% at imbalance 100, ResNet-32, 200 epochs). I did not reproduce that gap, and I do not claim it. The network, the data, the augmentation, and the training length are all smaller here. The honest reading is that separating the prior from the likelihood is enough to *move* accuracy in the predicted direction. It is not, by itself, enough to beat a strong implicit method on a 2,480-image subset.

# What I Have Learnt from This AI Assignment

## About Bayes decision theory, now that it is a program

The Bayes classifier is a product of two objects that a softmax layer refuses to keep separate: a class prior $\pi_y$, and a class-conditional likelihood $p(\mathbf z\mid y)$. Softmax cross-entropy estimates the product. That is why a model trained on a long-tailed sample is a good model of the training distribution and a biased model of a balanced test. Logit adjustment tries to undo the prior by adding $\log\pi_y$ during training, so the weights are pressured to represent the likelihood. BAPE goes one step further and never lets the prior enter the estimated parameters. Once $\boldsymbol\mu$ and $\kappa$ are on record, any test imbalance is a change of $\pi$, not a change of the network.

The numerical identity matters more than the slogan. Because the logit shift equals $\log(\pi^{\text{test}}/\pi^{\text{train}})$ and nothing else, a wrong implementation cannot hide inside a higher accuracy. The unit test would fail even if the Fashion-MNIST number looked flattering.

I also learnt not to trust a shot-group average until I know which classes landed in the group. "Few-shot 80.9%, medium-shot 68.3%" sounds like imbalance does not hurt. It is Bag and Ankle boot versus Shirt. The per-class plot is the result; the group means are a summary that can lie.

Concentration was a useful surprise. $\kappa$ is estimated from how aligned a class's features are, so an easy class with fewer images (Trouser) can be tighter than a hard class with more images (Pullover, Coat). Frequency and difficulty are different axes. A method that only reweights by $n_y$ never sees the second one.

## About using an AI agent on a theory paper

The search was useful because it was narrow. "Recent", "Bayes", "classification", and "must run on this CPU" knocked out three of the four serious hits, including two CVPR papers, for a reason I can explain: they assume CLIP. A search for "best long-tail paper" would have returned a leaderboard method whose contribution is not Bayes theory, and the assignment would have been the wrong assignment.

There is no released BAPE code. That was an advantage. The agent had to transcribe equations, and the transcription is checked by the synthetic-vMF test rather than by "it matches the author's script". Where the paper is specific, the code follows it: the Sra approximation, the online sum, the ETF prior, the sum of the two losses, the test-time choice of $\pi$. Where I scaled a hyperparameter ($\hat\alpha_0$ from 40 to 4), the report says so and says why the ratio was preserved.

The result I did not get is part of the lesson. I expected the explicit classifier to win, because the theory is cleaner. It did not. The LA head, which is the implicit method, scored higher. Cleaner bookkeeping of $\pi$ is not the same thing as a better feature extractor. The paper's own gains are measured on top of a large network trained for a long time; the bookkeeping cannot invent features the CNN never learnt. Shirt at 32% rather than 38% is the trace of that limit.

A GPU would change the scale of Section 5. It would let me train ResNet-32 on CIFAR-10-LT and ask whether the two-point gap in the paper's Table 1 shows up. It would not change Section 4.4. Those checks are identities. They are the part of the assignment that a faster machine cannot improve, and the part I would want marked even if the accuracy table were ignored.

# Webpage Link of Source Codes

## <https://github.com/HOWENGHONG/CISC3024-AI-Assignment/tree/main/assignment2>

| File | Content |
| --- | --- |
| `assignment2/bape.py` | vMF MAP classifier, ETF prior, distribution adjustment, synthetic checks |
| `assignment2/train.py` | Long-tailed Fashion-MNIST comparison and the figures in Section 5 |
| `assignment2/README.md` | How to rerun the numbers |

```bash
git clone https://github.com/HOWENGHONG/CISC3024-AI-Assignment.git
cd CISC3024-AI-Assignment/assignment2
pip install torch torchvision scipy matplotlib numpy
python bape.py
python train.py
```

`python bape.py` needs no download and should print the MAP and prior-swap checks. `python train.py` downloads Fashion-MNIST into `data/` and takes about twelve minutes on CPU.

# References

1. Du, C., Wang, Y., Song, S., Huang, G. (2025). *BAPE: Learning an Explicit Bayes Classifier for Long-tailed Visual Recognition.* arXiv:2506.23280.
2. Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., Kumar, S. (2021). *Long-tail Learning via Logit Adjustment.* ICLR 2021.
3. Sra, S. (2012). A short note on parameter approximation for von Mises–Fisher distributions. *Computational Statistics*.
4. Mardia, K. V., Jupp, P. E. (2000). *Directional Statistics.* Wiley.
5. Yang, Y., Xie, L., Chen, S., Li, X., Lin, Z., Tao, D. (2022). *Do We Really Need a Learnable Classifier at the End of Deep Neural Network?* NeurIPS 2022. (ETF classifier used as the BAPE prior direction.)
6. Cao, K., Wei, C., Gaidon, A., Arechiga, N., Ma, T. (2019). *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss.* NeurIPS 2019. (Source of the exponential long-tail construction.)
7. Xiao, H., Rasul, K., Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747.
8. Wood, A. T. A. (1994). Simulation of the von Mises–Fisher distribution. *Communications in Statistics — Simulation and Computation*, 23(1), 157–164. (Sampler used only in the synthetic check.)

# Appendix A: `bape.py`

```{.python .numberLines}
<<<BAPE>>>
```

# Appendix B: `train.py`

```{.python .numberLines}
<<<TRAIN>>>
```
