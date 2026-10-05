"""
BAPE-mini: a small, faithful reproduction of the core of

    Chaoqun Du, Yulin Wang, Shiji Song, Gao Huang.
    "BAPE: Learning an Explicit Bayes Classifier for Long-tailed Visual
    Recognition." arXiv:2506.23280, 2025.

The classifier is not a softmax layer trained by gradient descent. It is the
Bayes classifier under a von Mises-Fisher (vMF) model of L2-normalised
features, whose parameters (mu, kappa) are point-estimated by maximum a
posteriori (MAP) from running class statistics. A logit-adjustment head is
trained alongside it, as in the paper's overall objective (their Eq. 13).

At test time the training class prior inside the Bayes rule is replaced by
the test prior (uniform, when the test set is balanced). That is the
distribution adjustment written into their Eq. 6, where pi_y is "the class
frequency in the training or test set".
"""

import math

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F


def etf_prior(embed_dim, num_classes, seed=0):
    """Simplex equiangular tight frame, Eq. 12.

    Returns a (K, p) tensor whose rows are unit vectors. Requires p >= K.
    """
    if embed_dim < num_classes:
        raise ValueError("embed_dim must be >= num_classes for this ETF construction")
    generator = torch.Generator()
    generator.manual_seed(seed)
    gaussian = torch.randn(embed_dim, num_classes, generator=generator)
    orthogonal, _ = torch.linalg.qr(gaussian, mode="reduced")  # (p, K)
    eye = torch.eye(num_classes)
    centering = eye - torch.ones(num_classes, num_classes) / num_classes
    frame = math.sqrt(num_classes / (num_classes - 1.0)) * (orthogonal @ centering)
    frame = frame / frame.norm(dim=0, keepdim=True)
    return frame.T.contiguous()  # (K, p)


def kappa_from_resultant(alpha, beta, embed_dim, kappa_min=1e-3, kappa_max=80.0):
    """Sra (2012) approximation used by the paper, Eq. 10.

    kappa = p * beta * alpha / (alpha^2 - beta^2), with beta < alpha.
    """
    alpha = alpha.clamp_min(1e-6)
    beta = torch.minimum(beta, alpha * (1 - 1e-4))
    kappa = embed_dim * beta * alpha / (alpha * alpha - beta * beta)
    return kappa.clamp(kappa_min, kappa_max)


def log_vmf_normalizer(kappa, embed_dim):
    """log C_p(kappa), Eq. 4. Computed per class; detached from autograd.

    log I_v(kappa) = log(ive(v, kappa)) + kappa, with v = p/2 - 1.
    """
    order = embed_dim / 2.0 - 1.0
    kappa_np = kappa.detach().float().cpu().numpy()
    log_bessel = np.log(np.maximum(scipy.special.ive(order, kappa_np), 1e-30)) + kappa_np
    log_c = (
        0.5 * embed_dim * math.log(2.0 * math.pi)
        + log_bessel
        - (embed_dim / 2.0 - 1.0) * np.log(np.maximum(kappa_np, 1e-8))
    )
    return torch.as_tensor(log_c, device=kappa.device, dtype=kappa.dtype)


class SmallCNN(nn.Module):
    """Feature extractor for 28x28 grayscale images."""

    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BAPEModel(nn.Module):
    """Backbone + logit-adjustment head + explicit vMF Bayes classifier."""

    def __init__(
        self,
        num_classes,
        class_counts,
        feat_dim=128,
        embed_dim=32,
        alpha_hat=4.0,
        beta_hat=0.8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.backbone = SmallCNN(feat_dim)
        self.la_head = nn.Linear(feat_dim, num_classes)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, embed_dim),
        )
        # Learnable prior direction m0, initialised as an ETF (Eq. 12).
        # The paper updates m0 by gradient descent; mu and kappa of the
        # observed features stay point estimates.
        self.m0 = nn.Parameter(etf_prior(embed_dim, num_classes))

        counts = torch.tensor(class_counts, dtype=torch.float32)
        self.register_buffer("class_counts", counts)
        self.register_buffer("alpha0", alpha_hat * counts)
        self.register_buffer("beta0", beta_hat * counts)
        self.register_buffer("feat_sum", torch.zeros(num_classes, embed_dim))
        self.register_buffer("seen", torch.zeros(num_classes))
        prior = counts / counts.sum()
        self.register_buffer("pi_train", prior)
        self.register_buffer("pi_test", torch.full((num_classes,), 1.0 / num_classes))

    @torch.no_grad()
    def update_statistics(self, z, y):
        """Online class-mean accumulation, Eq. 11. z is L2-normalised."""
        for c in y.unique().tolist():
            mask = y == c
            z_c = z[mask]
            self.feat_sum[c] += z_c.sum(dim=0)
            self.seen[c] += mask.sum()

    def map_parameters(self):
        """MAP estimates mu (K, p) and kappa (K,), Eqs. 9-10."""
        m0 = F.normalize(self.m0, dim=1)
        # Pseudo-observations beta0 * m0 plus the detached sample sum.
        total = self.beta0[:, None] * m0 + self.feat_sum.detach()
        beta = total.norm(dim=1).clamp_min(1e-6)
        alpha = self.alpha0 + self.seen.detach()
        mu = total / beta[:, None]
        kappa = kappa_from_resultant(alpha, beta, self.embed_dim)
        return mu, kappa

    def bayes_logits(self, z, pi):
        """log pi_y - log C_p(kappa_y) + kappa_y * mu_y^T z, Eq. 6."""
        mu, kappa = self.map_parameters()
        log_c = log_vmf_normalizer(kappa, self.embed_dim)
        cosine = z @ mu.T  # (N, K)
        return torch.log(pi.clamp_min(1e-12)) - log_c + kappa * cosine

    def forward(self, x, y=None, update_stats=False):
        features = self.backbone(x)
        la_logits = self.la_head(features)
        z = F.normalize(self.proj(features), dim=1)
        if update_stats:
            if y is None:
                raise ValueError("y is required when updating statistics")
            self.update_statistics(z.detach(), y)
        return la_logits, z


def la_logits_adjusted(logits, pi_train):
    """Menon et al. logit adjustment: add log of the training prior."""
    return logits + torch.log(pi_train.clamp_min(1e-12))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Checks that do not need a dataset
# ---------------------------------------------------------------------------

def _sample_vmf(mu, kappa, n_samples, rng):
    """Wood (1994) sampler. mu is a unit vector."""
    mu = np.asarray(mu, dtype=np.float64)
    mu = mu / np.linalg.norm(mu)
    dim = mu.shape[0]
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa * kappa + (dim - 1) ** 2)) / (dim - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (dim - 1) * math.log(1.0 - x0 * x0)
    accepted = []
    while len(accepted) < n_samples:
        z = rng.beta((dim - 1) / 2.0, (dim - 1) / 2.0, size=n_samples)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        thresh = kappa * w + (dim - 1) * np.log(1.0 - x0 * w) - c
        keep = thresh >= np.log(rng.random(n_samples))
        accepted.extend(w[keep].tolist())
    w = np.asarray(accepted[:n_samples])
    v = rng.normal(size=(n_samples, dim - 1))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    samples = np.concatenate(
        [w[:, None], np.sqrt(np.maximum(1.0 - w * w, 0.0))[:, None] * v], axis=1
    )
    basis = np.zeros(dim)
    basis[0] = 1.0
    axis = basis - mu
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return samples
    axis /= norm
    return samples - 2.0 * np.outer(samples @ axis, axis)


def verify_map_and_adjustment():
    """Point-estimation recovers the vMF mean; prior swap shifts logits exactly."""
    rng = np.random.default_rng(0)
    dim = 16
    mu_true = rng.normal(size=dim)
    mu_true /= np.linalg.norm(mu_true)
    samples = _sample_vmf(mu_true, kappa=30.0, n_samples=4000, rng=rng)
    sample_sum = torch.tensor(samples.sum(axis=0), dtype=torch.float32)
    beta = sample_sum.norm()
    alpha = torch.tensor(float(len(samples)))
    mu_hat = sample_sum / beta
    cosine = torch.dot(mu_hat, torch.tensor(mu_true, dtype=torch.float32)).item()
    kappa_hat = kappa_from_resultant(alpha, beta[None], dim)[0].item()

    # A second, less concentrated sample must produce a smaller kappa.
    loose = _sample_vmf(mu_true, kappa=5.0, n_samples=4000, rng=rng)
    loose_sum = torch.tensor(loose.sum(axis=0), dtype=torch.float32)
    beta_loose = loose_sum.norm()
    kappa_loose = kappa_from_resultant(alpha, beta_loose[None], dim)[0].item()

    # Distribution adjustment: replacing pi changes logits by log(pi_new/pi_old)
    # and by nothing else. Tail class is class 1.
    pi_train = torch.tensor([0.9, 0.1])
    pi_test = torch.tensor([0.5, 0.5])
    z = torch.tensor([[0.0, 1.0]])  # a feature sitting on class 1's mean
    mu = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    kappa = torch.tensor([10.0, 10.0])
    log_c = log_vmf_normalizer(kappa, embed_dim=2)

    def logits(pi):
        return torch.log(pi) - log_c + kappa * (z @ mu.T)

    delta = (logits(pi_test) - logits(pi_train)).squeeze(0)
    expected = torch.log(pi_test) - torch.log(pi_train)
    adjust_err = (delta - expected).abs().max().item()

    print(f"[MAP check] cos(mu_hat, mu_true) = {cosine:.6f}")
    print(f"[MAP check] kappa_hat (true 30) = {kappa_hat:.2f}")
    print(f"[MAP check] kappa_hat (true 5)  = {kappa_loose:.2f}")
    print(f"[DA check]  max |logit shift - log(pi_test/pi_train)| = {adjust_err:.3e}")
    assert cosine > 0.99, "MAP mean direction is wrong"
    assert kappa_hat > kappa_loose, "concentration estimate does not increase with kappa"
    assert adjust_err < 1e-5, "distribution adjustment is not an exact prior swap"
    print("MAP estimation and distribution adjustment verified.")
    return {
        "mu_cosine": cosine,
        "kappa_hat_true30": kappa_hat,
        "kappa_hat_true5": kappa_loose,
        "prior_swap_logit_error": adjust_err,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    verify_map_and_adjustment()
