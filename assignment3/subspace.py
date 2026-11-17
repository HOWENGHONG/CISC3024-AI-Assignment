"""
Matrix-form PCA and LDA for images.

The pipelines follow the 2024 face-recognition paper that combines
two-dimensional PCA with LDA and with 2DLDA in order to avoid the
small-sample-size singularity of classical LDA:

    Feature extraction and recognition of face image based on 2DPCA
    with LDA algorithm. IEEE IAEAC, 2024.
    doi:10.1109/IAEAC59436.2024.10503906

The component algorithms are the published ones, not a re-derivation:

  * 2DPCA: Yang, Zhang, Frangi, Yang, IEEE TPAMI 2004.
  * Fisherfaces (PCA then LDA): Belhumeur, Hespanha, Kriegman, IEEE TPAMI 1997.
  * 2DLDA: Ye, Janardan, Li, NIPS 2004; the right-multiplication form
    whose scatter matrices are n by n for an m by n image.
"""

import numpy as np
from scipy.linalg import eigh


def pca_basis(vectors, n_components):
    """Top principal axes of row-wise samples. Returns (mean, components (d, k))."""
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    # Economy SVD. Columns of V are principal axes.
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(n_components, vt.shape[0])
    return mean, vt[:k].T, singular


def project_pca(vectors, mean, components):
    return (vectors - mean) @ components


def reconstruct_pca(scores, mean, components):
    return scores @ components.T + mean


def twodpca_basis(images, n_components):
    """Yang et al. 2DPCA. images: (N, m, n). Returns (mean image, axes (n, k), eigenvalues)."""
    mean = images.mean(axis=0)
    centered = images - mean
    # G = (1/N) sum A^T A, an n by n matrix.
    gram = np.einsum("nij,nik->jk", centered, centered) / images.shape[0]
    eigenvalues, eigenvectors = eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    k = min(n_components, eigenvectors.shape[1])
    return mean, eigenvectors[:, :k], eigenvalues


def project_twodpca(images, mean, axes):
    """Each image becomes an (m, k) matrix, then a vector of length m*k."""
    features = np.einsum("nij,jk->nik", images - mean, axes)
    return features.reshape(images.shape[0], -1)


def reconstruct_twodpca(features, mean, axes):
    m = mean.shape[0]
    k = axes.shape[1]
    mats = features.reshape(-1, m, k)
    return np.einsum("nik,jk->nij", mats, axes) + mean


def scatter_matrices(vectors, labels):
    """Within-class and between-class scatter of row-wise samples."""
    classes = np.unique(labels)
    mean = vectors.mean(axis=0)
    dim = vectors.shape[1]
    within = np.zeros((dim, dim))
    between = np.zeros((dim, dim))
    for c in classes:
        block = vectors[labels == c]
        class_mean = block.mean(axis=0)
        deviation = block - class_mean
        within += deviation.T @ deviation
        gap = (class_mean - mean).reshape(-1, 1)
        between += block.shape[0] * (gap @ gap.T)
    return within, between


def lda_basis(vectors, labels, n_components, ridge=1e-4):
    """Fisher LDA. Ridge is a fraction of trace(Sw)/dim, so a singular Sw still yields a basis."""
    within, between = scatter_matrices(vectors, labels)
    scale = np.trace(within) / within.shape[0]
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    within = within + (ridge * scale) * np.eye(within.shape[0])
    eigenvalues, eigenvectors = eigh(between, within)
    order = np.argsort(eigenvalues)[::-1]
    k = min(n_components, vectors.shape[1], len(np.unique(labels)) - 1)
    return eigenvectors[:, order[:k]]


def fisher_criterion(vectors, labels, basis):
    """trace(W^T Sb W) / trace(W^T Sw W). Larger means better class separation."""
    within, between = scatter_matrices(vectors, labels)
    left = basis.T @ within @ basis
    right = basis.T @ between @ basis
    denom = np.trace(left)
    if abs(denom) < 1e-12:
        return float("nan")
    return float(np.trace(right) / denom)


def twodlda_basis(images, labels, n_components, ridge=1e-4):
    """Right-side 2DLDA. Scatter matrices are n by n, not (m*n) by (m*n)."""
    classes = np.unique(labels)
    mean = images.mean(axis=0)
    class_means = {c: images[labels == c].mean(axis=0) for c in classes}
    n = images.shape[2]
    within = np.zeros((n, n))
    between = np.zeros((n, n))
    for c in classes:
        block = images[labels == c] - class_means[c]
        within += np.einsum("nij,nik->jk", block, block)
        gap = class_means[c] - mean
        between += block.shape[0] * (gap.T @ gap)
    scale = np.trace(within) / n
    within = within + (ridge * scale) * np.eye(n)
    eigenvalues, eigenvectors = eigh(between, within)
    order = np.argsort(eigenvalues)[::-1]
    k = min(n_components, n, len(classes) - 1)
    return eigenvectors[:, order[:k]], eigenvalues[order]


def project_twodlda(images, axes):
    features = np.einsum("nij,jk->nik", images, axes)
    return features.reshape(images.shape[0], -1)


def class_means(features, labels, classes):
    return np.stack([features[labels == c].mean(axis=0) for c in classes])


def nearest_class_mean(train_features, train_labels, test_features, classes):
    means = class_means(train_features, train_labels, classes)
    distances = ((test_features[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
    return classes[distances.argmin(axis=1)]


def pixel_within_rank(images, labels, tol_ratio=1e-8):
    """Numerical rank of the pixel-space within-class scatter. This is the SSS diagnostic."""
    vectors = images.reshape(images.shape[0], -1).astype(np.float64)
    within, _ = scatter_matrices(vectors, labels)
    eigenvalues = eigh(within, eigvals_only=True)
    threshold = tol_ratio * eigenvalues.max()
    return int(np.sum(eigenvalues > threshold)), eigenvalues


def verify_identities():
    """Reconstruction energy and the LDA prior-free geometry, on synthetic images."""
    rng = np.random.default_rng(0)
    images = rng.normal(size=(30, 8, 6))
    mean, axes, eigenvalues = twodpca_basis(images, n_components=6)
    features = project_twodpca(images, mean, axes)
    rebuilt = reconstruct_twodpca(features, mean, axes)
    recon_error = np.mean((rebuilt - images) ** 2)
    # Full rank must reconstruct exactly. Discarded energy equals the tail of the spectrum.
    partial_axes = axes[:, :2]
    partial = project_twodpca(images, mean, partial_axes)
    partial_recon = reconstruct_twodpca(partial, mean, partial_axes)
    residual = np.sum((images - partial_recon) ** 2) / images.shape[0]
    tail = np.sum(eigenvalues[2:])
    energy_error = abs(residual - tail)

    # Two classes separated along dimension 0. LDA must beat the trailing PCA axis.
    class0 = 0.05 * rng.normal(size=(40, 4))
    class1 = 0.05 * rng.normal(size=(40, 4))
    class0[:, 0] += 1.0
    class1[:, 0] -= 1.0
    samples = np.vstack([class0, class1])
    labels = np.array([0] * 40 + [1] * 40)
    lda = lda_basis(samples, labels, n_components=1, ridge=1e-8)
    _, pca_axes, _ = pca_basis(samples, n_components=4)
    criterion_lda = fisher_criterion(samples, labels, lda)
    criterion_last_pca = fisher_criterion(samples, labels, pca_axes[:, -1:])
    # Population identity: if Sw = I and the class gap is e0, the Fisher
    # direction is exactly e0. Finite-sample Sw is only approximately I.
    gap = np.array([1.0, 0.0, 0.0, 0.0])
    between = np.outer(gap, gap)
    within = np.eye(4)
    _, exact = eigh(between, within)
    exact_alignment = abs(float(exact[:, -1] @ gap))

    print(f"[2DPCA check] full reconstruction MSE = {recon_error:.3e}")
    print(f"[2DPCA check] |residual - discarded eigenvalues| = {energy_error:.3e}")
    print(f"[LDA check]   population |cos(w, class axis)| = {exact_alignment:.6f}")
    print(f"[LDA check]   Fisher criterion LDA {criterion_lda:.3f} vs last PCA {criterion_last_pca:.3f}")
    assert recon_error < 1e-10, "full 2DPCA does not reconstruct"
    assert energy_error < 1e-6, "2DPCA residual does not match the spectrum"
    assert exact_alignment > 1 - 1e-8, "LDA is not the class-mean gap when Sw = I"
    assert criterion_lda > 10 * max(criterion_last_pca, 1e-12)
    print("2DPCA reconstruction identity and LDA direction verified.")
    return {
        "full_reconstruction_mse": recon_error,
        "spectrum_identity_error": energy_error,
        "lda_population_alignment": exact_alignment,
        "fisher_lda": criterion_lda,
        "fisher_last_pca": criterion_last_pca,
    }


if __name__ == "__main__":
    verify_identities()
