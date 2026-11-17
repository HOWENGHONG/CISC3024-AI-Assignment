"""
Compare pixel PCA, Fisherfaces, 2DPCA, 2DLDA, and the 2024 cascades
(2DPCA+LDA, 2DPCA+2DLDA, 2DPCA+PCA+LDA) on a small-sample Fashion-MNIST split.

The split is deliberately small: 20 training images per class, so a vectorised
28x28 image (784 dimensions) makes the within-class scatter singular. That is
the small-sample-size failure the 2024 paper is written to avoid.
"""

import json
import struct
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subspace import (
    fisher_criterion,
    lda_basis,
    nearest_class_mean,
    pca_basis,
    pixel_within_rank,
    project_pca,
    project_twodlda,
    project_twodpca,
    reconstruct_pca,
    reconstruct_twodpca,
    twodlda_basis,
    twodpca_basis,
    verify_identities,
)

SEED = 0
TRAIN_PER_CLASS = 20
TEST_PER_CLASS = 200
FINAL_DIM = 9  # C - 1 for 10 classes
PCA_COMPONENTS = 40
TWODPCA_COMPONENTS = 5  # feature length 28*5 = 140 < 200 - 10

FASHION_DIR = Path(__file__).resolve().parents[1] / "assignment2" / "data" / "FashionMNIST" / "raw"
CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_idx_images(path):
    with open(path, "rb") as handle:
        _, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def load_idx_labels(path):
    with open(path, "rb") as handle:
        _, count = struct.unpack(">II", handle.read(8))
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count)


def subsample(images, labels, per_class, rng):
    chosen = []
    for c in range(10):
        pool = np.where(labels == c)[0]
        chosen.append(rng.choice(pool, size=per_class, replace=False))
    index = np.concatenate(chosen)
    rng.shuffle(index)
    return images[index], labels[index]


def accuracy(pred, target):
    return float((pred == target).mean())


def per_class_accuracy(pred, target):
    return [float((pred[target == c] == c).mean()) for c in range(10)]


def fit_predict_pca(train, test, y_train, y_test, components):
    vectors_train = train.reshape(len(train), -1).astype(np.float64) / 255.0
    vectors_test = test.reshape(len(test), -1).astype(np.float64) / 255.0
    mean, axes, _ = pca_basis(vectors_train, components)
    z_train = project_pca(vectors_train, mean, axes)
    z_test = project_pca(vectors_test, mean, axes)
    pred = nearest_class_mean(z_train, y_train, z_test, np.arange(10))
    return accuracy(pred, y_test), per_class_accuracy(pred, y_test), mean, axes, vectors_train, vectors_test


def fit_predict_fisher(train, test, y_train, y_test):
    vectors_train = train.reshape(len(train), -1).astype(np.float64) / 255.0
    vectors_test = test.reshape(len(test), -1).astype(np.float64) / 255.0
    # PCA down to n - C, the largest safe dimension before LDA (Belhumeur et al.).
    n_pca = vectors_train.shape[0] - 10
    mean, axes, _ = pca_basis(vectors_train, n_pca)
    z_train = project_pca(vectors_train, mean, axes)
    z_test = project_pca(vectors_test, mean, axes)
    lda = lda_basis(z_train, y_train, FINAL_DIM)
    pred = nearest_class_mean(z_train @ lda, y_train, z_test @ lda, np.arange(10))
    criterion = fisher_criterion(z_train, y_train, lda)
    return accuracy(pred, y_test), per_class_accuracy(pred, y_test), criterion


def fit_predict_twodpca(train, test, y_train, y_test, components):
    images_train = train.astype(np.float64) / 255.0
    images_test = test.astype(np.float64) / 255.0
    mean, axes, eigenvalues = twodpca_basis(images_train, components)
    z_train = project_twodpca(images_train, mean, axes)
    z_test = project_twodpca(images_test, mean, axes)
    pred = nearest_class_mean(z_train, y_train, z_test, np.arange(10))
    return accuracy(pred, y_test), per_class_accuracy(pred, y_test), mean, axes, eigenvalues, z_train, z_test, images_train, images_test


def fit_predict_twodpca_lda(z_train, z_test, y_train, y_test):
    lda = lda_basis(z_train, y_train, FINAL_DIM)
    pred = nearest_class_mean(z_train @ lda, y_train, z_test @ lda, np.arange(10))
    criterion = fisher_criterion(z_train, y_train, lda)
    return accuracy(pred, y_test), per_class_accuracy(pred, y_test), criterion


def fit_predict_twodpca_pca_lda(z_train, z_test, y_train, y_test):
    """The speed cascade in the 2024 paper: 2DPCA, then ordinary PCA, then LDA."""
    n_pca = min(40, z_train.shape[0] - 10, z_train.shape[1])
    mean, axes, _ = pca_basis(z_train, n_pca)
    u_train = project_pca(z_train, mean, axes)
    u_test = project_pca(z_test, mean, axes)
    lda = lda_basis(u_train, y_train, FINAL_DIM)
    pred = nearest_class_mean(u_train @ lda, y_train, u_test @ lda, np.arange(10))
    return accuracy(pred, y_test), per_class_accuracy(pred, y_test)


def fit_predict_twodlda(train, test, y_train, y_test, components):
    images_train = train.astype(np.float64) / 255.0
    images_test = test.astype(np.float64) / 255.0
    axes, eigenvalues = twodlda_basis(images_train, y_train, components)
    z_train = project_twodlda(images_train, axes)
    z_test = project_twodlda(images_test, axes)
    pred = nearest_class_mean(z_train, y_train, z_test, np.arange(10))
    return accuracy(pred, y_test), per_class_accuracy(pred, y_test), float(eigenvalues[0])


def fit_predict_twodpca_twodlda(train, test, y_train, y_test, twodpca_k, twodlda_k):
    images_train = train.astype(np.float64) / 255.0
    images_test = test.astype(np.float64) / 255.0
    mean, axes, _ = twodpca_basis(images_train, twodpca_k)
    reduced_train = np.einsum("nij,jk->nik", images_train - mean, axes)
    reduced_test = np.einsum("nij,jk->nik", images_test - mean, axes)
    lda_axes, _ = twodlda_basis(reduced_train, y_train, twodlda_k)
    z_train = project_twodlda(reduced_train, lda_axes)
    z_test = project_twodlda(reduced_test, lda_axes)
    pred = nearest_class_mean(z_train, y_train, z_test, np.arange(10))
    return accuracy(pred, y_test), per_class_accuracy(pred, y_test)


def main():
    started = time.perf_counter()
    checks = verify_identities()
    rng = np.random.default_rng(SEED)
    train_images = load_idx_images(FASHION_DIR / "train-images-idx3-ubyte")
    train_labels = load_idx_labels(FASHION_DIR / "train-labels-idx1-ubyte")
    test_images = load_idx_images(FASHION_DIR / "t10k-images-idx3-ubyte")
    test_labels = load_idx_labels(FASHION_DIR / "t10k-labels-idx1-ubyte")
    x_train, y_train = subsample(train_images, train_labels, TRAIN_PER_CLASS, rng)
    x_test, y_test = subsample(test_images, test_labels, TEST_PER_CLASS, rng)
    print(f"train {x_train.shape} test {x_test.shape}")

    rank, eigenvalues = pixel_within_rank(x_train.astype(np.float64) / 255.0, y_train)
    print(f"pixel within-class scatter rank {rank} / {28 * 28}, smallest eigenvalue {eigenvalues.min():.3e}")

    pca_curve = []
    for k in (5, 10, 20, 40, 80):
        acc, _, *_ = fit_predict_pca(x_train, x_test, y_train, y_test, k)
        pca_curve.append({"components": k, "accuracy": acc})
        print(f"  PCA-{k:3d}  {acc:.4f}")

    twod_curve = []
    for k in (1, 2, 3, 5, 8, 12):
        acc, _, *_ = fit_predict_twodpca(x_train, x_test, y_train, y_test, k)
        twod_curve.append({"components": k, "feature_dim": 28 * k, "accuracy": acc})
        print(f"  2DPCA-{k:2d}  dim {28 * k:4d}  {acc:.4f}")

    pca_acc, pca_per_class, pca_mean, pca_axes, vec_train, vec_test = fit_predict_pca(
        x_train, x_test, y_train, y_test, PCA_COMPONENTS
    )
    fisher_acc, fisher_per_class, fisher_j = fit_predict_fisher(x_train, x_test, y_train, y_test)
    mean40, axes40, _ = pca_basis(vec_train, PCA_COMPONENTS)
    pca_lda_acc, pca_lda_per_class, pca_lda_j = fit_predict_twodpca_lda(
        project_pca(vec_train, mean40, axes40),
        project_pca(vec_test, mean40, axes40),
        y_train,
        y_test,
    )
    twod_acc, twod_per_class, img_mean, img_axes, img_eigs, z_train, z_test, img_train, img_test = fit_predict_twodpca(
        x_train, x_test, y_train, y_test, TWODPCA_COMPONENTS
    )
    cascade_acc, cascade_per_class, cascade_j = fit_predict_twodpca_lda(z_train, z_test, y_train, y_test)
    speed_acc, speed_per_class = fit_predict_twodpca_pca_lda(z_train, z_test, y_train, y_test)
    twodlda_acc, twodlda_per_class, twodlda_top = fit_predict_twodlda(
        x_train, x_test, y_train, y_test, FINAL_DIM
    )
    both_acc, both_per_class = fit_predict_twodpca_twodlda(
        x_train, x_test, y_train, y_test, TWODPCA_COMPONENTS, min(FINAL_DIM, TWODPCA_COMPONENTS)
    )

    # Same-dimension PCA criterion, so the LDA gain is not an artefact of dimension.
    z_for_criterion = project_pca(vec_train, pca_mean, pca_axes[:, :FINAL_DIM])
    # fisher_criterion needs the basis in the space of z. Use the identity on the PCA scores.
    pca_j = fisher_criterion(z_for_criterion, y_train, np.eye(FINAL_DIM))

    print(f"PCA-{PCA_COMPONENTS} {pca_acc:.4f}")
    print(f"PCA-{PCA_COMPONENTS}+LDA {pca_lda_acc:.4f}  Fisher {pca_lda_j:.3f}")
    print(f"Fisherfaces {fisher_acc:.4f}  Fisher {fisher_j:.3f}")
    print(f"2DPCA-{TWODPCA_COMPONENTS} {twod_acc:.4f}")
    print(f"2DPCA+LDA {cascade_acc:.4f}  Fisher {cascade_j:.3f}")
    print(f"2DPCA+PCA+LDA {speed_acc:.4f}")
    print(f"2DLDA {twodlda_acc:.4f}  top eigenvalue {twodlda_top:.3f}")
    print(f"2DPCA+2DLDA {both_acc:.4f}")

    plot_summary(pca_curve, twod_curve, {
        "PCA-40": pca_acc,
        "PCA-40+LDA": pca_lda_acc,
        "Fisherfaces": fisher_acc,
        "2DPCA-5": twod_acc,
        "2DPCA+LDA": cascade_acc,
        "2DPCA+PCA+LDA": speed_acc,
        "2DLDA": twodlda_acc,
        "2DPCA+2DLDA": both_acc,
    })
    plot_reconstruction(vec_test, pca_mean, pca_axes, img_test, img_mean, img_axes)
    plot_spectrum(eigenvalues, img_eigs)

    results = {
        "dataset": "Fashion-MNIST small-sample split",
        "train_per_class": TRAIN_PER_CLASS,
        "test_per_class": TEST_PER_CLASS,
        "image_shape": [28, 28],
        "pixel_dim": 784,
        "within_class_rank": rank,
        "within_class_dim": 784,
        "smallest_within_eigenvalue": float(eigenvalues.min()),
        "largest_within_eigenvalue": float(eigenvalues.max()),
        "checks": checks,
        "pca_curve": pca_curve,
        "twodpca_curve": twod_curve,
        "methods": {
            "pca40": {"accuracy": pca_acc, "per_class": pca_per_class, "fisher": pca_j},
            "pca40_lda": {"accuracy": pca_lda_acc, "per_class": pca_lda_per_class, "fisher": pca_lda_j},
            "fisherfaces": {"accuracy": fisher_acc, "per_class": fisher_per_class, "fisher": fisher_j},
            "twodpca5": {"accuracy": twod_acc, "per_class": twod_per_class},
            "twodpca_lda": {"accuracy": cascade_acc, "per_class": cascade_per_class, "fisher": cascade_j},
            "twodpca_pca_lda": {"accuracy": speed_acc, "per_class": speed_per_class},
            "twodlda": {"accuracy": twodlda_acc, "per_class": twodlda_per_class, "top_eigenvalue": twodlda_top},
            "twodpca_twodlda": {"accuracy": both_acc, "per_class": both_per_class},
        },
        "seconds": time.perf_counter() - started,
    }
    Path("results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"done in {results['seconds']:.1f}s")


def plot_summary(pca_curve, twod_curve, finals):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot([r["components"] for r in pca_curve], [r["accuracy"] for r in pca_curve], "o-", label="PCA")
    axes[0].plot([r["feature_dim"] for r in twod_curve], [r["accuracy"] for r in twod_curve], "s-", label="2DPCA")
    axes[0].set_xlabel("feature dimension")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Nearest class mean after PCA or 2DPCA")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    names = list(finals)
    axes[1].bar(range(len(names)), [finals[n] for n in names], color="tab:blue")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("test accuracy")
    axes[1].set_title("Test accuracy of each pipeline")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("accuracy_comparison.png", dpi=150)
    plt.close(fig)


def plot_reconstruction(vectors, pca_mean, pca_axes, images, img_mean, img_axes):
    scores = project_pca(vectors[:8], pca_mean, pca_axes[:, :20])
    pca_recon = reconstruct_pca(scores, pca_mean, pca_axes[:, :20]).reshape(-1, 28, 28)
    features = project_twodpca(images[:8], img_mean, img_axes[:, :5])
    twod_recon = reconstruct_twodpca(features, img_mean, img_axes[:, :5])
    originals = images[:8]
    fig, axes = plt.subplots(3, 8, figsize=(10, 4.2))
    for i in range(8):
        axes[0, i].imshow(originals[i], cmap="gray")
        axes[1, i].imshow(pca_recon[i], cmap="gray")
        axes[2, i].imshow(twod_recon[i], cmap="gray")
        for row in range(3):
            axes[row, i].axis("off")
    axes[0, 0].set_ylabel("original", rotation=90, size=8)
    axes[1, 0].set_ylabel("PCA-20", rotation=90, size=8)
    axes[2, 0].set_ylabel("2DPCA-5", rotation=90, size=8)
    fig.tight_layout()
    fig.savefig("reconstructions.png", dpi=150)
    plt.close(fig)


def plot_spectrum(within_eigs, twod_eigs):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(np.sort(within_eigs)[::-1] + 1e-18)
    axes[0].set_title("Pixel within-class eigenvalues")
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("eigenvalue")
    axes[0].grid(alpha=0.3)
    axes[1].plot(twod_eigs)
    axes[1].set_title("2DPCA covariance eigenvalues")
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("eigenvalue")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("spectra.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
