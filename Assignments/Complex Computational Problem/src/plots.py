from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def savefig(path: str, dpi: int = 160) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_decision_boundary_2d(
    x2: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    title: str,
    path: str | None = None,
) -> None:
    """
    x2: (N,2) features, w: (3,) bias + 2 weights for boundary: w0 + w1*x + w2*y = 0
    """
    x2 = np.asarray(x2)
    y = np.asarray(y)

    plt.figure(figsize=(6, 5))
    plt.scatter(x2[:, 0], x2[:, 1], c=y, s=8, cmap="coolwarm", alpha=0.85)

    x_min, x_max = x2[:, 0].min() - 0.5, x2[:, 0].max() + 0.5
    xs = np.linspace(x_min, x_max, 200)
    if abs(w[2]) < 1e-12:
        ys = np.zeros_like(xs)
    else:
        ys = -(w[0] + w[1] * xs) / w[2]
    plt.plot(xs, ys, "k-", lw=2)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    if path is not None:
        savefig(path)


def plot_curves(history: dict[str, list[float]], title: str, path: str | None = None) -> None:
    plt.figure(figsize=(7, 4))
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    if path is not None:
        savefig(path)


def plot_confusion_matrix(cm: np.ndarray, title: str, path: str | None = None) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if path is not None:
        savefig(path)


def plot_activation_curves(path: str | None = None) -> None:
    xs = np.linspace(-5, 5, 600)
    sigmoid = 1.0 / (1.0 + np.exp(-xs))
    tanh = np.tanh(xs)
    relu = np.maximum(0, xs)
    dsigmoid = sigmoid * (1 - sigmoid)
    dtanh = 1 - tanh**2
    drelu = (xs > 0).astype(float)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(xs, sigmoid, label="Sigmoid")
    plt.plot(xs, tanh, label="Tanh")
    plt.plot(xs, relu, label="ReLU")
    plt.title("Activations")
    plt.legend()
    plt.grid(True, alpha=0.25)

    plt.subplot(1, 2, 2)
    plt.plot(xs, dsigmoid, label="Sigmoid'")
    plt.plot(xs, dtanh, label="Tanh'")
    plt.plot(xs, drelu, label="ReLU'")
    plt.title("Gradients")
    plt.legend()
    plt.grid(True, alpha=0.25)

    if path is not None:
        savefig(path)

