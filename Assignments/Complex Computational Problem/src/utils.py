from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


@dataclass(frozen=True)
class Standardizer:
    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / (self.std_ + 1e-8)

    @staticmethod
    def fit(x: np.ndarray) -> "Standardizer":
        mean_ = x.mean(axis=0, keepdims=True)
        std_ = x.std(axis=0, keepdims=True)
        return Standardizer(mean_=mean_, std_=std_)


def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    split = int(x.shape[0] * (1.0 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
):
    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple PCA (NumPy SVD) returning 2D projection.
    Returns (x2, components, mean).
    """
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    # SVD on covariance-equivalent: right singular vectors of centered data
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:2]  # (2, D)
    x2 = xc @ components.T
    return x2, components, mean


def apply_pca(x: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - mean) @ components.T

