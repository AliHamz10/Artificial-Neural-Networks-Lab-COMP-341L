from __future__ import annotations

import numpy as np

from .utils import Standardizer, train_val_split


def load_mnist(flatten: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads MNIST using TensorFlow/Keras (allowed for loading only).
    Returns float32 images scaled to [0, 1].
    """
    try:
        from tensorflow.keras.datasets import mnist  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is required only to *load* MNIST in Colab. "
            "In Google Colab it is preinstalled."
        ) from exc

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    return x_train, y_train.astype(int), x_test, y_test.astype(int)


def mnist_binary_subset(
    x: np.ndarray,
    y: np.ndarray,
    digit_a: int,
    digit_b: int,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (y == digit_a) | (y == digit_b)
    x_sub = x[mask]
    y_sub = y[mask]
    y_bin = (y_sub == digit_b).astype(int)  # digit_a -> 0, digit_b -> 1
    if max_samples is not None and x_sub.shape[0] > max_samples:
        x_sub = x_sub[:max_samples]
        y_bin = y_bin[:max_samples]
    return x_sub.astype(np.float32), y_bin.astype(int)


def standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, Standardizer]:
    scaler = Standardizer.fit(x_train)
    return scaler.transform(x_train).astype(np.float32), scaler.transform(x_test).astype(np.float32), scaler


def make_mnist_splits(
    val_ratio: float = 0.1,
    standardize: bool = True,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    x_train, y_train, x_test, y_test = load_mnist(flatten=True)
    x_tr, y_tr, x_val, y_val = train_val_split(x_train, y_train, val_ratio=val_ratio, seed=seed)

    if standardize:
        x_tr_s, x_val_s, scaler = standardize_train_test(x_tr, x_val)
        x_test_s = scaler.transform(x_test).astype(np.float32)
        return {
            "x_train": x_tr_s,
            "y_train": y_tr,
            "x_val": x_val_s,
            "y_val": y_val,
            "x_test": x_test_s,
            "y_test": y_test,
        }

    return {
        "x_train": x_tr.astype(np.float32),
        "y_train": y_tr,
        "x_val": x_val.astype(np.float32),
        "y_val": y_val,
        "x_test": x_test.astype(np.float32),
        "y_test": y_test,
    }

