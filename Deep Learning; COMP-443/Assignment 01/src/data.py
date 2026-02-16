"""
Data loading and preprocessing for COMP-443 Assignment 01.

We use the Fashion-MNIST dataset (10-way image classification) as a
reasonably sized, standard benchmark suitable for both classical
machine-learning models and deep neural networks.

This module exposes a single function:

- load_fashion_mnist_splits(...)

which returns (x_train, y_train, x_val, y_val, x_test, y_test) in
both flattened (for classical models) and image (for deep models)
formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from tensorflow import keras


@dataclass
class FashionMNISTSplits:
    # For classical models (scikit-learn)
    x_train_flat: np.ndarray
    x_val_flat: np.ndarray
    x_test_flat: np.ndarray

    # For deep models (CNNs)
    x_train_img: np.ndarray
    x_val_img: np.ndarray
    x_test_img: np.ndarray

    # Labels (shared)
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def load_fashion_mnist_splits(
    val_fraction: float = 0.2,
    seed: int = 42,
) -> FashionMNISTSplits:
    """
    Load Fashion-MNIST and create train/validation/test splits.

    - Images are in shape (28, 28) with integer pixel values [0, 255].
    - We normalise them to [0, 1] for deep models.
    - For classical models we flatten each image to a 784-dim vector.
    """

    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalise to [0, 1] for deep models
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Create a validation split from the original training set
    rng = np.random.default_rng(seed)
    n_train = x_train_full.shape[0]
    n_val = int(val_fraction * n_train)

    indices = np.arange(n_train)
    rng.shuffle(indices)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]
    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]

    # For CNNs: add channel dimension (grayscale)
    x_train_img = np.expand_dims(x_train, -1)  # (N, 28, 28, 1)
    x_val_img = np.expand_dims(x_val, -1)
    x_test_img = np.expand_dims(x_test, -1)

    # For classical models: flatten
    x_train_flat = x_train.reshape((x_train.shape[0], -1))
    x_val_flat = x_val.reshape((x_val.shape[0], -1))
    x_test_flat = x_test.reshape((x_test.shape[0], -1))

    return FashionMNISTSplits(
        x_train_flat=x_train_flat,
        x_val_flat=x_val_flat,
        x_test_flat=x_test_flat,
        x_train_img=x_train_img,
        x_val_img=x_val_img,
        x_test_img=x_test_img,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


__all__ = ["FashionMNISTSplits", "load_fashion_mnist_splits"]

