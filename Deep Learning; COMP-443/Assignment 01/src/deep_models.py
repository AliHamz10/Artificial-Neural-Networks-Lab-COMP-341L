"""
Deep learning models for COMP-443 Assignment 01.

We define two convolutional neural networks for Fashion-MNIST:

- build_simple_cnn: a compact baseline CNN.
- build_deeper_cnn: a slightly deeper CNN with more filters.

Both return compiled Keras models; training is handled in the
experiment script.
"""

from __future__ import annotations

from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers


def build_simple_cnn(input_shape: Tuple[int, int, int] = (28, 28, 1), num_classes: int = 10) -> keras.Model:
    """
    A small CNN baseline: Conv -> Conv -> MaxPool -> Dense.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="simple_cnn")
    return model


def build_deeper_cnn(input_shape: Tuple[int, int, int] = (28, 28, 1), num_classes: int = 10) -> keras.Model:
    """
    A deeper CNN with more filters and two conv blocks.
    """
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="deeper_cnn")
    return model


__all__ = ["build_simple_cnn", "build_deeper_cnn"]

