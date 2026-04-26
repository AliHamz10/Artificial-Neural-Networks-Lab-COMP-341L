from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    keras_path: Path
    input_size: tuple[int, int]
    preprocess: Callable[[tf.Tensor], tf.Tensor]


def _project_root() -> Path:
    # .../Assignment 04/streamlit_app/inference.py -> .../Assignment 04
    return Path(__file__).resolve().parents[1]


def build_model_specs() -> dict[str, ModelSpec]:
    root = _project_root()
    models_dir = root / "models"

    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input as mobilenet_preprocess,
    )
    from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
    from tensorflow.keras.applications.efficientnet import (
        preprocess_input as effnet_preprocess,
    )

    input_size = (224, 224)
    return {
        "mobilenetv2": ModelSpec(
            key="mobilenetv2",
            display_name="MobileNetV2 (Transfer Learning)",
            keras_path=models_dir / "mobilenetv2_cifar10_transfer.keras",
            input_size=input_size,
            preprocess=mobilenet_preprocess,
        ),
        "densenet121": ModelSpec(
            key="densenet121",
            display_name="DenseNet121 (Transfer Learning)",
            keras_path=models_dir / "densenet121_cifar10_transfer.keras",
            input_size=input_size,
            preprocess=densenet_preprocess,
        ),
        "efficientnetb0": ModelSpec(
            key="efficientnetb0",
            display_name="EfficientNetB0 (Transfer Learning)",
            keras_path=models_dir / "efficientnetb0_cifar10_transfer.keras",
            input_size=input_size,
            preprocess=effnet_preprocess,
        ),
    }


def load_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_image_rgb(
    img_rgb_uint8: np.ndarray,
    input_size: tuple[int, int],
    preprocess: Callable[[tf.Tensor], tf.Tensor],
) -> tf.Tensor:
    if img_rgb_uint8.ndim != 3 or img_rgb_uint8.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    x = tf.convert_to_tensor(img_rgb_uint8, dtype=tf.float32)
    x = tf.image.resize(x, input_size, method="bilinear")
    x = preprocess(x)
    x = tf.expand_dims(x, axis=0)
    return x


def predict_topk(model: tf.keras.Model, x: tf.Tensor, k: int = 5) -> list[tuple[str, float]]:
    preds = model(x, training=False)
    preds = tf.convert_to_tensor(preds)
    if preds.shape.rank != 2 or preds.shape[0] != 1:
        raise ValueError(f"Unexpected prediction shape: {preds.shape}")

    probs = tf.squeeze(preds, axis=0)
    if probs.dtype != tf.float32:
        probs = tf.cast(probs, tf.float32)
    probs = probs / tf.reduce_sum(probs)

    k = int(min(k, len(CLASS_NAMES)))
    topk = tf.math.top_k(probs, k=k)
    indices = topk.indices.numpy().tolist()
    values = topk.values.numpy().tolist()
    return [(CLASS_NAMES[i], float(v)) for i, v in zip(indices, values)]

