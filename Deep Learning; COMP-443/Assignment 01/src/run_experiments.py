"""
COMP-443 Assignment 01 experiments.

This script:
- Loads Fashion-MNIST and prepares train/val/test splits.
- Trains two classical baselines (Logistic Regression, Random Forest).
- Trains two deep CNNs (simple and deeper).
- Saves learning-curve plots and confusion matrices under figures/.
- Prints a comparison table at the end for easy inclusion in the report.

Usage (from repo root):
  ./venv/bin/python3 "Deep Learning; COMP-443/Assignment 01/src/run_experiments.py" --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras

# Allow running this file directly (not as a package module)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data import load_fashion_mnist_splits
from baseline_models import (
    BaselineResult,
    as_dict as baseline_as_dict,
    train_logistic_regression,
    train_random_forest,
)
from deep_models import build_simple_cnn, build_deeper_cnn


ROOT_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def plot_history(history: keras.callbacks.History, title: str, filename: str) -> None:
    """Plot loss and accuracy curves for a Keras training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="train loss")
    ax1.plot(history.history["val_loss"], label="val loss")
    ax1.set_title(f"{title} — loss")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["accuracy"], label="train acc")
    ax2.plot(history.history["val_accuracy"], label="val acc")
    ax2.set_title(f"{title} — accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def save_confusion_matrix(cm: np.ndarray, classes: List[str], title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def train_cnn(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    name: str,
) -> Dict[str, float]:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )
    plot_history(history, name, f"{name.lower().replace(' ', '_')}_curves.png")
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    return {"name": name, "val_loss": float(val_loss), "val_accuracy": float(val_acc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COMP-443 Assignment 01 experiments.")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for deep models (default: 50)")
    args = parser.parse_args()

    print("Loading Fashion-MNIST and preparing splits...")
    splits = load_fashion_mnist_splits()

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # ---------- Classical baselines ----------
    print("\n=== Classical baselines ===")
    lr_res: BaselineResult = train_logistic_regression(
        splits.x_train_flat, splits.y_train, splits.x_val_flat, splits.y_val
    )
    rf_res: BaselineResult = train_random_forest(
        splits.x_train_flat, splits.y_train, splits.x_val_flat, splits.y_val
    )

    save_confusion_matrix(
        lr_res.confusion, class_names, "Logistic Regression — Confusion (val)", "logreg_confusion.png"
    )
    save_confusion_matrix(
        rf_res.confusion, class_names, "Random Forest — Confusion (val)", "rf_confusion.png"
    )

    # ---------- Deep models ----------
    print("\n=== Deep learning models ===")
    simple_cnn = build_simple_cnn()
    deeper_cnn = build_deeper_cnn()

    simple_stats = train_cnn(
        simple_cnn,
        splits.x_train_img,
        splits.y_train,
        splits.x_val_img,
        splits.y_val,
        epochs=args.epochs,
        name="Simple CNN",
    )
    deeper_stats = train_cnn(
        deeper_cnn,
        splits.x_train_img,
        splits.y_train,
        splits.x_val_img,
        splits.y_val,
        epochs=args.epochs,
        name="Deeper CNN",
    )

    # ---------- Collect results ----------
    results = [
        baseline_as_dict(lr_res),
        baseline_as_dict(rf_res),
        {
            "name": simple_stats["name"],
            "accuracy": simple_stats["val_accuracy"],
            "val_loss": simple_stats["val_loss"],
        },
        {
            "name": deeper_stats["name"],
            "accuracy": deeper_stats["val_accuracy"],
            "val_loss": deeper_stats["val_loss"],
        },
    ]

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE (validation set)")
    print("=" * 60)
    print(f"{'Model':<20} {'Val Loss':>10} {'Val Acc':>10}")
    print("-" * 60)
    for r in results:
        name = r["name"]
        loss = r.get("val_loss", 0.0)
        acc = r["accuracy"]
        print(f"{name:<20} {loss:>10.4f} {acc:>10.4f}")
    print("=" * 60)

    # Save results JSON for the report
    out_json = os.path.join(REPORTS_DIR, "assignment01_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results JSON to {out_json}")


if __name__ == "__main__":
    main()

