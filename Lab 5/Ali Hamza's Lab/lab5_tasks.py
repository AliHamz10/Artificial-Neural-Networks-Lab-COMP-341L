"""
Lab 05: Dropout, Batch Normalization, and Optimizers (Lab 05.pdf).
Dataset: Breast Cancer (sklearn). Tasks 1–6: baseline, dropout, batchnorm,
combined, SGD vs Adam, learning rate sensitivity.
Run from project root: ./venv/bin/python3 "Lab 5/Ali Hamza's Lab/lab5_tasks.py"
Or from this folder: ../../venv/bin/python3 lab5_tasks.py
Quick test: ../../venv/bin/python3 lab5_tasks.py --epochs 5
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(LAB_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Data (Breast Cancer) ---
def get_data():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val, X_train.shape[1]

BATCH_SIZE = 32
INPUT_DIM = 30  # breast_cancer features

def build_baseline(input_dim=INPUT_DIM):
    """Task 1: No Dropout, No BatchNorm."""
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="baseline")

def build_dropout(input_dim=INPUT_DIM, rate=0.3):
    """Task 2: Add Dropout."""
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(rate),
        layers.Dense(64, activation="relu"),
        layers.Dropout(rate),
        layers.Dense(32, activation="relu"),
        layers.Dropout(rate),
        layers.Dense(1, activation="sigmoid"),
    ], name="dropout")

def build_batchnorm(input_dim=INPUT_DIM):
    """Task 3: Add BatchNorm (after Dense, before activation)."""
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="batchnorm")

def build_dropout_batchnorm(input_dim=INPUT_DIM, rate=0.3):
    """Task 4: Dropout + BatchNorm."""
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(rate),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(rate),
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(rate),
        layers.Dense(1, activation="sigmoid"),
    ], name="dropout_batchnorm")

def train_model(model, X_train, y_train, X_val, y_val, optimizer="adam", lr=0.001, epochs=50, verbose=0):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr) if optimizer == "adam" else keras.optimizers.SGD(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=verbose,
    )
    return hist

def plot_history(hist, title, filename, task_label=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist.history["loss"], label="Train loss")
    axes[0].plot(hist.history["val_loss"], label="Val loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(hist.history["accuracy"], label="Train acc")
    axes[1].plot(hist.history["val_accuracy"], label="Val acc")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle(task_label or title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description="Lab 05: Dropout, BatchNorm, Optimizers")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per model (default 50)")
    args = parser.parse_args()
    epochs = args.epochs

    print("Loading Breast Cancer data...")
    X_train, X_val, y_train, y_val, input_dim = get_data()
    print(f"Training with {epochs} epochs per model.")

    # -------- Task 1: Baseline --------
    print("\n--- Task 1: Baseline (no Dropout, no BatchNorm) ---")
    model1 = build_baseline(input_dim)
    hist1 = train_model(model1, X_train, y_train, X_val, y_val, epochs=epochs, verbose=0)
    plot_history(hist1, "Baseline", "task1_baseline.png", "Task 1: Baseline")
    r1 = (hist1.history["val_loss"][-1], hist1.history["val_accuracy"][-1])
    print(f"  Val loss: {r1[0]:.4f}, Val acc: {r1[1]:.4f}")

    # -------- Task 2: Dropout --------
    print("\n--- Task 2: Add Dropout ---")
    model2 = build_dropout(input_dim)
    hist2 = train_model(model2, X_train, y_train, X_val, y_val, epochs=epochs, verbose=0)
    plot_history(hist2, "Dropout (0.3)", "task2_dropout.png", "Task 2: Dropout")
    r2 = (hist2.history["val_loss"][-1], hist2.history["val_accuracy"][-1])
    print(f"  Val loss: {r2[0]:.4f}, Val acc: {r2[1]:.4f}")

    # -------- Task 3: BatchNorm --------
    print("\n--- Task 3: Add BatchNorm ---")
    model3 = build_batchnorm(input_dim)
    hist3 = train_model(model3, X_train, y_train, X_val, y_val, epochs=epochs, verbose=0)
    plot_history(hist3, "BatchNorm", "task3_batchnorm.png", "Task 3: BatchNorm")
    r3 = (hist3.history["val_loss"][-1], hist3.history["val_accuracy"][-1])
    print(f"  Val loss: {r3[0]:.4f}, Val acc: {r3[1]:.4f}")

    # -------- Task 4: Dropout + BatchNorm --------
    print("\n--- Task 4: Dropout + BatchNorm ---")
    model4 = build_dropout_batchnorm(input_dim)
    hist4 = train_model(model4, X_train, y_train, X_val, y_val, epochs=epochs, verbose=0)
    plot_history(hist4, "Dropout + BatchNorm", "task4_combined.png", "Task 4: Combined")
    r4 = (hist4.history["val_loss"][-1], hist4.history["val_accuracy"][-1])
    print(f"  Val loss: {r4[0]:.4f}, Val acc: {r4[1]:.4f}")

    # -------- Task 5: Optimizer comparison (SGD vs Adam) --------
    print("\n--- Task 5: SGD vs Adam ---")
    model_sgd = build_baseline(input_dim)
    model_adam = build_baseline(input_dim)
    hist_sgd = train_model(model_sgd, X_train, y_train, X_val, y_val, optimizer="sgd", lr=0.01, epochs=epochs, verbose=0)
    hist_adam = train_model(model_adam, X_train, y_train, X_val, y_val, optimizer="adam", lr=0.001, epochs=epochs, verbose=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist_sgd.history["loss"], label="SGD train", alpha=0.8)
    axes[0].plot(hist_sgd.history["val_loss"], label="SGD val", alpha=0.8)
    axes[0].plot(hist_adam.history["loss"], label="Adam train", alpha=0.8)
    axes[0].plot(hist_adam.history["val_loss"], label="Adam val", alpha=0.8)
    axes[0].set_title("Task 5: Loss — SGD vs Adam")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(hist_sgd.history["val_accuracy"], label="SGD val acc")
    axes[1].plot(hist_adam.history["val_accuracy"], label="Adam val acc")
    axes[1].set_title("Task 5: Validation Accuracy — SGD vs Adam")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task5_optimizer_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved task5_optimizer_comparison.png")

    # -------- Task 6: Learning rate sensitivity --------
    print("\n--- Task 6: Learning rate sensitivity ---")
    lrs = [0.0001, 0.01, 0.5]
    fig, ax = plt.subplots(figsize=(10, 5))
    for lr in lrs:
        model_lr = build_baseline(input_dim)
        hist_lr = train_model(model_lr, X_train, y_train, X_val, y_val, lr=lr, epochs=epochs, verbose=0)
        ax.plot(hist_lr.history["val_loss"], label=f"lr={lr}", alpha=0.8)
    ax.set_title("Task 6: Validation loss vs learning rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "task6_learning_rate_sensitivity.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved task6_learning_rate_sensitivity.png")

    # -------- Comparison table (print) --------
    print("\n" + "=" * 60)
    print("COMPARISON TABLE (final epoch)")
    print("=" * 60)
    print(f"{'Model':<25} {'Val Loss':>10} {'Val Acc':>10}")
    print("-" * 60)
    print(f"{'Baseline':<25} {r1[0]:>10.4f} {r1[1]:>10.4f}")
    print(f"{'Dropout':<25} {r2[0]:>10.4f} {r2[1]:>10.4f}")
    print(f"{'BatchNorm':<25} {r3[0]:>10.4f} {r3[1]:>10.4f}")
    print(f"{'Dropout+BatchNorm':<25} {r4[0]:>10.4f} {r4[1]:>10.4f}")
    print(f"{'SGD (lr=0.01)':<25} {hist_sgd.history['val_loss'][-1]:>10.4f} {hist_sgd.history['val_accuracy'][-1]:>10.4f}")
    print(f"{'Adam (lr=0.001)':<25} {hist_adam.history['val_loss'][-1]:>10.4f} {hist_adam.history['val_accuracy'][-1]:>10.4f}")
    print("=" * 60)
    print("\nPlots saved to:", PLOTS_DIR)

if __name__ == "__main__":
    main()
