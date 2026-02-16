"""
Lab 05 experiments: Dropout, Batch Normalization, and Optimizers (Lab 05.pdf).
Uses the Breast Cancer dataset from sklearn. Runs six experiments: baseline MLP,
dropout regularization, batch normalization, combined, optimizer comparison (SGD vs Adam),
and learning rate sensitivity.
Execute from repo root: ./venv/bin/python3 "Lab 5/Zarmeena Jawad's Lab/lab5_experiments.py"
From this directory: ../../venv/bin/python3 lab5_experiments.py
Optional: ../../venv/bin/python3 lab5_experiments.py --epochs 5  (quick run)
"""

import argparse
import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MINI_BATCH = 32
N_FEATURES = 30  # Breast Cancer has 30 input features


def prepare_breast_cancer_data(test_fraction=0.2, seed=42):
    """Load Breast Cancer data, split into train/validation, and standardize features."""
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    return X_tr, X_va, y_tr, y_va, X_tr.shape[1]


def baseline_mlp(input_dim):
    """Experiment 1: Deep MLP with no regularization (no Dropout, no BatchNorm)."""
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="baseline_mlp")


def mlp_with_dropout(input_dim, drop_rate=0.3):
    """Experiment 2: Same MLP with Dropout after each hidden layer."""
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(drop_rate),
        layers.Dense(64, activation="relu"),
        layers.Dropout(drop_rate),
        layers.Dense(32, activation="relu"),
        layers.Dropout(drop_rate),
        layers.Dense(1, activation="sigmoid"),
    ], name="mlp_dropout")


def mlp_with_batchnorm(input_dim):
    """Experiment 3: MLP with BatchNorm (Dense -> BN -> ReLU)."""
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
    ], name="mlp_batchnorm")


def mlp_combined(input_dim, drop_rate=0.3):
    """Experiment 4: MLP with both BatchNorm and Dropout."""
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(drop_rate),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(drop_rate),
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(drop_rate),
        layers.Dense(1, activation="sigmoid"),
    ], name="mlp_combined")


def run_training(model, X_tr, y_tr, X_va, y_va, optimizer_key="adam", learning_rate=0.001, epochs=50):
    """Compile and fit the model; return history dict."""
    if optimizer_key == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=MINI_BATCH,
        verbose=0,
    )
    return history


def save_training_curves(history, plot_title, file_name, suptitle=None):
    """Save a 2-panel figure: loss and accuracy over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"], label="Training loss")
    ax1.plot(history.history["val_loss"], label="Validation loss")
    ax1.set_title(f"{plot_title} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(history.history["accuracy"], label="Training accuracy")
    ax2.plot(history.history["val_accuracy"], label="Validation accuracy")
    ax2.set_title(f"{plot_title} — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Written: {file_name}")


def main():
    ap = argparse.ArgumentParser(description="Lab 05 experiments: Dropout, BatchNorm, Optimizers")
    ap.add_argument("--epochs", type=int, default=50, help="Number of epochs per run")
    args = ap.parse_args()
    n_epochs = args.epochs

    print("Loading and preparing Breast Cancer dataset...")
    X_tr, X_va, y_tr, y_va, n_in = prepare_breast_cancer_data()
    print(f"Running all experiments for {n_epochs} epochs each.\n")

    results = []

    # ---------- Experiment 1: Baseline ----------
    print("[Experiment 1] Baseline MLP (no Dropout, no BatchNorm)")
    m1 = baseline_mlp(n_in)
    h1 = run_training(m1, X_tr, y_tr, X_va, y_va, epochs=n_epochs)
    save_training_curves(h1, "Baseline", "task1_baseline.png", "Task 1: Baseline")
    results.append(("Baseline", h1.history["val_loss"][-1], h1.history["val_accuracy"][-1]))
    print(f"  Final val loss: {results[-1][1]:.4f}, val accuracy: {results[-1][2]:.4f}\n")

    # ---------- Experiment 2: Dropout ----------
    print("[Experiment 2] MLP with Dropout (rate=0.3)")
    m2 = mlp_with_dropout(n_in)
    h2 = run_training(m2, X_tr, y_tr, X_va, y_va, epochs=n_epochs)
    save_training_curves(h2, "Dropout", "task2_dropout.png", "Task 2: Dropout")
    results.append(("Dropout", h2.history["val_loss"][-1], h2.history["val_accuracy"][-1]))
    print(f"  Final val loss: {results[-1][1]:.4f}, val accuracy: {results[-1][2]:.4f}\n")

    # ---------- Experiment 3: BatchNorm ----------
    print("[Experiment 3] MLP with BatchNorm")
    m3 = mlp_with_batchnorm(n_in)
    h3 = run_training(m3, X_tr, y_tr, X_va, y_va, epochs=n_epochs)
    save_training_curves(h3, "BatchNorm", "task3_batchnorm.png", "Task 3: BatchNorm")
    results.append(("BatchNorm", h3.history["val_loss"][-1], h3.history["val_accuracy"][-1]))
    print(f"  Final val loss: {results[-1][1]:.4f}, val accuracy: {results[-1][2]:.4f}\n")

    # ---------- Experiment 4: Dropout + BatchNorm ----------
    print("[Experiment 4] MLP with Dropout and BatchNorm")
    m4 = mlp_combined(n_in)
    h4 = run_training(m4, X_tr, y_tr, X_va, y_va, epochs=n_epochs)
    save_training_curves(h4, "Dropout + BatchNorm", "task4_combined.png", "Task 4: Combined")
    results.append(("Dropout+BatchNorm", h4.history["val_loss"][-1], h4.history["val_accuracy"][-1]))
    print(f"  Final val loss: {results[-1][1]:.4f}, val accuracy: {results[-1][2]:.4f}\n")

    # ---------- Experiment 5: SGD vs Adam ----------
    print("[Experiment 5] Optimizer comparison: SGD vs Adam")
    m_sgd = baseline_mlp(n_in)
    m_adam = baseline_mlp(n_in)
    h_sgd = run_training(m_sgd, X_tr, y_tr, X_va, y_va, optimizer_key="sgd", learning_rate=0.01, epochs=n_epochs)
    h_adam = run_training(m_adam, X_tr, y_tr, X_va, y_va, optimizer_key="adam", learning_rate=0.001, epochs=n_epochs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(h_sgd.history["loss"], label="SGD (train)", alpha=0.8)
    ax1.plot(h_sgd.history["val_loss"], label="SGD (val)", alpha=0.8)
    ax1.plot(h_adam.history["loss"], label="Adam (train)", alpha=0.8)
    ax1.plot(h_adam.history["val_loss"], label="Adam (val)", alpha=0.8)
    ax1.set_title("Task 5: Loss — SGD vs Adam")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(h_sgd.history["val_accuracy"], label="SGD")
    ax2.plot(h_adam.history["val_accuracy"], label="Adam")
    ax2.set_title("Task 5: Validation accuracy — SGD vs Adam")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "task5_optimizer_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()
    results.append(("SGD (lr=0.01)", h_sgd.history["val_loss"][-1], h_sgd.history["val_accuracy"][-1]))
    results.append(("Adam (lr=0.001)", h_adam.history["val_loss"][-1], h_adam.history["val_accuracy"][-1]))
    print("  Written: task5_optimizer_comparison.png\n")

    # ---------- Experiment 6: Learning rate sensitivity ----------
    print("[Experiment 6] Learning rate sensitivity (0.0001, 0.01, 0.5)")
    fig, ax = plt.subplots(figsize=(10, 5))
    for lr in [0.0001, 0.01, 0.5]:
        m_lr = baseline_mlp(n_in)
        h_lr = run_training(m_lr, X_tr, y_tr, X_va, y_va, learning_rate=lr, epochs=n_epochs)
        ax.plot(h_lr.history["val_loss"], label=f"lr={lr}", alpha=0.8)
    ax.set_title("Task 6: Validation loss vs learning rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "task6_learning_rate_sensitivity.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  Written: task6_learning_rate_sensitivity.png\n")

    # ---------- Summary table ----------
    print("=" * 58)
    print("COMPARISON TABLE (final epoch)")
    print("=" * 58)
    print(f"{'Model':<24} {'Val Loss':>10} {'Val Acc':>10}")
    print("-" * 58)
    for name, vloss, vacc in results:
        print(f"{name:<24} {vloss:>10.4f} {vacc:>10.4f}")
    print("=" * 58)
    print(f"\nAll plots saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
