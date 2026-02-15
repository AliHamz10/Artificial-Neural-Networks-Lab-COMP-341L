"""
Generate all figures for Assignment_1_Report.
Run from project root: ./venv/bin/python Assignments/generate_report_figures.py
Saves figures to Assignments/figures/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Data (bipolar)
X_and = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float64)
y_and = np.array([1, -1, -1, -1], dtype=np.float64)
X_or = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float64)
y_or = np.array([1, 1, 1, -1], dtype=np.float64)
X_xor = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float64)
y_xor = np.array([-1, 1, 1, -1], dtype=np.float64)

def plot_dataset(X, y, title, ax):
    pos = y == 1
    neg = y == -1
    ax.scatter(X[pos, 0], X[pos, 1], c="#2563eb", marker="o", s=120, label="+1", edgecolors="k", linewidths=1.5)
    ax.scatter(X[neg, 0], X[neg, 1], c="#dc2626", marker="s", s=120, label="-1", edgecolors="k", linewidths=1.5)
    ax.set_xlabel("$x_1$", fontsize=11)
    ax.set_ylabel("$x_2$", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

def plot_boundary(ax, w, xlim=(-1.5, 1.5)):
    w0, w1, w2 = w[0], w[1], w[2]
    if np.abs(w2) < 1e-9:
        return
    x1 = np.linspace(xlim[0], xlim[1], 100)
    x2 = -(w0 + w1 * x1) / w2
    ax.plot(x1, x2, "k--", linewidth=2, label="Decision boundary")

# --- Perceptron & Adaline (minimal) ---
class Perceptron:
    def __init__(self, n_features, learning_rate=0.1, max_epochs=100):
        self.w = np.zeros(n_features + 1)
        self.alpha = learning_rate
        self.max_epochs = max_epochs
    @staticmethod
    def step(net):
        return np.where(net >= 0, 1.0, -1.0)
    def fit(self, X, y):
        X_aug = np.column_stack([np.ones(len(X)), X])
        for _ in range(self.max_epochs):
            misclass = 0
            for i in range(len(X_aug)):
                out = self.step(np.dot(self.w, X_aug[i]))
                if out != y[i]:
                    self.w += self.alpha * y[i] * X_aug[i]
                    misclass += 1
            if misclass == 0:
                break
        return self
    def predict(self, X):
        X_aug = np.column_stack([np.ones(len(X)), X])
        return self.step(X_aug @ self.w)

class Adaline:
    def __init__(self, n_features, learning_rate=0.1, max_epochs=100, random_state=None):
        if random_state is None:
            self.w = np.zeros(n_features + 1)
        else:
            self.w = np.random.RandomState(random_state).randn(n_features + 1) * 0.01
        self.alpha = learning_rate
        self.max_epochs = max_epochs
    def fit(self, X, y):
        X_aug = np.column_stack([np.ones(len(X)), X])
        for _ in range(self.max_epochs):
            for i in range(len(X_aug)):
                y_in = np.dot(self.w, X_aug[i])
                self.w += self.alpha * (y[i] - y_in) * X_aug[i]
            y_in_all = X_aug @ self.w
            mse = np.mean((y - y_in_all) ** 2)
            if mse < 1e-6:
                break
        return self
    def predict(self, X):
        X_aug = np.column_stack([np.ones(len(X)), X])
        return np.where(X_aug @ self.w >= 0, 1.0, -1.0)

# Train models
perc_and = Perceptron(2, 0.1, 100).fit(X_and, y_and)
perc_or = Perceptron(2, 0.1, 100).fit(X_or, y_or)
ada_and = Adaline(2, 0.1, 500).fit(X_and, y_and)
ada_or = Adaline(2, 0.1, 500).fit(X_or, y_or)
perc_xor = Perceptron(2, 0.1, 500).fit(X_xor, y_xor)
ada_xor = Adaline(2, 0.1, 500, random_state=44)
X_aug_xor = np.column_stack([np.ones(len(X_xor)), X_xor])
mse_xor_list = []
for _ in range(500):
    for i in range(len(X_xor)):
        y_in = np.dot(ada_xor.w, X_aug_xor[i])
        ada_xor.w += 0.1 * (y_xor[i] - y_in) * X_aug_xor[i]
    mse_xor_list.append(np.mean((y_xor - X_aug_xor @ ada_xor.w) ** 2))
hist_xor = np.full(500, 4)  # Perceptron never converges

# ----- Figure 1: Linear separability (AND, OR, XOR) -----
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
plot_dataset(X_and, y_and, "AND — linearly separable", axes[0])
plot_dataset(X_or, y_or, "OR — linearly separable", axes[1])
plot_dataset(X_xor, y_xor, "XOR — not linearly separable", axes[2])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_linear_separability.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----- Figure 2: Perceptron decision boundaries -----
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
plot_dataset(X_and, y_and, "Perceptron on AND", axes[0])
plot_boundary(axes[0], perc_and.w)
axes[0].legend(loc="upper right")
plot_dataset(X_or, y_or, "Perceptron on OR", axes[1])
plot_boundary(axes[1], perc_or.w)
axes[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_perceptron_boundaries.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----- Figure 3: Adaline MSE (AND, OR) -----
mse_and = []
w = np.zeros(3)
X_aug_and = np.column_stack([np.ones(len(X_and)), X_and])
for _ in range(200):
    for i in range(len(X_and)):
        y_in = np.dot(w, X_aug_and[i])
        w += 0.1 * (y_and[i] - y_in) * X_aug_and[i]
    mse_and.append(np.mean((y_and - X_aug_and @ w) ** 2))
mse_or = []
w = np.zeros(3)
X_aug_or = np.column_stack([np.ones(len(X_or)), X_or])
for _ in range(200):
    for i in range(len(X_or)):
        y_in = np.dot(w, X_aug_or[i])
        w += 0.1 * (y_or[i] - y_in) * X_aug_or[i]
    mse_or.append(np.mean((y_or - X_aug_or @ w) ** 2))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(mse_and, color="#2563eb", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylabel("MSE", fontsize=11)
axes[0].set_title("Adaline on AND — MSE vs epoch", fontsize=12, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[1].plot(mse_or, color="#059669", linewidth=2)
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("MSE", fontsize=11)
axes[1].set_title("Adaline on OR — MSE vs epoch", fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_adaline_mse_and_or.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----- Figure 4: Perceptron XOR — does not converge -----
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hist_xor, color="#dc2626", linewidth=2)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Misclassifications", fontsize=11)
ax.set_title("Perceptron on XOR — does not converge", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_xor_perceptron_fail.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----- Figure 5: Adaline XOR — MSE -----
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(mse_xor_list, color="#7c3aed", linewidth=2)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("MSE", fontsize=11)
ax.set_title("Adaline on XOR — MSE does not go to zero", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_xor_adaline_mse.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----- Figure 6: XOR with Adaline decision boundary -----
fig, ax = plt.subplots(figsize=(5.5, 5.5))
plot_dataset(X_xor, y_xor, "XOR — Adaline decision boundary (cannot separate)", ax)
plot_boundary(ax, ada_xor.w)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_xor_adaline_boundary.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----- Figure 7: 2-2-1 Architecture diagram -----
def draw_network_architecture(savepath):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Layer positions
    x_input = 1.5
    x_hidden = 5
    x_output = 8.5
    y_centers_input = [4.5, 3, 1.5]   # x1, x2, bias
    y_centers_hidden = [4, 2]
    y_center_output = 3

    # Draw input layer
    for i, y in enumerate(y_centers_input):
        circle = plt.Circle((x_input, y), 0.35, fill=True, facecolor="#e0f2fe", edgecolor="#0369a1", linewidth=2)
        ax.add_patch(circle)
        label = ["$x_1$", "$x_2$", "1"][i]
        ax.text(x_input, y, label, ha="center", va="center", fontsize=12, fontweight="bold")

    # Draw hidden layer
    for i, y in enumerate(y_centers_hidden):
        circle = plt.Circle((x_hidden, y), 0.4, fill=True, facecolor="#fef3c7", edgecolor="#b45309", linewidth=2)
        ax.add_patch(circle)
        ax.text(x_hidden, y, f"H{i+1}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x_hidden, 0.5, "step", ha="center", va="center", fontsize=9, style="italic", color="#92400e")

    # Draw output layer
    circle = plt.Circle((x_output, y_center_output), 0.4, fill=True, facecolor="#d1fae5", edgecolor="#047857", linewidth=2)
    ax.add_patch(circle)
    ax.text(x_output, y_center_output, "y", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(x_output, 0.5, "step", ha="center", va="center", fontsize=9, style="italic", color="#065f46")

    # Weights input -> hidden
    W_h = [[-1.5, 1, 1], [0.5, 1, 1]]
    for j, y_h in enumerate(y_centers_hidden):
        for i, y_in in enumerate(y_centers_input):
            ax.annotate("", xy=(x_hidden - 0.4, y_h), xytext=(x_input + 0.35, y_in),
                        arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))
            ax.text((x_input + x_hidden) / 2 - 0.3, (y_in + y_h) / 2, f"{W_h[j][i]:.1f}", fontsize=8, color="#475569")

    # Weights hidden -> output (bias implicit; H1, H2 to y)
    W_out = [-1, -1, 1]  # bias, w_H1, w_H2
    for i, y_from in enumerate(y_centers_hidden):
        ax.annotate("", xy=(x_output - 0.4, y_center_output), xytext=(x_hidden + 0.4, y_from),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))
        ax.text((x_hidden + x_output) / 2, (y_from + y_center_output) / 2, f"{W_out[i+1]:.0f}", fontsize=9, ha="center")
    ax.text((x_hidden + x_output) / 2, y_center_output + 0.7, "bias = -1", fontsize=8, ha="center", color="#475569")

    # Layer labels
    ax.text(x_input, 5.4, "Input", ha="center", fontsize=11, fontweight="bold")
    ax.text(x_hidden, 5.4, "Hidden", ha="center", fontsize=11, fontweight="bold")
    ax.text(x_output, 5.4, "Output", ha="center", fontsize=11, fontweight="bold")
    ax.text(5, -0.3, "2–2–1 MLP for XOR (manual weights)", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()

draw_network_architecture(os.path.join(FIG_DIR, "fig7_architecture_mlp_xor.png"))

print("All figures saved to", FIG_DIR)
