"""
Produce figures for the written report. Saves into ./figures/.
Run from repo root: python "Assignments/Zarmeena Jawad's Assignments/build_report_figures.py"
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Always use this script's directory (Zarmeena's assignment folder)
_SCRIPT_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
BASE = _SCRIPT_DIR
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Bipolar data
D_and = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float64)
t_and = np.array([1, -1, -1, -1], dtype=np.float64)
D_or = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float64)
t_or = np.array([1, 1, 1, -1], dtype=np.float64)
D_xor = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.float64)
t_xor = np.array([-1, 1, 1, -1], dtype=np.float64)

# Distinct style for this report: purple/green, diamond/plus markers (differs from Ali's blue/red circles/squares)
def scatter_by_class(ax, D, t, title):
    mask_pos = t == 1
    mask_neg = t == -1
    ax.scatter(D[mask_pos, 0], D[mask_pos, 1], c="#7c3aed", marker="D", s=110, label="+1", zorder=2, edgecolors="k", linewidths=1)
    ax.scatter(D[mask_neg, 0], D[mask_neg, 1], c="#16a34a", marker="P", s=110, label="-1", zorder=2, edgecolors="k", linewidths=1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal")

def draw_separator(ax, w):
    if np.abs(w[2]) < 1e-9:
        return
    x1 = np.linspace(-1.5, 1.5, 100)
    x2 = -(w[0] + w[1] * x1) / w[2]
    ax.plot(x1, x2, "k-", lw=2, label="Separating line")

# Minimal training for figures
class RosenblattPerceptron:
    def __init__(self, dim, lr=0.1, max_iter=100):
        self.weights = np.zeros(dim + 1)
        self.lr = lr
        self.max_iter = max_iter
    def _th(self, z):
        return np.where(z >= 0, 1.0, -1.0)
    def train(self, D, t):
        X = np.column_stack([np.ones(len(D)), D])
        for _ in range(self.max_iter):
            n = 0
            for i in range(X.shape[0]):
                y = self._th(X[i] @ self.weights)
                if y != t[i]:
                    self.weights += self.lr * t[i] * X[i]
                    n += 1
            if n == 0:
                break
        return self

class LMSNeuron:
    def __init__(self, dim, lr=0.1, max_iter=100, seed=None):
        self.weights = np.random.RandomState(seed).randn(dim + 1) * 0.01 if seed is not None else np.zeros(dim + 1)
        self.lr = lr
        self.max_iter = max_iter
    def train(self, D, t):
        X = np.column_stack([np.ones(len(D)), D])
        for _ in range(self.max_iter):
            for j in range(X.shape[0]):
                y_in = X[j] @ self.weights
                self.weights += self.lr * (t[j] - y_in) * X[j]
        return self

rp_and = RosenblattPerceptron(2).train(D_and, t_and)
rp_or = RosenblattPerceptron(2).train(D_or, t_or)
lms_xor = LMSNeuron(2, seed=99)
X_xor = np.column_stack([np.ones(len(D_xor)), D_xor])
for _ in range(500):
    for j in range(len(D_xor)):
        y_in = X_xor[j] @ lms_xor.weights
        lms_xor.weights += 0.1 * (t_xor[j] - y_in) * X_xor[j]

# 1) Separability
fig, axs = plt.subplots(1, 3, figsize=(11, 4))
scatter_by_class(axs[0], D_and, t_and, "AND")
scatter_by_class(axs[1], D_or, t_or, "OR")
scatter_by_class(axs[2], D_xor, t_xor, "XOR")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "separability.png"), dpi=150, bbox_inches="tight")
plt.close()

# 2) Perceptron boundaries
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
scatter_by_class(axs[0], D_and, t_and, "Perceptron — AND")
draw_separator(axs[0], rp_and.weights)
axs[0].legend()
scatter_by_class(axs[1], D_or, t_or, "Perceptron — OR")
draw_separator(axs[1], rp_or.weights)
axs[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "perceptron_boundaries.png"), dpi=150, bbox_inches="tight")
plt.close()

# 3) Adaline MSE AND/OR
mse_and = []
w = np.zeros(3)
X_and = np.column_stack([np.ones(len(D_and)), D_and])
for _ in range(200):
    for j in range(len(D_and)):
        w += 0.1 * (t_and[j] - X_and[j] @ w) * X_and[j]
    mse_and.append(np.mean((t_and - X_and @ w) ** 2))
mse_or = []
w = np.zeros(3)
X_or = np.column_stack([np.ones(len(D_or)), D_or])
for _ in range(200):
    for j in range(len(D_or)):
        w += 0.1 * (t_or[j] - X_or[j] @ w) * X_or[j]
    mse_or.append(np.mean((t_or - X_or @ w) ** 2))
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
axs[0].plot(mse_and, color="#0d9488")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("MSE")
axs[0].set_title("Adaline — AND")
axs[0].grid(True, alpha=0.25)
axs[1].plot(mse_or, color="#0f766e")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("MSE")
axs[1].set_title("Adaline — OR")
axs[1].grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "adaline_mse.png"), dpi=150, bbox_inches="tight")
plt.close()

# 4) XOR Perceptron
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(np.full(100, 4), color="#c2410c")
ax.set_xlabel("Epoch")
ax.set_ylabel("Misclassifications")
ax.set_title("Perceptron on XOR — no convergence")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "xor_perceptron.png"), dpi=150, bbox_inches="tight")
plt.close()

# 5) XOR Adaline MSE
mse_xor_list = []
w = np.random.RandomState(99).randn(3) * 0.01
for _ in range(500):
    for j in range(len(D_xor)):
        w += 0.1 * (t_xor[j] - X_xor[j] @ w) * X_xor[j]
    mse_xor_list.append(np.mean((t_xor - X_xor @ w) ** 2))
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(mse_xor_list, color="#7c3aed")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("Adaline on XOR — MSE does not go to zero")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "xor_adaline_mse.png"), dpi=150, bbox_inches="tight")
plt.close()

# 6) XOR + Adaline line
fig, ax = plt.subplots(figsize=(5, 5))
scatter_by_class(ax, D_xor, t_xor, "XOR and Adaline decision line")
draw_separator(ax, lms_xor.weights)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "xor_adaline_line.png"), dpi=150, bbox_inches="tight")
plt.close()

# 7) Architecture — vertical layout, rectangles
fig, ax = plt.subplots(figsize=(7, 8))
ax.set_xlim(0, 7)
ax.set_ylim(0, 10)
ax.axis("off")
# Input layer (rectangles)
for i, (label, y) in enumerate(zip(["$x_1$", "$x_2$", "1"], [7, 5, 3])):
    rect = plt.Rectangle((1, y - 0.35), 0.7, 0.7, facecolor="#e0f2fe", edgecolor="#0c4a6e", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(1.35, y, label, ha="center", va="center", fontsize=11)
# Hidden layer
for i, (label, y) in enumerate(zip(["H1", "H2"], [6.5, 3.5])):
    rect = plt.Rectangle((3.5, y - 0.4), 0.8, 0.8, facecolor="#fef3c7", edgecolor="#92400e", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3.9, y, label, ha="center", va="center", fontsize=11)
# Output
rect = plt.Rectangle((5.5, 4.9), 0.8, 0.8, facecolor="#d1fae5", edgecolor="#065f46", linewidth=1.5)
ax.add_patch(rect)
ax.text(5.9, 5.3, "$y$", ha="center", va="center", fontsize=12)
# Arrows input -> hidden
W_h = [[-1.5, 1, 1], [0.5, 1, 1]]
for j, yh in enumerate([6.5, 3.5]):
    for i, yi in enumerate([7, 5, 3]):
        ax.annotate("", xy=(3.5, yh), xytext=(1.7, yi), arrowprops=dict(arrowstyle="->", color="gray", lw=1))
        ax.text(2.4, (yi + yh) / 2, str(W_h[j][i]), fontsize=8)
# Arrows hidden -> output
ax.annotate("", xy=(5.5, 5.3), xytext=(4.3, 6.5), arrowprops=dict(arrowstyle="->", color="gray", lw=1))
ax.text(4.8, 6, "-1", fontsize=8)
ax.annotate("", xy=(5.5, 5.3), xytext=(4.3, 3.5), arrowprops=dict(arrowstyle="->", color="gray", lw=1))
ax.text(4.8, 4.2, "1", fontsize=8)
ax.text(2.5, 8.2, "Input", fontsize=10, fontweight="bold")
ax.text(3.3, 8.2, "Hidden", fontsize=10, fontweight="bold")
ax.text(5.2, 8.2, "Output", fontsize=10, fontweight="bold")
ax.text(3.5, 0.5, "Two-layer (2-2-1) network for XOR", fontsize=11, fontweight="bold", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "architecture_xor.png"), dpi=150, bbox_inches="tight")
plt.close()

print("Figures saved under", FIGDIR)
