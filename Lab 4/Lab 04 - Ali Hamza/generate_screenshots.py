"""
Generate task screenshots (PNG) from backprop_manual_calc output.
Saves each task section as screenshots/taskN_*.png using matplotlib.
"""

import subprocess
import matplotlib.pyplot as plt
import os

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(LAB_DIR, "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

result = subprocess.run(
    ["python3", os.path.join(LAB_DIR, "backprop_manual_calc.py")],
    capture_output=True,
    text=True,
    cwd=LAB_DIR,
)
full_output = result.stdout

# Split by task headers (--- Task N: ... ---)
blocks = []
current = []
for line in full_output.splitlines():
    if "--- Task " in line and "---" in line:
        if current:
            blocks.append("\n".join(current))
        current = [line]
    else:
        current.append(line)
if current:
    blocks.append("\n".join(current))
# Drop empty blocks (e.g. before first "--- Task 1")
blocks = [b for b in blocks if b.strip()]
if len(blocks) < 5:
    blocks = [full_output]  # fallback: one block with all output

# Task 1 = first block, Task 2 = second, ...
task_names = [
    ("task1_forward_pass", "Task 1: Forward Pass"),
    ("task2_error_calculation", "Task 2: Error Calculation"),
    ("task3_output_delta", "Task 3: Output Neuron Responsibility (delta5)"),
    ("task4_hidden_deltas", "Task 4: Hidden Neuron Responsibility (delta3, delta4)"),
    ("task5_weight_updates", "Task 5: Weight Updates"),
]

for i, (filename, title) in enumerate(task_names):
    if i >= len(blocks):
        break
    text = blocks[i]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    ax.set_title(title, fontsize=12)
    outpath = os.path.join(SCREENSHOTS_DIR, f"{filename}.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {outpath}")

# Task 6: reflection is text-only in report; optional placeholder image with text
task6_text = (
    "Task 6: Interpretation & Reflection\n"
    "1. Backpropagation as blame assignment: assigns responsibility for the error to each neuron/weight.\n"
    "2. Very large learning rate: unstable, overshoot, may diverge.\n"
    "   Very small learning rate: very slow learning, may not converge.\n"
    "3. 'Backward' propagation: error is propagated from output layer back toward the input layer."
)
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
ax.text(0.02, 0.98, task6_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
ax.set_title("Task 6: Interpretation & Reflection", fontsize=12)
outpath = os.path.join(SCREENSHOTS_DIR, "task6_reflection.png")
plt.tight_layout()
plt.savefig(outpath, dpi=120, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {outpath}")
