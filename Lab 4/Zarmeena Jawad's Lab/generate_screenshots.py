"""
Generate task screenshots (PNG) from backprop_trace.py output.
Saves each step section as screenshots/taskN_*.png using matplotlib.
"""

import subprocess
import sys
import matplotlib.pyplot as plt
import os

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(LAB_DIR, "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

result = subprocess.run(
    [sys.executable, os.path.join(LAB_DIR, "backprop_trace.py")],
    capture_output=True,
    text=True,
    cwd=LAB_DIR,
)
full_output = result.stdout

# Split by "Step N — ..." headers
blocks = []
current = []
for line in full_output.splitlines():
    if line.strip().startswith("Step ") and " — " in line:
        if current:
            blocks.append("\n".join(current))
        current = [line]
    else:
        current.append(line)
if current:
    blocks.append("\n".join(current))
blocks = [b for b in blocks if b.strip()]
# Keep only blocks that start with "Step N —" (first line)
step_blocks = [
    b
    for b in blocks
    if b.strip().split("\n")[0].startswith("Step ")
    and " — " in b.strip().split("\n")[0]
]
if len(step_blocks) >= 6:
    blocks = step_blocks[:6]
elif len(blocks) < 6:
    blocks = [full_output] * 6

task_names = [
    ("task1_forward_pass", "Step 1 — Forward pass"),
    ("task2_error_calculation", "Step 2 — Output error"),
    ("task3_output_delta", "Step 3 — Output neuron delta"),
    ("task4_hidden_deltas", "Step 4 — Hidden neuron deltas"),
    ("task5_weight_updates", "Step 5 — Weight updates"),
    ("task6_reflection", "Step 6 — Reflection"),
]

for i, (filename, title) in enumerate(task_names):
    if i >= len(blocks):
        break
    text = blocks[i]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    ax.set_title(title, fontsize=12)
    outpath = os.path.join(SCREENSHOTS_DIR, f"{filename}.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {outpath}")
