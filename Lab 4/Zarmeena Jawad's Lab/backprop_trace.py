"""
Lab 04: Backpropagation trace (Lab 04.pdf).
2 inputs, 2 hidden, 1 output; sigmoid; eta=1; target=0.5.
Variable names: net_h1/net_h2, out_h1/out_h2, net_out, y_pred, err, d_out, d_h1/d_h2.
"""

import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# Network parameters (Lab 04.pdf)
x1, x2 = 0.35, 0.7
w11, w21 = 0.2, 0.2
w12, w22 = 0.3, 0.3
w13, w23 = 0.3, 0.9
target = 0.5
eta = 1.0

# ========== Step 1 — Forward pass ==========
print("=" * 55)
print("Step 1 — Forward pass")
print("=" * 55)

net_h1 = w11 * x1 + w21 * x2
net_h2 = w12 * x1 + w22 * x2
print(f"Hidden net inputs: net_h1 = {net_h1},  net_h2 = {net_h2}")

out_h1 = sigmoid(net_h1)
out_h2 = sigmoid(net_h2)
print(f"Hidden outputs: out_h1 = {out_h1:.4f},  out_h2 = {out_h2:.4f}")

net_out = w13 * out_h1 + w23 * out_h2
y_pred = sigmoid(net_out)
print(f"Output: net_out = {net_out:.4f},  y_pred = {y_pred:.4f}")
print(f"Target = {target}. Mismatch: y_pred != target.")

# ========== Step 2 — Output error ==========
print()
print("=" * 55)
print("Step 2 — Output error")
print("=" * 55)

err = target - y_pred
print(f"err = target - y_pred = {target} - {y_pred:.4f} = {err:.4f}")
print("Negative err: prediction above target; reduce output.")

# ========== Step 3 — Output neuron delta ==========
print()
print("=" * 55)
print("Step 3 — Output neuron delta")
print("=" * 55)

d_out = y_pred * (1 - y_pred) * (target - y_pred)
print(f"d_out = y_pred*(1-y_pred)*(target - y_pred) = {d_out:.4f}")

# ========== Step 4 — Hidden neuron deltas ==========
print()
print("=" * 55)
print("Step 4 — Hidden neuron deltas")
print("=" * 55)

d_h1 = out_h1 * (1 - out_h1) * (w13 * d_out)
d_h2 = out_h2 * (1 - out_h2) * (w23 * d_out)
print(f"d_h1 = {d_h1:.4f},  d_h2 = {d_h2:.4f}")

# ========== Step 5 — Weight updates ==========
print()
print("=" * 55)
print("Step 5 — Weight updates")
print("=" * 55)
print("Rule: Delta_w = eta * delta * input")
print()

d_w13 = eta * d_out * out_h1
d_w23 = eta * d_out * out_h2
w13_updated = w13 + d_w13
w23_updated = w23 + d_w23
print("Hidden to output:")
print(f"  d_w13 = {d_w13:.6f},  d_w23 = {d_w23:.6f}")
print(f"  w13_updated = {w13_updated:.6f},  w23_updated = {w23_updated:.6f}")
print()

d_w11 = eta * d_h1 * x1
d_w21 = eta * d_h1 * x2
d_w12 = eta * d_h2 * x1
d_w22 = eta * d_h2 * x2
w11_updated = w11 + d_w11
w21_updated = w21 + d_w21
w12_updated = w12 + d_w12
w22_updated = w22 + d_w22
print("Input to hidden:")
print(
    f"  d_w11 = {d_w11:.6f},  d_w21 = {d_w21:.6f},  d_w12 = {d_w12:.6f},  d_w22 = {d_w22:.6f}"
)
print(
    f"  Updated: w11={w11_updated:.6f} w21={w21_updated:.6f} w12={w12_updated:.6f} w22={w22_updated:.6f}"
)

# ========== Step 6 — Reflection ==========
print()
print("=" * 55)
print("Step 6 — Reflection")
print("=" * 55)
print(
    "(1) Backprop: error distributed backward; each unit gets a share of responsibility."
)
print("(2) Large eta: big steps, instability. Small eta: slow but stable learning.")
print("(3) 'Backward': gradients flow from output layer back toward inputs.")
