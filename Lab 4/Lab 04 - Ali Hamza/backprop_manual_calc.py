"""
Lab 04: Manual backpropagation step-by-step (Lab 04.pdf).
Network: 2 inputs, 2 hidden (h1->y3, h2->y4), 1 output (o1->y5). Sigmoid, eta=1, target=0.5.
"""

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# --- Network (from Lab 04.pdf) ---
x1, x2 = 0.35, 0.7
w11, w21 = 0.2, 0.2   # input -> h1
w12, w22 = 0.3, 0.3   # input -> h2
w13, w23 = 0.3, 0.9   # h1->y3, h2->y4 -> output
y_target = 0.5
eta = 1.0

# ========== Task 1: Forward Pass ==========
print("=" * 60)
print("--- Task 1: Forward Pass (Understanding the Prediction) ---")
print("=" * 60)

# Weighted sums at hidden neurons
a1 = w11 * x1 + w21 * x2
a2 = w12 * x1 + w22 * x2
print(f"1. Weighted sum at h1: a1 = w11*x1 + w21*x2 = {w11}*{x1} + {w21}*{x2} = {a1}")
print(f"2. Weighted sum at h2: a2 = w12*x1 + w22*x2 = {w12}*{x1} + {w22}*{x2} = {a2}")

# Sigmoid at hidden
y3 = sigmoid(a1)
y4 = sigmoid(a2)
print(f"3. After sigmoid: y3 = sigmoid(a1) = {y3:.4f},  y4 = sigmoid(a2) = {y4:.4f}")

# Output layer
a3 = w13 * y3 + w23 * y4
y5 = sigmoid(a3)
print(f"4. Weighted sum at output: a3 = w13*y3 + w23*y4 = {w13}*{y3:.4f} + {w23}*{y4:.4f} = {a3:.4f}")
print(f"5. Output: y5 = sigmoid(a3) = {y5:.4f}")
print(f"6. Target = {y_target},  Predicted y5 = {y5:.4f}  =>  Prediction is INCORRECT (y5 != target)")

# ========== Task 2: Error Calculation ==========
print()
print("=" * 60)
print("--- Task 2: Error Calculation (Identifying the Mistake) ---")
print("=" * 60)

error = y_target - y5
print(f"Error = y_target - y5 = {y_target} - {y5:.4f} = {error:.4f}")
print(f"Sign: negative => prediction (y5) is TOO HIGH; we need to decrease y5 toward 0.5.")

# ========== Task 3: Output Neuron Responsibility (delta5) ==========
print()
print("=" * 60)
print("--- Task 3: Output Neuron Responsibility (delta5) ---")
print("=" * 60)

delta5 = y5 * (1 - y5) * (y_target - y5)
print(f"delta5 = y5*(1-y5)*(y_target - y5)")
print(f"       = {y5:.4f} * (1 - {y5:.4f}) * ({y_target} - {y5:.4f})")
print(f"       = {y5 * (1 - y5):.4f} * ({y_target - y5:.4f}) = {delta5:.4f}")

# ========== Task 4: Hidden Neuron Responsibility (delta3, delta4) ==========
print()
print("=" * 60)
print("--- Task 4: Hidden Neuron Responsibility (delta3, delta4) ---")
print("=" * 60)

# w_j,output: from hidden j to output. So w13 from h1(y3), w23 from h2(y4)
delta3 = y3 * (1 - y3) * (w13 * delta5)
delta4 = y4 * (1 - y4) * (w23 * delta5)
print(f"delta3 = y3*(1-y3)*(w13*delta5) = {y3:.4f}*{1-y3:.4f}*({w13}*{delta5:.4f}) = {delta3:.4f}")
print(f"delta4 = y4*(1-y4)*(w23*delta5) = {y4:.4f}*{1-y4:.4f}*({w23}*{delta5:.4f}) = {delta4:.4f}")

# ========== Task 5: Weight Updates ==========
print()
print("=" * 60)
print("--- Task 5: Weight Updates (Learning from Mistakes) ---")
print("=" * 60)
print("Formula: Delta_w = eta * delta * input")
print()

# Hidden -> Output: delta_w = eta * delta5 * (input to output neuron = y3 or y4)
dw13 = eta * delta5 * y3
dw23 = eta * delta5 * y4
print("Hidden -> Output:")
print(f"  Delta_w13 = eta * delta5 * y3 = {eta} * {delta5:.4f} * {y3:.4f} = {dw13:.6f}")
print(f"  Delta_w23 = eta * delta5 * y4 = {eta} * {delta5:.4f} * {y4:.4f} = {dw23:.6f}")
w13_new = w13 + dw13
w23_new = w23 + dw23
print(f"  w13_new = w13 + Delta_w13 = {w13:.4f} + {dw13:.6f} = {w13_new:.6f}")
print(f"  w23_new = w23 + Delta_w23 = {w23:.4f} + {dw23:.6f} = {w23_new:.6f}")
print()

# Input -> Hidden: delta_w = eta * delta_j * input (x1 or x2)
dw11 = eta * delta3 * x1
dw21 = eta * delta3 * x2
dw12 = eta * delta4 * x1
dw22 = eta * delta4 * x2
print("Input -> Hidden:")
print(f"  Delta_w11 = eta * delta3 * x1 = {eta} * {delta3:.4f} * {x1} = {dw11:.6f}")
print(f"  Delta_w21 = eta * delta3 * x2 = {eta} * {delta3:.4f} * {x2} = {dw21:.6f}")
print(f"  Delta_w12 = eta * delta4 * x1 = {eta} * {delta4:.4f} * {x1} = {dw12:.6f}")
print(f"  Delta_w22 = eta * delta4 * x2 = {eta} * {delta4:.4f} * {x2} = {dw22:.6f}")
w11_new = w11 + dw11
w21_new = w21 + dw21
w12_new = w12 + dw12
w22_new = w22 + dw22
print(f"  w11_new = {w11_new:.6f},  w21_new = {w21_new:.6f}")
print(f"  w12_new = {w12_new:.6f},  w22_new = {w22_new:.6f}")
print()
print("Summary of new weights:")
print(f"  w11={w11_new:.6f} w21={w21_new:.6f}  w12={w12_new:.6f} w22={w22_new:.6f}")
print(f"  w13={w13_new:.6f} w23={w23_new:.6f}")
