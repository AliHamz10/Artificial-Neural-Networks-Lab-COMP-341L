"""
Lab 2 - Task 2: The Threshold Experiment
Author: Ali Hamza
Roll Number: B23F0063AI106
Section: B.S AI - Red

This task explores how bias controls when a neuron "wakes up" by sweeping bias values
and observing neuron behavior with different activation functions.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: Fix Everything Except Bias
# ============================================================================
# Keep inputs and weights constant, only vary bias

# Fixed inputs (same as Task 1)
inputs = np.array([0.85, 0.72, 0.25])  # Speech rate, pitch variation, pause duration

# Fixed weights (same as Task 1)
weights = np.array([0.4, 0.5, 0.1])

# Fixed weighted sum (without bias)
weighted_sum_no_bias = np.dot(inputs, weights)
print("=" * 70)
print("TASK 2: The Threshold Experiment - Bias Sweep")
print("=" * 70)
print(f"\nFixed Parameters:")
print(f"  Inputs: {inputs}")
print(f"  Weights: {weights}")
print(f"  Weighted Sum (without bias): {weighted_sum_no_bias:.4f}")

# ============================================================================
# STEP 2: Sweep Bias Values
# ============================================================================
# Gradually change bias from very negative to very positive
bias_range = np.linspace(-3.0, 3.0, 300)  # 300 points for smooth curve

# Compute weighted sum with bias for each bias value
z_values = weighted_sum_no_bias + bias_range

# ============================================================================
# STEP 3: Apply Activation Functions
# ============================================================================

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow

def tanh(z):
    """Hyperbolic tangent activation function"""
    return np.tanh(z)

def relu(z):
    """Rectified Linear Unit activation function"""
    return np.maximum(0, z)

# Compute outputs for each activation function
sigmoid_outputs = sigmoid(z_values)
tanh_outputs = tanh(z_values)
relu_outputs = relu(z_values)

# ============================================================================
# STEP 4: Plot Behavior
# ============================================================================

plt.figure(figsize=(14, 10))

# Plot 1: Sigmoid
plt.subplot(2, 2, 1)
plt.plot(bias_range, sigmoid_outputs, 'b-', linewidth=2.5, label='Sigmoid Output')
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (0.5)')
plt.xlabel('Bias Value', fontsize=11, fontweight='bold')
plt.ylabel('Neuron Output', fontsize=11, fontweight='bold')
plt.title('Sigmoid: Gradual Wake-Up', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-3, 3)
plt.ylim(-0.1, 1.1)

# Find where sigmoid crosses 0.5 (wake-up point)
sigmoid_wake_up_idx = np.argmin(np.abs(sigmoid_outputs - 0.5))
sigmoid_wake_up_bias = bias_range[sigmoid_wake_up_idx]
plt.plot(sigmoid_wake_up_bias, 0.5, 'ro', markersize=10, label=f'Wake-up: {sigmoid_wake_up_bias:.2f}')
plt.legend()

# Plot 2: ReLU
plt.subplot(2, 2, 2)
plt.plot(bias_range, relu_outputs, 'g-', linewidth=2.5, label='ReLU Output')
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.axvline(x=-weighted_sum_no_bias, color='red', linestyle='--', linewidth=1.5, 
            alpha=0.7, label=f'Threshold ({-weighted_sum_no_bias:.2f})')
plt.xlabel('Bias Value', fontsize=11, fontweight='bold')
plt.ylabel('Neuron Output', fontsize=11, fontweight='bold')
plt.title('ReLU: Sudden Wake-Up', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-3, 3)
plt.ylim(-0.2, max(relu_outputs) + 0.2)

# Find where ReLU wakes up (where z becomes positive)
relu_wake_up_bias = -weighted_sum_no_bias
plt.plot(relu_wake_up_bias, 0, 'ro', markersize=10, label=f'Wake-up: {relu_wake_up_bias:.2f}')
plt.legend()

# Plot 3: Tanh
plt.subplot(2, 2, 3)
plt.plot(bias_range, tanh_outputs, 'm-', linewidth=2.5, label='Tanh Output')
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Neutral (0)')
plt.xlabel('Bias Value', fontsize=11, fontweight='bold')
plt.ylabel('Neuron Output', fontsize=11, fontweight='bold')
plt.title('Tanh: Neutral Zone', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-3, 3)
plt.ylim(-1.1, 1.1)

# Find where tanh crosses zero
tanh_wake_up_idx = np.argmin(np.abs(tanh_outputs))
tanh_wake_up_bias = bias_range[tanh_wake_up_idx]
plt.plot(tanh_wake_up_bias, 0, 'ro', markersize=10, label=f'Neutral: {tanh_wake_up_bias:.2f}')
plt.legend()

# Plot 4: Comparison Overlay
plt.subplot(2, 2, 4)
plt.plot(bias_range, sigmoid_outputs, 'b-', linewidth=2, label='Sigmoid', alpha=0.8)
plt.plot(bias_range, tanh_outputs, 'm-', linewidth=2, label='Tanh', alpha=0.8)
plt.plot(bias_range, relu_outputs, 'g-', linewidth=2, label='ReLU', alpha=0.8)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.xlabel('Bias Value', fontsize=11, fontweight='bold')
plt.ylabel('Neuron Output', fontsize=11, fontweight='bold')
plt.title('All Activations: Side-by-Side Comparison', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(-3, 3)

plt.tight_layout()
plt.savefig('task2_bias_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Critical Thinking: When Does Each Neuron Wake Up?
# ============================================================================
print(f"\n" + "=" * 70)
print("CRITICAL THINKING: At What Bias Does the Neuron 'Wake Up'?")
print("=" * 70)

print(f"\n1. SIGMOID - Gradual Wake-Up:")
print(f"   Wake-up bias: {sigmoid_wake_up_bias:.4f}")
print(f"   At this bias, output = 0.5 (50% confidence)")
print(f"   Why gradual?")
print(f"   - Sigmoid is a smooth S-curve")
print(f"   - No sharp transition, output changes smoothly from 0 to 1")
print(f"   - Even before 'wake-up', neuron gives small responses")
print(f"   - After 'wake-up', response increases gradually, not instantly")

print(f"\n2. ReLU - Sudden Wake-Up:")
print(f"   Wake-up bias: {relu_wake_up_bias:.4f}")
print(f"   At this bias, weighted sum z becomes positive")
print(f"   Why sudden?")
print(f"   - ReLU has a hard threshold at z = 0")
print(f"   - For z < 0: output = 0 (completely silent)")
print(f"   - For z > 0: output = z (linear activation)")
print(f"   - No gradual transition, it's either OFF or ON")
print(f"   - Binary-like behavior: silent until threshold, then active")

print(f"\n3. TANH - Neutral Zone:")
print(f"   Neutral point bias: {tanh_wake_up_bias:.4f}")
print(f"   At this bias, output = 0 (neither positive nor negative)")
print(f"   Why neutral zone?")
print(f"   - Tanh outputs range from -1 to +1, centered at 0")
print(f"   - Negative outputs = suppression/inhibition")
print(f"   - Positive outputs = activation/amplification")
print(f"   - Zero = neutral state (no effect)")
print(f"   - The 'wake-up' is when it crosses from negative to positive")

print(f"\n" + "=" * 70)
print("KEY INSIGHT:")
print("=" * 70)
print("Bias controls WHEN the neuron activates, but HOW it activates depends")
print("on the activation function:")
print("- Sigmoid: Soft, gradual response (probabilistic thinking)")
print("- ReLU: Hard, sudden response (binary-like decision)")
print("- Tanh: Balanced response with neutral zone (signed activation)")

# ============================================================================
# Detailed Analysis: Output Values at Key Bias Points
# ============================================================================
print(f"\n" + "=" * 70)
print("DETAILED ANALYSIS: Output at Key Bias Values")
print("=" * 70)

key_biases = [-2.0, -1.0, 0.0, 1.0, 2.0]
print(f"\n{'Bias':<8} {'Sigmoid':<12} {'Tanh':<12} {'ReLU':<12}")
print("-" * 50)

for b in key_biases:
    z = weighted_sum_no_bias + b
    s = sigmoid(z)
    t = tanh(z)
    r = relu(z)
    print(f"{b:>6.1f}   {s:>10.4f}   {t:>10.4f}   {r:>10.4f}")

print(f"\nObservation:")
print(f"- At very negative bias (-2.0): All activations are low/suppressed")
print(f"- At zero bias (0.0): Sigmoid ~0.5, Tanh ~0, ReLU depends on weighted sum")
print(f"- At very positive bias (2.0): All activations are high/active")
