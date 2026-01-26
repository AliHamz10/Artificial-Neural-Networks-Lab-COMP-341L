"""
Lab 2 - Task 1: Build a Neuron That Thinks, Not Just Calculates
Author: Ali Hamza
Roll Number: B23F0063AI106
Section: B.S AI - Red

This task implements a single artificial neuron for stress detection from speech signals.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: Choose Input Features
# ============================================================================
# Three speech-related features for stress detection:
# 1. Speech rate (words per minute) - normalized to 0-1 scale
# 2. Pitch variation (standard deviation of pitch) - normalized to 0-1 scale  
# 3. Pause duration (average pause length) - normalized to 0-1 scale

# Simulate input values for demonstration
# Example: A person speaking quickly with high pitch variation and short pauses
speech_rate = 0.85      # High speech rate (0 = slow, 1 = fast)
pitch_variation = 0.72  # Moderate-high pitch variation (0 = flat, 1 = very shaky)
pause_duration = 0.25   # Short pauses (0 = long pauses, 1 = no pauses)

inputs = np.array([speech_rate, pitch_variation, pause_duration])
print("=" * 70)
print("TASK 1: Building a Decision-Making Neuron")
print("=" * 70)
print(f"\nInput Features:")
print(f"  Speech Rate: {speech_rate:.2f}")
print(f"  Pitch Variation: {pitch_variation:.2f}")
print(f"  Pause Duration: {pause_duration:.2f}")

# ============================================================================
# STEP 2: Assign Weights
# ============================================================================
# Weight justification:
# - Speech rate (w1=0.4): Moderate importance - fast speech can indicate stress
#   but not always (some people naturally speak fast)
# - Pitch variation (w2=0.5): HIGHEST importance - stressed speech typically
#   shows more pitch instability and shakiness
# - Pause duration (w3=0.1): LOWEST importance - pause patterns are less
#   reliable indicators compared to pitch and rate

weights = np.array([0.4, 0.5, 0.1])
print(f"\nWeights (Importance):")
print(f"  Speech Rate Weight: {weights[0]:.1f} (Moderate - fast speech may indicate stress)")
print(f"  Pitch Variation Weight: {weights[1]:.1f} (HIGHEST - key stress indicator)")
print(f"  Pause Duration Weight: {weights[2]:.1f} (LOWEST - less reliable)")

# ============================================================================
# STEP 3: Add Bias (Sensitivity Control)
# ============================================================================
# Bias = -0.3 (moderately negative)
# - High positive bias: Neuron fires easily (too sensitive, flags stress often)
# - Low/negative bias: Neuron is strict (requires strong evidence to activate)
# - Our bias of -0.3 means the neuron needs moderate evidence before activating

bias = -0.3
print(f"\nBias (Sensitivity): {bias:.1f}")
print(f"  Interpretation: Moderately strict - requires evidence before activating")
print(f"  High bias → fires easily (too sensitive)")
print(f"  Low/negative bias → strict (needs strong evidence)")

# ============================================================================
# STEP 4: Compute Weighted Sum
# ============================================================================
weighted_sum = np.dot(inputs, weights) + bias
print(f"\nWeighted Sum (Evidence): {weighted_sum:.4f}")
print(f"  Calculation: ({speech_rate:.2f} × {weights[0]:.1f}) + "
      f"({pitch_variation:.2f} × {weights[1]:.1f}) + "
      f"({pause_duration:.2f} × {weights[2]:.1f}) + ({bias:.1f}) = {weighted_sum:.4f}")

# ============================================================================
# STEP 5: Apply Different Activation Functions
# ============================================================================

def sigmoid(z):
    """Sigmoid activation: smooth S-curve, outputs 0-1"""
    return 1 / (1 + np.exp(-z))

def tanh(z):
    """Hyperbolic tangent: outputs -1 to 1, centered at zero"""
    return np.tanh(z)

def relu(z):
    """Rectified Linear Unit: linear for positive, zero for negative"""
    return np.maximum(0, z)

# Apply activations
output_sigmoid = sigmoid(weighted_sum)
output_tanh = tanh(weighted_sum)
output_relu = relu(weighted_sum)

print(f"\n" + "=" * 70)
print("ACTIVATION FUNCTION RESULTS")
print("=" * 70)
print(f"\nSigmoid Output: {output_sigmoid:.4f}")
print(f"Tanh Output: {output_tanh:.4f}")
print(f"ReLU Output: {output_relu:.4f}")

# ============================================================================
# Visualization: Compare Activation Functions
# ============================================================================
z_range = np.linspace(-5, 5, 1000)
sigmoid_vals = sigmoid(z_range)
tanh_vals = tanh(z_range)
relu_vals = relu(z_range)

plt.figure(figsize=(12, 5))

# Plot 1: All activations together
plt.subplot(1, 2, 1)
plt.plot(z_range, sigmoid_vals, label='Sigmoid', linewidth=2)
plt.plot(z_range, tanh_vals, label='Tanh', linewidth=2)
plt.plot(z_range, relu_vals, label='ReLU', linewidth=2)
plt.axvline(x=weighted_sum, color='red', linestyle='--', linewidth=1.5, 
            label=f'Our Input (z={weighted_sum:.2f})')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Weighted Sum (z)', fontsize=11)
plt.ylabel('Activation Output', fontsize=11)
plt.title('Activation Functions Comparison', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Zoomed view around our input
plt.subplot(1, 2, 2)
zoom_range = np.linspace(weighted_sum - 1, weighted_sum + 1, 200)
plt.plot(zoom_range, sigmoid(zoom_range), label='Sigmoid', linewidth=2.5, marker='o', markersize=3)
plt.plot(zoom_range, tanh(zoom_range), label='Tanh', linewidth=2.5, marker='s', markersize=3)
plt.plot(zoom_range, relu(zoom_range), label='ReLU', linewidth=2.5, marker='^', markersize=3)
plt.axvline(x=weighted_sum, color='red', linestyle='--', linewidth=2, 
            label=f'Input z={weighted_sum:.2f}')
plt.scatter([weighted_sum], [output_sigmoid], color='blue', s=100, zorder=5, label='Sigmoid Output')
plt.scatter([weighted_sum], [output_tanh], color='green', s=100, zorder=5, label='Tanh Output')
plt.scatter([weighted_sum], [output_relu], color='orange', s=100, zorder=5, label='ReLU Output')
plt.xlabel('Weighted Sum (z)', fontsize=11)
plt.ylabel('Activation Output', fontsize=11)
plt.title('Zoomed View: Our Neuron Output', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_activation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# ANALYSIS: Why Different Behaviors?
# ============================================================================
print(f"\n" + "=" * 70)
print("ANALYSIS: Why Does the Same Neuron Behave Differently?")
print("=" * 70)

print(f"\n1. SOFT MARGIN:")
print(f"   - Sigmoid ({output_sigmoid:.4f}): Provides soft, gradual response")
print(f"     → No hard cutoff, gives confidence level (0-1 probability)")
print(f"   - Tanh ({output_tanh:.4f}): Also soft but centered at zero")
print(f"     → Can express negative responses (suppression)")
print(f"   - ReLU ({output_relu:.4f}): Hard threshold at zero")
print(f"     → Either activates (positive) or stays silent (zero)")

print(f"\n2. CONFIDENCE:")
print(f"   - Sigmoid: {output_sigmoid:.4f} = {output_sigmoid*100:.1f}% confidence in stress")
print(f"   - Tanh: {output_tanh:.4f} = moderate positive activation")
print(f"   - ReLU: {output_relu:.4f} = {output_relu*100:.1f}% of maximum activation")

print(f"\n3. SUPPRESSION vs AMPLIFICATION:")
print(f"   - Sigmoid: Only amplifies (0 to 1), cannot suppress")
print(f"   - Tanh: Can both amplify (positive) and suppress (negative)")
print(f"   - ReLU: Amplifies positive signals, completely suppresses negative")

print(f"\n4. NON-LINEARITY:")
print(f"   - Sigmoid: Highly non-linear S-curve → smooth transitions")
print(f"   - Tanh: Non-linear but symmetric → balanced response")
print(f"   - ReLU: Piecewise linear → simple but effective")

print(f"\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("The same weighted sum produces different outputs because each activation")
print("function transforms the evidence differently:")
print("- Sigmoid: Interprets as probability (soft decision)")
print("- Tanh: Interprets as signed strength (can be negative)")
print("- ReLU: Interprets as raw activation (only if positive)")
