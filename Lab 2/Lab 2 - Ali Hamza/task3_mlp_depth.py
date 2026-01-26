"""
Lab 2 - Task 3: Why One Neuron Is Not Enough (The Depth Problem)
Author: Ali Hamza
Roll Number: B23F0063AI106
Section: B.S AI - Red

This task demonstrates why a single neuron fails for complex patterns and how
adding a hidden layer enables the network to learn non-linear decision boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# SCENARIO: Two Types of Speech That Confuse a Single Neuron
# ============================================================================
# Type 1: Calm but fast speakers
#   - High speech rate (fast)
#   - Low pitch variation (calm, stable)
#   - Short pauses (natural fast speech)
#
# Type 2: Stressed but slow speakers  
#   - Low speech rate (slow, hesitant)
#   - High pitch variation (stressed, shaky)
#   - Long pauses (uncertainty)

print("=" * 70)
print("TASK 3: Why One Neuron Is Not Enough")
print("=" * 70)

# ============================================================================
# PART A: Try With ONE Neuron
# ============================================================================
print(f"\n{'='*70}")
print("PART A: Single Neuron Analysis")
print(f"{'='*70}")

# Example inputs
calm_fast = np.array([0.9, 0.2, 0.3])   # Fast but calm
stressed_slow = np.array([0.2, 0.9, 0.8])  # Slow but stressed

# Single neuron weights (trying to detect stress)
weights_single = np.array([0.3, 0.6, 0.1])  # Emphasize pitch variation
bias_single = -0.2

# Compute outputs
def single_neuron_output(inputs, weights, bias):
    """Single neuron forward pass"""
    z = np.dot(inputs, weights) + bias
    return 1 / (1 + np.exp(-z))  # Sigmoid activation

output_calm = single_neuron_output(calm_fast, weights_single, bias_single)
output_stressed = single_neuron_output(stressed_slow, weights_single, bias_single)

print(f"\nInput 1 (Calm but Fast):")
print(f"  Features: Speech Rate={calm_fast[0]:.1f}, Pitch Var={calm_fast[1]:.1f}, Pauses={calm_fast[2]:.1f}")
print(f"  Single Neuron Output: {output_calm:.4f}")

print(f"\nInput 2 (Stressed but Slow):")
print(f"  Features: Speech Rate={stressed_slow[0]:.1f}, Pitch Var={stressed_slow[1]:.1f}, Pauses={stressed_slow[2]:.1f}")
print(f"  Single Neuron Output: {output_stressed:.4f}")

print(f"\nPROBLEM:")
print(f"  Both outputs are similar ({output_calm:.4f} vs {output_stressed:.4f})")
print(f"  The neuron cannot distinguish between:")
print(f"    - Fast speech (high rate) + calm (low pitch var)")
print(f"    - Slow speech (low rate) + stressed (high pitch var)")
print(f"  This is because a single neuron can only create a LINEAR boundary!")

# Test with multiple examples
test_inputs = [
    np.array([0.9, 0.2, 0.3]),  # Calm fast
    np.array([0.8, 0.25, 0.4]), # Calm fast (variant)
    np.array([0.2, 0.9, 0.8]),  # Stressed slow
    np.array([0.3, 0.85, 0.75]), # Stressed slow (variant)
    np.array([0.5, 0.5, 0.5]),   # Ambiguous
]

print(f"\n{'='*70}")
print("Testing Multiple Examples with Single Neuron:")
print(f"{'='*70}")
print(f"{'Speech Rate':<12} {'Pitch Var':<12} {'Pauses':<12} {'Output':<12} {'Interpretation':<20}")
print("-" * 80)

for inp in test_inputs:
    out = single_neuron_output(inp, weights_single, bias_single)
    interp = "Stressed" if out > 0.5 else "Calm"
    print(f"{inp[0]:>10.2f}   {inp[1]:>10.2f}   {inp[2]:>10.2f}   {out:>10.4f}   {interp:<20}")

print(f"\nOBSERVATION: Overlaps occur - single neuron fails to separate complex patterns!")

# ============================================================================
# PART B: Add a Hidden Layer (Manual Forward Pass)
# ============================================================================
print(f"\n{'='*70}")
print("PART B: Multi-Layer Perceptron (MLP) with Hidden Layer")
print(f"{'='*70}")

# Network architecture:
# Input layer: 3 features
# Hidden layer: 2 neurons
# Output layer: 1 neuron

# Hidden layer weights (2 neurons, each with 3 inputs + bias)
# Neuron 1: Specialized for detecting fast speech patterns
W_hidden_1 = np.array([0.5, 0.2, 0.1])  # Emphasizes speech rate
b_hidden_1 = -0.3

# Neuron 2: Specialized for detecting pitch instability
W_hidden_2 = np.array([0.1, 0.6, 0.2])  # Emphasizes pitch variation
b_hidden_2 = -0.4

# Output layer weights (combines hidden layer outputs)
W_output = np.array([0.4, 0.6])  # Combines both hidden neurons
b_output = -0.5

def mlp_forward_pass(inputs):
    """Manual forward pass through MLP"""
    # Hidden layer
    z1 = np.dot(inputs, W_hidden_1) + b_hidden_1
    z2 = np.dot(inputs, W_hidden_2) + b_hidden_2
    a1 = 1 / (1 + np.exp(-z1))  # Sigmoid activation
    a2 = 1 / (1 + np.exp(-z2))  # Sigmoid activation
    
    hidden_outputs = np.array([a1, a2])
    
    # Output layer
    z_out = np.dot(hidden_outputs, W_output) + b_output
    output = 1 / (1 + np.exp(-z_out))  # Sigmoid activation
    
    return hidden_outputs, output

print(f"\nHidden Layer Weights:")
print(f"  Neuron 1 (Fast Speech Detector):")
print(f"    Weights: {W_hidden_1}, Bias: {b_hidden_1}")
print(f"  Neuron 2 (Pitch Instability Detector):")
print(f"    Weights: {W_hidden_2}, Bias: {b_hidden_2}")

print(f"\nOutput Layer Weights:")
print(f"  Weights: {W_output}, Bias: {b_output}")

# Test with same inputs
print(f"\n{'='*70}")
print("MLP Results (with Hidden Layer):")
print(f"{'='*70}")

for i, inp in enumerate(test_inputs):
    hidden_outs, final_out = mlp_forward_pass(inp)
    single_out = single_neuron_output(inp, weights_single, bias_single)
    
    print(f"\nExample {i+1}:")
    print(f"  Input: Speech Rate={inp[0]:.2f}, Pitch Var={inp[1]:.2f}, Pauses={inp[2]:.2f}")
    print(f"  Hidden Layer Outputs: [{hidden_outs[0]:.4f}, {hidden_outs[1]:.4f}]")
    print(f"  Final Output (MLP): {final_out:.4f}")
    print(f"  Single Neuron Output: {single_out:.4f}")
    print(f"  Difference: {abs(final_out - single_out):.4f}")

# ============================================================================
# Visualization: Comparison
# ============================================================================
# Create a grid of inputs to visualize decision boundaries
speech_rates = np.linspace(0, 1, 20)
pitch_vars = np.linspace(0, 1, 20)
pause_durs = 0.5  # Fixed pause duration for 2D visualization

single_neuron_grid = np.zeros((20, 20))
mlp_grid = np.zeros((20, 20))

for i, sr in enumerate(speech_rates):
    for j, pv in enumerate(pitch_vars):
        test_input = np.array([sr, pv, pause_durs])
        
        # Single neuron
        single_neuron_grid[i, j] = single_neuron_output(test_input, weights_single, bias_single)
        
        # MLP
        _, mlp_output = mlp_forward_pass(test_input)
        mlp_grid[i, j] = mlp_output

plt.figure(figsize=(14, 6))

# Single neuron decision boundary
plt.subplot(1, 2, 1)
plt.contourf(speech_rates, pitch_vars, single_neuron_grid, levels=20, cmap='RdYlGn')
plt.colorbar(label='Output (Stress Probability)')
plt.xlabel('Speech Rate', fontsize=11, fontweight='bold')
plt.ylabel('Pitch Variation', fontsize=11, fontweight='bold')
plt.title('Single Neuron: Linear Decision Boundary', fontsize=12, fontweight='bold')
plt.scatter([0.9, 0.8], [0.2, 0.25], c='blue', s=100, marker='o', label='Calm Fast', edgecolors='black')
plt.scatter([0.2, 0.3], [0.9, 0.85], c='red', s=100, marker='s', label='Stressed Slow', edgecolors='black')
plt.legend()
plt.grid(True, alpha=0.3)

# MLP decision boundary
plt.subplot(1, 2, 2)
plt.contourf(speech_rates, pitch_vars, mlp_grid, levels=20, cmap='RdYlGn')
plt.colorbar(label='Output (Stress Probability)')
plt.xlabel('Speech Rate', fontsize=11, fontweight='bold')
plt.ylabel('Pitch Variation', fontsize=11, fontweight='bold')
plt.title('MLP with Hidden Layer: Non-Linear Decision Boundary', fontsize=12, fontweight='bold')
plt.scatter([0.9, 0.8], [0.2, 0.25], c='blue', s=100, marker='o', label='Calm Fast', edgecolors='black')
plt.scatter([0.2, 0.3], [0.9, 0.85], c='red', s=100, marker='s', label='Stressed Slow', edgecolors='black')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task3_mlp_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Reflection Question: Why Does Adding a Layer Help?
# ============================================================================
print(f"\n{'='*70}")
print("REFLECTION: Why Does Adding a Layer Help Even Without New Data?")
print(f"{'='*70}")

print(f"\n1. FEATURE TRANSFORMATION:")
print(f"   - Hidden layer transforms raw inputs into new feature representations")
print(f"   - Neuron 1 learns: 'fast speech patterns'")
print(f"   - Neuron 2 learns: 'pitch instability patterns'")
print(f"   - These are INTERMEDIATE features, not raw inputs")

print(f"\n2. SPACE BENDING:")
print(f"   - Single neuron: Can only create a straight line (linear boundary)")
print(f"   - Hidden layer: Creates curved boundaries (non-linear)")
print(f"   - The combination of two neurons allows the network to 'bend'")
print(f"     the decision space, creating complex regions")

print(f"\n3. COMBINATION OF SOFT DECISIONS:")
print(f"   - Each hidden neuron makes a soft decision (0 to 1)")
print(f"   - Output neuron combines these soft decisions")
print(f"   - Example: 'If fast speech AND low pitch variation → calm'")
print(f"   - This logic emerges from the weight combinations")

print(f"\n4. NON-LINEAR COMPOSITION:")
print(f"   - Single neuron: f(x) = sigmoid(W·x + b) → still linear in x")
print(f"   - MLP: f(x) = sigmoid(W2 · sigmoid(W1·x + b1) + b2)")
print(f"   - The nested sigmoids create non-linearity")
print(f"   - Multiple layers = multiple non-linear transformations")

print(f"\n{'='*70}")
print("KEY INSIGHT:")
print(f"{'='*70}")
print("Depth allows the network to learn HIERARCHICAL features:")
print("  Layer 1: Low-level features (speech rate, pitch, pauses)")
print("  Layer 2: Mid-level features (fast speech pattern, pitch instability)")
print("  Layer 3: High-level decision (stress vs calm)")
print("\nThis hierarchical learning is why deep networks can solve complex problems!")
