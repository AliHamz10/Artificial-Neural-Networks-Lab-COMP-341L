"""
Lab 2 - Task 3: Limitations of Single Neurons and Multi-Layer Solutions
Author: Zarmeena Jawad
Roll Number: B23F0115AI125
Section: AI Red

This task demonstrates why single neurons fail for complex classification
and how layered architectures enable sophisticated pattern recognition.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PROBLEM SCENARIO: Confusing Speech Patterns
# ============================================================================
# Two speech patterns that confuse a single neuron:
# Pattern A: Relaxed rapid speakers
#   - High vocal tempo (fast delivery)
#   - Low frequency instability (stable pitch)
#   - Short silence gaps (natural flow)
#
# Pattern B: Anxious slow speakers
#   - Low vocal tempo (hesitant delivery)
#   - High frequency instability (trembling voice)
#   - Long silence gaps (uncertain pauses)

print("=" * 75)
print("TASK 3: Single Neuron Limitations and Layered Solutions")
print("=" * 75)

# ============================================================================
# PART A: Single Neuron Attempt
# ============================================================================
print(f"\n{'='*75}")
print("PART A: Single Neuron Analysis")
print(f"{'='*75}")

# Sample input patterns
relaxed_fast = np.array([0.88, 0.18, 0.28])    # Fast but relaxed
anxious_slow = np.array([0.15, 0.92, 0.85])   # Slow but anxious

# Single neuron configuration (attempting stress detection)
single_weights = np.array([0.25, 0.65, 0.10])  # Emphasize frequency instability
single_threshold = -0.15

# Single neuron computation
def compute_single_neuron(input_features, weights, threshold):
    """Compute output of a single neuron"""
    evidence = np.dot(input_features, weights) + threshold
    return 1.0 / (1.0 + np.exp(-evidence))  # Logistic activation

output_relaxed = compute_single_neuron(relaxed_fast, single_weights, single_threshold)
output_anxious = compute_single_neuron(anxious_slow, single_weights, single_threshold)

print(f"\nPattern A (Relaxed but Fast):")
print(f"  Features: Tempo={relaxed_fast[0]:.2f}, Freq Instability={relaxed_fast[1]:.2f}, Gaps={relaxed_fast[2]:.2f}")
print(f"  Single Neuron Output: {output_relaxed:.4f}")

print(f"\nPattern B (Anxious but Slow):")
print(f"  Features: Tempo={anxious_slow[0]:.2f}, Freq Instability={anxious_slow[1]:.2f}, Gaps={anxious_slow[2]:.2f}")
print(f"  Single Neuron Output: {output_anxious:.4f}")

print(f"\nISSUE IDENTIFIED:")
print(f"  Outputs are nearly identical ({output_relaxed:.4f} vs {output_anxious:.4f})")
print(f"  The neuron cannot differentiate between:")
print(f"    - High tempo + low instability (relaxed fast speech)")
print(f"    - Low tempo + high instability (anxious slow speech)")
print(f"  Reason: Single neuron can only form LINEAR separation boundaries!")

# Test multiple examples
test_patterns = [
    np.array([0.88, 0.18, 0.28]),  # Relaxed fast
    np.array([0.85, 0.22, 0.35]),  # Relaxed fast (variant)
    np.array([0.15, 0.92, 0.85]),  # Anxious slow
    np.array([0.22, 0.88, 0.78]),  # Anxious slow (variant)
    np.array([0.55, 0.55, 0.55]),  # Ambiguous middle
]

print(f"\n{'='*75}")
print("Testing Multiple Patterns with Single Neuron:")
print(f"{'='*75}")
print(f"{'Tempo':<10} {'Freq Inst':<12} {'Gaps':<10} {'Output':<12} {'Classification':<20}")
print("-" * 75)

for pattern in test_patterns:
    result = compute_single_neuron(pattern, single_weights, single_threshold)
    classification = "Anxious" if result > 0.5 else "Relaxed"
    print(f"{pattern[0]:>8.2f}   {pattern[1]:>10.2f}   {pattern[2]:>8.2f}   {result:>10.4f}   {classification:<20}")

print(f"\nOBSERVATION: Significant overlap - single neuron fails on complex patterns!")

# ============================================================================
# PART B: Layered Network Architecture
# ============================================================================
print(f"\n{'='*75}")
print("PART B: Multi-Layer Perceptron with Intermediate Layer")
print(f"{'='*75}")

# Network structure:
# Input: 3 acoustic features
# Intermediate layer: 2 specialized neurons
# Output: 1 decision neuron

# Intermediate layer neuron configurations
# Neuron H1: Detects rapid speech characteristics
W_h1 = np.array([0.6, 0.15, 0.1])  # Strong emphasis on tempo
b_h1 = -0.35

# Neuron H2: Detects frequency instability patterns
W_h2 = np.array([0.12, 0.7, 0.15])  # Strong emphasis on frequency instability
b_h2 = -0.45

# Output layer configuration (combines intermediate outputs)
W_output = np.array([0.45, 0.55])  # Combines both intermediate neurons
b_output = -0.6

def layered_network_forward(input_features):
    """Forward propagation through layered network"""
    # Intermediate layer processing
    evidence_h1 = np.dot(input_features, W_h1) + b_h1
    evidence_h2 = np.dot(input_features, W_h2) + b_h2
    activation_h1 = 1.0 / (1.0 + np.exp(-evidence_h1))  # Logistic
    activation_h2 = 1.0 / (1.0 + np.exp(-evidence_h2))  # Logistic
    
    intermediate_outputs = np.array([activation_h1, activation_h2])
    
    # Output layer processing
    evidence_output = np.dot(intermediate_outputs, W_output) + b_output
    final_output = 1.0 / (1.0 + np.exp(-evidence_output))  # Logistic
    
    return intermediate_outputs, final_output

print(f"\nIntermediate Layer Configuration:")
print(f"  Neuron H1 (Tempo Specialist):")
print(f"    Weights: {W_h1}, Threshold: {b_h1}")
print(f"  Neuron H2 (Instability Specialist):")
print(f"    Weights: {W_h2}, Threshold: {b_h2}")

print(f"\nOutput Layer Configuration:")
print(f"    Weights: {W_output}, Threshold: {b_output}")

# Test with same patterns
print(f"\n{'='*75}")
print("Layered Network Results:")
print(f"{'='*75}")

for idx, pattern in enumerate(test_patterns):
    intermediate_acts, final_result = layered_network_forward(pattern)
    single_result = compute_single_neuron(pattern, single_weights, single_threshold)
    
    print(f"\nPattern {idx+1}:")
    print(f"  Input: Tempo={pattern[0]:.2f}, Freq Inst={pattern[1]:.2f}, Gaps={pattern[2]:.2f}")
    print(f"  Intermediate Activations: [{intermediate_acts[0]:.4f}, {intermediate_acts[1]:.4f}]")
    print(f"  Final Output (Layered): {final_result:.4f}")
    print(f"  Single Neuron Output: {single_result:.4f}")
    print(f"  Improvement: {abs(final_result - single_result):.4f}")

# ============================================================================
# VISUALIZATION: Decision Boundary Comparison
# ============================================================================
# Create feature space grid for visualization
tempo_range = np.linspace(0, 1, 25)
freq_inst_range = np.linspace(0, 1, 25)
gap_fixed = 0.5  # Fixed gap value for 2D visualization

single_neuron_surface = np.zeros((25, 25))
layered_surface = np.zeros((25, 25))

for i, tempo_val in enumerate(tempo_range):
    for j, freq_inst_val in enumerate(freq_inst_range):
        test_input = np.array([tempo_val, freq_inst_val, gap_fixed])
        
        # Single neuron
        single_neuron_surface[i, j] = compute_single_neuron(test_input, single_weights, single_threshold)
        
        # Layered network
        _, layered_output = layered_network_forward(test_input)
        layered_surface[i, j] = layered_output

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Single neuron decision surface
im1 = axes[0].contourf(tempo_range, freq_inst_range, single_neuron_surface, levels=25, cmap='RdYlGn')
axes[0].set_xlabel('Vocal Tempo', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency Instability', fontsize=12, fontweight='bold')
axes[0].set_title('Single Neuron: Linear Decision Boundary', fontsize=13, fontweight='bold')
plt.colorbar(im1, ax=axes[0], label='Output (Anxiety Probability)')
axes[0].scatter([0.88, 0.85], [0.18, 0.22], c='blue', s=120, marker='o',
                label='Relaxed Fast', edgecolors='black', linewidth=2)
axes[0].scatter([0.15, 0.22], [0.92, 0.88], c='red', s=120, marker='s',
                label='Anxious Slow', edgecolors='black', linewidth=2)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Layered network decision surface
im2 = axes[1].contourf(tempo_range, freq_inst_range, layered_surface, levels=25, cmap='RdYlGn')
axes[1].set_xlabel('Vocal Tempo', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency Instability', fontsize=12, fontweight='bold')
axes[1].set_title('Layered Network: Non-Linear Decision Boundary', fontsize=13, fontweight='bold')
plt.colorbar(im2, ax=axes[1], label='Output (Anxiety Probability)')
axes[1].scatter([0.88, 0.85], [0.18, 0.22], c='blue', s=120, marker='o',
                label='Relaxed Fast', edgecolors='black', linewidth=2)
axes[1].scatter([0.15, 0.22], [0.92, 0.88], c='red', s=120, marker='s',
                label='Anxious Slow', edgecolors='black', linewidth=2)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task3_network_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# REFLECTION: Why Layering Helps
# ============================================================================
print(f"\n{'='*75}")
print("REFLECTION: Why Does Layering Help Without Additional Data?")
print(f"{'='*75}")

print(f"\n1. REPRESENTATION TRANSFORMATION:")
print(f"   - Intermediate layer converts raw inputs into abstracted features")
print(f"   - H1 learns: 'rapid speech characteristics'")
print(f"   - H2 learns: 'frequency instability patterns'")
print(f"   - These are DERIVED features, not original measurements")

print(f"\n2. GEOMETRIC TRANSFORMATION:")
print(f"   - Single neuron: Limited to straight-line boundaries (linear)")
print(f"   - Layered network: Can create curved, complex boundaries (non-linear)")
print(f"   - Multiple neurons enable 'warping' of decision space")
print(f"   - Creates distinct regions for different pattern types")

print(f"\n3. COMBINATORIAL LOGIC:")
print(f"   - Each intermediate neuron makes a partial decision (0 to 1)")
print(f"   - Output neuron synthesizes these partial decisions")
print(f"   - Example logic: 'If rapid speech AND stable frequency → relaxed'")
print(f"   - This logic emerges from weight interactions")

print(f"\n4. COMPOSITIONAL NON-LINEARITY:")
print(f"   - Single neuron: f(x) = logistic(W·x + b) → linear in x")
print(f"   - Layered: f(x) = logistic(W2 · logistic(W1·x + b1) + b2)")
print(f"   - Nested logistic functions create complex non-linear mappings")
print(f"   - Multiple layers = multiple non-linear transformations")

print(f"\n{'='*75}")
print("CORE INSIGHT:")
print(f"{'='*75}")
print("Layering enables HIERARCHICAL feature learning:")
print("  Level 1: Raw acoustic measurements (tempo, frequency, gaps)")
print("  Level 2: Abstracted patterns (rapid speech, instability)")
print("  Level 3: High-level decision (anxiety vs relaxation)")
print("\nThis hierarchical abstraction is fundamental to deep learning success!")
