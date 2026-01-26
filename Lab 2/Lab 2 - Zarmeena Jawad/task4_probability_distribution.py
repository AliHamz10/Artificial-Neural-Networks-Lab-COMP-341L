"""
Lab 2 - Task 4: Softmax for Probability Distribution Generation
Author: Zarmeena Jawad
Roll Number: B23F0115AI125
Section: AI Red

This implementation demonstrates softmax transformation for converting raw
network outputs into valid probability distributions for multi-class scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# SCENARIO: Multi-Class Emotional State Classification
# ============================================================================
# Network classifies three emotional states:
#   - Relaxed (class 0)
#   - Anxious (class 1)
#   - Stressed (class 2)

print("=" * 75)
print("TASK 4: Softmax for Multi-Class Probability Distribution")
print("=" * 75)

# ============================================================================
# STEP 1: Manual Softmax Computation
# ============================================================================
# Raw network outputs (logits) for each class
logit_relaxed = 2.3    # Raw score for "Relaxed"
logit_anxious = 1.6    # Raw score for "Anxious"
logit_stressed = 0.7   # Raw score for "Stressed"

raw_logits = np.array([logit_relaxed, logit_anxious, logit_stressed])

print(f"\nRaw Network Outputs (Logits):")
print(f"  Relaxed: {logit_relaxed:.2f}")
print(f"  Anxious: {logit_anxious:.2f}")
print(f"  Stressed: {logit_stressed:.2f}")

print(f"\n{'='*75}")
print("STEP 1: Step-by-Step Softmax Calculation")
print(f"{'='*75}")

# Manual computation steps
print(f"\nStep 1: Maximum subtraction (numerical stability)")
max_logit = np.max(raw_logits)
shifted_logits = raw_logits - max_logit
print(f"  Maximum value: {max_logit:.2f}")
print(f"  Shifted logits: {shifted_logits}")

print(f"\nStep 2: Exponential computation")
exp_values = np.exp(shifted_logits)
print(f"  exp(Relaxed - max): {exp_values[0]:.6f}")
print(f"  exp(Anxious - max): {exp_values[1]:.6f}")
print(f"  exp(Stressed - max): {exp_values[2]:.6f}")

print(f"\nStep 3: Normalization sum")
normalization_sum = np.sum(exp_values)
print(f"  Sum of exponentials: {normalization_sum:.6f}")

print(f"\nStep 4: Probability computation")
softmax_probabilities = exp_values / normalization_sum
print(f"  P(Relaxed) = {exp_values[0]:.6f} / {normalization_sum:.6f} = {softmax_probabilities[0]:.6f}")
print(f"  P(Anxious) = {exp_values[1]:.6f} / {normalization_sum:.6f} = {softmax_probabilities[1]:.6f}")
print(f"  P(Stressed) = {exp_values[2]:.6f} / {normalization_sum:.6f} = {softmax_probabilities[2]:.6f}")

# ============================================================================
# STEP 2: Probability Property Verification
# ============================================================================
print(f"\n{'='*75}")
print("STEP 2: Probability Distribution Verification")
print(f"{'='*75}")

# Verification 1: Range check
all_in_range = np.all((softmax_probabilities >= 0) & (softmax_probabilities <= 1))
print(f"\nVerification 1: All probabilities in [0, 1]?")
print(f"  P(Relaxed) = {softmax_probabilities[0]:.6f} ∈ [0, 1]? {0 <= softmax_probabilities[0] <= 1}")
print(f"  P(Anxious) = {softmax_probabilities[1]:.6f} ∈ [0, 1]? {0 <= softmax_probabilities[1] <= 1}")
print(f"  P(Stressed) = {softmax_probabilities[2]:.6f} ∈ [0, 1]? {0 <= softmax_probabilities[2] <= 1}")
print(f"  Result: {'✓ VALID' if all_in_range else '✗ INVALID'}")

# Verification 2: Sum check
probability_sum = np.sum(softmax_probabilities)
print(f"\nVerification 2: Sum equals 1.0?")
print(f"  Sum = {probability_sum:.10f}")
print(f"  Result: {'✓ VALID' if np.isclose(probability_sum, 1.0, atol=1e-10) else '✗ INVALID'}")

# ============================================================================
# Softmax Function Implementation
# ============================================================================
def compute_softmax(logits):
    """
    Transform raw logits into probability distribution.
    
    Parameters:
        logits: Array of raw network outputs
    
    Returns:
        Array of probabilities (sums to 1.0)
    """
    # Maximum subtraction for numerical stability
    shifted = logits - np.max(logits)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted)

# Test implementation
computed_probs = compute_softmax(raw_logits)
print(f"\n{'='*75}")
print("Softmax Function Result:")
print(f"{'='*75}")
print(f"  P(Relaxed) = {computed_probs[0]:.6f} ({computed_probs[0]*100:.2f}%)")
print(f"  P(Anxious) = {computed_probs[1]:.6f} ({computed_probs[1]*100:.2f}%)")
print(f"  P(Stressed) = {computed_probs[2]:.6f} ({computed_probs[2]*100:.2f}%)")
print(f"  Total: {np.sum(computed_probs):.10f}")

# ============================================================================
# INTERPRETATION: Probability Redistribution Mechanism
# ============================================================================
print(f"\n{'='*75}")
print("INTERPRETATION: Why Increasing One Score Reduces Others?")
print(f"{'='*75}")

# Experiment: Increase Relaxed score
print(f"\nExperiment: Increase Relaxed logit from {logit_relaxed:.2f} to {logit_relaxed + 1.2:.2f}")

original_probs = compute_softmax(raw_logits)
modified_logits = np.array([logit_relaxed + 1.2, logit_anxious, logit_stressed])
modified_probs = compute_softmax(modified_logits)

print(f"\nOriginal Probabilities:")
print(f"  P(Relaxed): {original_probs[0]:.6f}")
print(f"  P(Anxious): {original_probs[1]:.6f}")
print(f"  P(Stressed): {original_probs[2]:.6f}")

print(f"\nModified Probabilities (after increasing Relaxed logit):")
print(f"  P(Relaxed): {modified_probs[0]:.6f} (change: {modified_probs[0] - original_probs[0]:+.6f})")
print(f"  P(Anxious): {modified_probs[1]:.6f} (change: {modified_probs[1] - original_probs[1]:+.6f})")
print(f"  P(Stressed): {modified_probs[2]:.6f} (change: {modified_probs[2] - original_probs[2]:+.6f})")

print(f"\n{'='*75}")
print("EXPLANATION:")
print(f"{'='*75}")

print(f"\n1. COMPETITIVE ALLOCATION:")
print(f"   - Softmax operates as a zero-sum probability allocation")
print(f"   - Total probability must always equal 1.0 (probability constraint)")
print(f"   - If P(Relaxed) increases, remaining probability must decrease")
print(f"   - Creates competitive dynamics between classes")

print(f"\n2. RELATIVE MAGNITUDE INTERPRETATION:")
print(f"   - Softmax focuses on RELATIVE differences, not absolute values")
print(f"   - Compares: 'How much larger is logit_relaxed vs others?'")
print(f"   - Increasing logit_relaxed makes it relatively larger")
print(f"   - This increases P(Relaxed) while proportionally decreasing others")

print(f"\n3. PROBABILITY SHARING MECHANISM:")
print(f"   - Unlike hard classification (winner-takes-all), softmax distributes")
print(f"     probability across ALL classes")
print(f"   - The 'sharing' means classes compete for probability mass")
print(f"   - When one class gains confidence, others must lose proportionally")
print(f"   - Reflects uncertainty: '58% Relaxed, 29% Anxious, 13% Stressed'")

# ============================================================================
# VISUALIZATION: Softmax Behavior
# ============================================================================
# Visualize how changing one logit affects all probabilities
relaxed_logit_range = np.linspace(0, 5, 120)
anxious_logit_fixed = 1.6
stressed_logit_fixed = 0.7

prob_relaxed_curve = []
prob_anxious_curve = []
prob_stressed_curve = []

for relaxed_val in relaxed_logit_range:
    logit_array = np.array([relaxed_val, anxious_logit_fixed, stressed_logit_fixed])
    probs = compute_softmax(logit_array)
    prob_relaxed_curve.append(probs[0])
    prob_anxious_curve.append(probs[1])
    prob_stressed_curve.append(probs[2])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Probability curves
axes[0].plot(relaxed_logit_range, prob_relaxed_curve, 'b-', linewidth=2.8, label='P(Relaxed)')
axes[0].plot(relaxed_logit_range, prob_anxious_curve, 'g-', linewidth=2.8, label='P(Anxious)')
axes[0].plot(relaxed_logit_range, prob_stressed_curve, 'r-', linewidth=2.8, label='P(Stressed)')
axes[0].axvline(x=logit_relaxed, color='black', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Original logit={logit_relaxed:.1f}')
axes[0].set_xlabel('Relaxed Logit Value', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Probability', fontsize=12, fontweight='bold')
axes[0].set_title('Softmax: Class Competition Dynamics', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.05, 1.05)

# Bar chart comparison
axes[1].bar(['Relaxed', 'Anxious', 'Stressed'], original_probs, width=0.35,
            label='Original', alpha=0.8, color=['blue', 'green', 'red'])
axes[1].bar(['Relaxed', 'Anxious', 'Stressed'], modified_probs, width=0.35,
            label='After Increasing Relaxed', alpha=0.8,
            color=['lightblue', 'lightgreen', 'lightcoral'], align='edge')
axes[1].set_xlabel('Emotional State', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Probability', fontsize=12, fontweight='bold')
axes[1].set_title('Probability Redistribution', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('task4_softmax_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*75}")
print("CONCLUSION:")
print(f"{'='*75}")
print("Softmax transforms raw logits into valid probability distributions where:")
print("  - All probabilities sum to 1.0 (valid probability distribution)")
print("  - Classes compete for probability allocation")
print("  - Increasing one logit automatically redistributes probabilities")
print("  - Provides interpretable confidence scores for each class")
