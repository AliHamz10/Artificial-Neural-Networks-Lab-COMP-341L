"""
Lab 2 - Task 4: Softmax Implementation
Author: Ali Hamza
Roll Number: B23F0063AI106
Section: B.S AI - Red

This task implements softmax activation for multi-class classification,
converting raw network scores into probability distributions.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# SCENARIO: Three Emotional States
# ============================================================================
# The network now predicts three emotional states:
#   - Calm (class 0)
#   - Anxious (class 1)
#   - Stressed (class 2)

print("=" * 70)
print("TASK 4: Softmax for Multi-Class Classification")
print("=" * 70)

# ============================================================================
# STEP 1: Apply Softmax Manually
# ============================================================================
# Given three raw outputs (logits) from the network
z1 = 2.5   # Raw score for "Calm"
z2 = 1.8   # Raw score for "Anxious"
z3 = 0.9   # Raw score for "Stressed"

raw_scores = np.array([z1, z2, z3])

print(f"\nRaw Network Outputs (Logits):")
print(f"  z1 (Calm): {z1:.2f}")
print(f"  z2 (Anxious): {z2:.2f}")
print(f"  z3 (Stressed): {z3:.2f}")

print(f"\n{'='*70}")
print("STEP 1: Manual Softmax Calculation")
print(f"{'='*70}")

# Step-by-step softmax calculation
print(f"\nStep 1: Subtract maximum for numerical stability")
z_max = np.max(raw_scores)
z_shifted = raw_scores - z_max
print(f"  Maximum value: {z_max:.2f}")
print(f"  Shifted scores: {z_shifted}")

print(f"\nStep 2: Compute exponentials")
exp_vals = np.exp(z_shifted)
print(f"  exp(z1 - max): {exp_vals[0]:.6f}")
print(f"  exp(z2 - max): {exp_vals[1]:.6f}")
print(f"  exp(z3 - max): {exp_vals[2]:.6f}")

print(f"\nStep 3: Compute sum of exponentials")
sum_exp = np.sum(exp_vals)
print(f"  Sum: {sum_exp:.6f}")

print(f"\nStep 4: Divide each exponential by the sum")
softmax_probs = exp_vals / sum_exp
print(f"  P(Calm) = {exp_vals[0]:.6f} / {sum_exp:.6f} = {softmax_probs[0]:.6f}")
print(f"  P(Anxious) = {exp_vals[1]:.6f} / {sum_exp:.6f} = {softmax_probs[1]:.6f}")
print(f"  P(Stressed) = {exp_vals[2]:.6f} / {sum_exp:.6f} = {softmax_probs[2]:.6f}")

# ============================================================================
# STEP 2: Verify Probability Behavior
# ============================================================================
print(f"\n{'='*70}")
print("STEP 2: Verification of Probability Properties")
print(f"{'='*70}")

# Check 1: All outputs between 0 and 1
all_valid = np.all((softmax_probs >= 0) & (softmax_probs <= 1))
print(f"\nCheck 1: All probabilities in [0, 1]?")
print(f"  P(Calm) = {softmax_probs[0]:.6f} ∈ [0, 1]? {0 <= softmax_probs[0] <= 1}")
print(f"  P(Anxious) = {softmax_probs[1]:.6f} ∈ [0, 1]? {0 <= softmax_probs[1] <= 1}")
print(f"  P(Stressed) = {softmax_probs[2]:.6f} ∈ [0, 1]? {0 <= softmax_probs[2] <= 1}")
print(f"  Result: {'✓ PASS' if all_valid else '✗ FAIL'}")

# Check 2: Sum equals 1
prob_sum = np.sum(softmax_probs)
print(f"\nCheck 2: Sum of probabilities equals 1?")
print(f"  Sum = {prob_sum:.10f}")
print(f"  Result: {'✓ PASS' if np.isclose(prob_sum, 1.0, atol=1e-10) else '✗ FAIL'}")

# ============================================================================
# Softmax Function Implementation
# ============================================================================
def softmax(z):
    """
    Compute softmax probabilities from raw scores.
    
    Args:
        z: Array of raw scores (logits)
    
    Returns:
        Array of probabilities (sums to 1)
    """
    # Subtract max for numerical stability
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

# Test with our example
softmax_result = softmax(raw_scores)
print(f"\n{'='*70}")
print("Softmax Function Result:")
print(f"{'='*70}")
print(f"  P(Calm) = {softmax_result[0]:.6f} ({softmax_result[0]*100:.2f}%)")
print(f"  P(Anxious) = {softmax_result[1]:.6f} ({softmax_result[1]*100:.2f}%)")
print(f"  P(Stressed) = {softmax_result[2]:.6f} ({softmax_result[2]*100:.2f}%)")
print(f"  Sum: {np.sum(softmax_result):.10f}")

# ============================================================================
# Interpretation Question: Why Does Increasing One Score Reduce Others?
# ============================================================================
print(f"\n{'='*70}")
print("INTERPRETATION: Why Does Increasing One Score Reduce Others?")
print(f"{'='*70}")

# Experiment: Increase z1 (Calm score) and observe changes
print(f"\nExperiment: Increase Calm score from {z1:.2f} to {z1 + 1.0:.2f}")

original_probs = softmax(raw_scores)
modified_scores = np.array([z1 + 1.0, z2, z3])
modified_probs = softmax(modified_scores)

print(f"\nOriginal Probabilities:")
print(f"  P(Calm): {original_probs[0]:.6f}")
print(f"  P(Anxious): {original_probs[1]:.6f}")
print(f"  P(Stressed): {original_probs[2]:.6f}")

print(f"\nModified Probabilities (after increasing Calm score):")
print(f"  P(Calm): {modified_probs[0]:.6f} (change: {modified_probs[0] - original_probs[0]:+.6f})")
print(f"  P(Anxious): {modified_probs[1]:.6f} (change: {modified_probs[1] - original_probs[1]:+.6f})")
print(f"  P(Stressed): {modified_probs[2]:.6f} (change: {modified_probs[2] - original_probs[2]:+.6f})")

print(f"\n{'='*70}")
print("EXPLANATION:")
print(f"{'='*70}")

print(f"\n1. COMPETITION:")
print(f"   - Softmax is a ZERO-SUM game")
print(f"   - Total probability must always equal 1.0")
print(f"   - If P(Calm) increases, the remaining probability must decrease")
print(f"   - This creates competition between classes")

print(f"\n2. RELATIVE CONFIDENCE:")
print(f"   - Softmax doesn't care about absolute scores, only RELATIVE scores")
print(f"   - It compares: 'How much larger is z1 compared to z2 and z3?'")
print(f"   - Increasing z1 makes it relatively larger → P(Calm) increases")
print(f"   - But z2 and z3 become relatively smaller → their probabilities decrease")

print(f"\n3. SOFT MARGINS ACROSS CLASSES:")
print(f"   - Unlike hard classification (winner takes all), softmax gives")
print(f"     probabilities to ALL classes")
print(f"   - The 'soft margin' means classes share the probability space")
print(f"   - When one class gains confidence, others must lose confidence")
print(f"   - This reflects uncertainty: 'I'm 70% sure it's Calm, 20% Anxious, 10% Stressed'")

# ============================================================================
# Visualization: Softmax Behavior
# ============================================================================
# Visualize how changing one score affects all probabilities
z1_range = np.linspace(0, 5, 100)
z2_fixed = 1.8
z3_fixed = 0.9

prob_calm = []
prob_anxious = []
prob_stressed = []

for z1_val in z1_range:
    scores = np.array([z1_val, z2_fixed, z3_fixed])
    probs = softmax(scores)
    prob_calm.append(probs[0])
    prob_anxious.append(probs[1])
    prob_stressed.append(probs[2])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(z1_range, prob_calm, 'b-', linewidth=2.5, label='P(Calm)')
plt.plot(z1_range, prob_anxious, 'g-', linewidth=2.5, label='P(Anxious)')
plt.plot(z1_range, prob_stressed, 'r-', linewidth=2.5, label='P(Stressed)')
plt.axvline(x=z1, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Original z1')
plt.xlabel('z1 (Calm Score)', fontsize=11, fontweight='bold')
plt.ylabel('Probability', fontsize=11, fontweight='bold')
plt.title('Softmax: Competition Between Classes', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 1.05)

# Bar chart comparison
plt.subplot(1, 2, 2)
classes = ['Calm', 'Anxious', 'Stressed']
x_pos = np.arange(len(classes))
width = 0.35

plt.bar(x_pos - width/2, original_probs, width, label='Original', alpha=0.8, color=['blue', 'green', 'red'])
plt.bar(x_pos + width/2, modified_probs, width, label='After Increasing z1', alpha=0.8, color=['lightblue', 'lightgreen', 'lightcoral'])
plt.xlabel('Emotional State', fontsize=11, fontweight='bold')
plt.ylabel('Probability', fontsize=11, fontweight='bold')
plt.title('Probability Redistribution', fontsize=12, fontweight='bold')
plt.xticks(x_pos, classes)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('task4_softmax_behavior.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n{'='*70}")
print("CONCLUSION:")
print(f"{'='*70}")
print("Softmax converts raw scores into a probability distribution where:")
print("  - All probabilities sum to 1 (valid probability distribution)")
print("  - Classes compete for probability mass")
print("  - Increasing one score automatically decreases others")
print("  - This creates interpretable 'confidence' scores for each class")
