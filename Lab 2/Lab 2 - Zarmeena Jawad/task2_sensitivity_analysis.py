"""
Lab 2 - Task 2: Sensitivity Analysis Through Threshold Variation
Author: Zarmeena Jawad
Roll Number: B23F0115AI125
Section: AI Red

This experiment investigates how threshold offset (bias) influences neuron
activation patterns across different response functions.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# FIXED PARAMETERS: Maintaining Consistency
# ============================================================================
# Keep acoustic features and connection strengths constant
# Only threshold offset varies

# Fixed acoustic measurements
acoustic_features = np.array([0.78, 0.65, 0.30])

# Fixed connection strengths
synaptic_weights = np.array([0.35, 0.55, 0.10])

# Base integrated signal (without threshold)
base_signal = np.dot(acoustic_features, synaptic_weights)

print("=" * 75)
print("TASK 2: Sensitivity Analysis - Threshold Variation")
print("=" * 75)
print(f"\nFixed Parameters:")
print(f"  Acoustic Features: {acoustic_features}")
print(f"  Connection Strengths: {synaptic_weights}")
print(f"  Base Signal (without threshold): {base_signal:.4f}")

# ============================================================================
# THRESHOLD SWEEP: Systematic Variation
# ============================================================================
# Vary threshold from strongly negative to strongly positive
threshold_values = np.linspace(-3.5, 3.5, 400)  # 400 points for smooth visualization

# Compute total signal for each threshold value
total_signals = base_signal + threshold_values

# ============================================================================
# RESPONSE FUNCTIONS
# ============================================================================

def logistic_activation(z):
    """Logistic response function"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def tanh_activation(z):
    """Hyperbolic tangent response function"""
    return np.tanh(z)

def relu_activation(z):
    """Rectified linear response function"""
    return np.maximum(0.0, z)

# Compute responses for each threshold value
logistic_responses = logistic_activation(total_signals)
tanh_responses = tanh_activation(total_signals)
relu_responses = relu_activation(total_signals)

# ============================================================================
# VISUALIZATION: Response Patterns
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Logistic response
axes[0, 0].plot(threshold_values, logistic_responses, 'b-', linewidth=2.8, label='Logistic Response')
axes[0, 0].axvline(x=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
axes[0, 0].axhline(y=0.5, color='red', linestyle='--', linewidth=1.8, alpha=0.75,
                   label='Midpoint (0.5)')
axes[0, 0].set_xlabel('Threshold Offset', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Neuron Response', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Logistic: Progressive Activation', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=10)
axes[0, 0].set_xlim(-3.5, 3.5)
axes[0, 0].set_ylim(-0.1, 1.1)

# Find activation midpoint
logistic_midpoint_idx = np.argmin(np.abs(logistic_responses - 0.5))
logistic_midpoint_threshold = threshold_values[logistic_midpoint_idx]
axes[0, 0].plot(logistic_midpoint_threshold, 0.5, 'ro', markersize=12,
                label=f'Midpoint: {logistic_midpoint_threshold:.2f}', zorder=5)
axes[0, 0].legend(fontsize=10)

# ReLU response
axes[0, 1].plot(threshold_values, relu_responses, 'g-', linewidth=2.8, label='ReLU Response')
axes[0, 1].axvline(x=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
activation_threshold = -base_signal
axes[0, 1].axvline(x=activation_threshold, color='red', linestyle='--', linewidth=1.8,
                   alpha=0.75, label=f'Activation Point ({activation_threshold:.2f})')
axes[0, 1].set_xlabel('Threshold Offset', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Neuron Response', fontsize=12, fontweight='bold')
axes[0, 1].set_title('ReLU: Instant Activation', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=10)
axes[0, 1].set_xlim(-3.5, 3.5)
axes[0, 1].set_ylim(-0.2, max(relu_responses) + 0.3)

axes[0, 1].plot(activation_threshold, 0, 'ro', markersize=12,
                label=f'Activation: {activation_threshold:.2f}', zorder=5)
axes[0, 1].legend(fontsize=10)

# Tanh response
axes[1, 0].plot(threshold_values, tanh_responses, 'm-', linewidth=2.8, label='Tanh Response')
axes[1, 0].axvline(x=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1.8, alpha=0.75,
                    label='Neutral (0)')
axes[1, 0].set_xlabel('Threshold Offset', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Neuron Response', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Tanh: Balanced Response Zone', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)
axes[1, 0].set_xlim(-3.5, 3.5)
axes[1, 0].set_ylim(-1.1, 1.1)

# Find neutral point
tanh_neutral_idx = np.argmin(np.abs(tanh_responses))
tanh_neutral_threshold = threshold_values[tanh_neutral_idx]
axes[1, 0].plot(tanh_neutral_threshold, 0, 'ro', markersize=12,
                label=f'Neutral: {tanh_neutral_threshold:.2f}', zorder=5)
axes[1, 0].legend(fontsize=10)

# Comparative overlay
axes[1, 1].plot(threshold_values, logistic_responses, 'b-', linewidth=2.2,
                label='Logistic', alpha=0.85)
axes[1, 1].plot(threshold_values, tanh_responses, 'm-', linewidth=2.2,
                label='Tanh', alpha=0.85)
axes[1, 1].plot(threshold_values, relu_responses, 'g-', linewidth=2.2,
                label='ReLU', alpha=0.85)
axes[1, 1].axvline(x=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
axes[1, 1].set_xlabel('Threshold Offset', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Neuron Response', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Comparative Analysis: All Response Functions', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=10)
axes[1, 1].set_xlim(-3.5, 3.5)

plt.tight_layout()
plt.savefig('task2_threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# ANALYSIS: Activation Characteristics
# ============================================================================
print(f"\n" + "=" * 75)
print("ANALYSIS: When Does Each Neuron Become Active?")
print("=" * 75)

print(f"\n1. LOGISTIC - Progressive Activation:")
print(f"   Activation threshold: {logistic_midpoint_threshold:.4f}")
print(f"   At this threshold, response = 0.5 (50% confidence level)")
print(f"   Why progressive?")
print(f"   - Logistic function forms a smooth S-curve")
print(f"   - No abrupt transitions, response evolves gradually from 0 to 1")
print(f"   - Even before midpoint, neuron provides small but meaningful responses")
print(f"   - After midpoint, response increases smoothly rather than instantaneously")

print(f"\n2. ReLU - Immediate Activation:")
print(f"   Activation threshold: {activation_threshold:.4f}")
print(f"   At this threshold, total signal becomes positive")
print(f"   Why immediate?")
print(f"   - ReLU implements a hard cutoff at signal = 0")
print(f"   - For negative signals: response = 0 (completely inactive)")
print(f"   - For positive signals: response = signal (linear scaling)")
print(f"   - No gradual transition zone, binary-like behavior")
print(f"   - Switches from completely OFF to linearly ON instantly")

print(f"\n3. TANH - Balanced Response Zone:")
print(f"   Neutral threshold: {tanh_neutral_threshold:.4f}")
print(f"   At this threshold, response = 0 (neither positive nor negative)")
print(f"   Why balanced zone?")
print(f"   - Tanh produces outputs in [-1, +1] range, centered at 0")
print(f"   - Negative responses represent signal suppression/inhibition")
print(f"   - Positive responses represent signal enhancement/amplification")
print(f"   - Zero represents neutral state (no net effect)")
print(f"   - Activation occurs when crossing from negative to positive region")

print(f"\n" + "=" * 75)
print("KEY OBSERVATION:")
print("=" * 75)
print("Threshold offset determines WHEN activation occurs, but the NATURE of")
print("activation depends on the response function:")
print("- Logistic: Soft, progressive response (probabilistic interpretation)")
print("- ReLU: Hard, immediate response (binary-like decision)")
print("- Tanh: Balanced response with neutral region (signed activation)")

# ============================================================================
# DETAILED EXAMINATION: Response at Specific Thresholds
# ============================================================================
print(f"\n" + "=" * 75)
print("DETAILED EXAMINATION: Response at Key Threshold Values")
print("=" * 75)

key_thresholds = [-2.0, -1.0, 0.0, 1.0, 2.0]
print(f"\n{'Threshold':<12} {'Logistic':<14} {'Tanh':<14} {'ReLU':<14}")
print("-" * 60)

for thresh in key_thresholds:
    signal = base_signal + thresh
    log_resp = logistic_activation(signal)
    tanh_resp = tanh_activation(signal)
    relu_resp = relu_activation(signal)
    print(f"{thresh:>10.1f}   {log_resp:>12.4f}   {tanh_resp:>12.4f}   {relu_resp:>12.4f}")

print(f"\nObservation:")
print(f"- At strongly negative threshold (-2.0): All responses are minimal/suppressed")
print(f"- At zero threshold (0.0): Logistic ~0.5, Tanh ~0, ReLU depends on base signal")
print(f"- At strongly positive threshold (2.0): All responses are substantial/active")
