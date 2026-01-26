"""
Lab 2 - Task 1: Constructing a Decision-Making Neuron
Author: Zarmeena Jawad
Roll Number: B23F0115AI125
Section: AI Red

This implementation creates an artificial neuron that evaluates emotional stress
by processing acoustic features from voice recordings.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# FEATURE SELECTION: Acoustic Properties for Stress Analysis
# ============================================================================
# Three acoustic measurements extracted from speech signals:
# 1. Vocal tempo (normalized 0-1): Rate of speech delivery
# 2. Frequency instability (normalized 0-1): Variation in fundamental frequency
# 3. Silence intervals (normalized 0-1): Average gap duration between utterances

# Sample measurement from a voice recording
tempo = 0.78        # Moderate-fast speaking pace
freq_instability = 0.65  # Somewhat unstable frequency
silence_gaps = 0.30      # Relatively brief pauses

feature_vector = np.array([tempo, freq_instability, silence_gaps])

print("=" * 75)
print("TASK 1: Constructing a Decision-Making Neuron")
print("=" * 75)
print(f"\nAcoustic Feature Measurements:")
print(f"  Vocal Tempo: {tempo:.2f}")
print(f"  Frequency Instability: {freq_instability:.2f}")
print(f"  Silence Intervals: {silence_gaps:.2f}")

# ============================================================================
# WEIGHT ASSIGNMENT: Determining Feature Significance
# ============================================================================
# Weight rationale:
# - Vocal tempo (0.35): Secondary indicator - rapid speech may suggest urgency
#   but isn't definitive (some individuals naturally speak quickly)
# - Frequency instability (0.55): Primary indicator - emotional stress often
#   manifests as voice tremors and pitch fluctuations
# - Silence intervals (0.10): Tertiary indicator - pause patterns are less
#   consistent across individuals and contexts

connection_strengths = np.array([0.35, 0.55, 0.10])

print(f"\nConnection Strengths (Feature Weights):")
print(f"  Vocal Tempo: {connection_strengths[0]:.2f} (Secondary - may indicate urgency)")
print(f"  Frequency Instability: {connection_strengths[1]:.2f} (PRIMARY - key stress marker)")
print(f"  Silence Intervals: {connection_strengths[2]:.2f} (Tertiary - less reliable)")

# ============================================================================
# THRESHOLD ADJUSTMENT: Setting Activation Sensitivity
# ============================================================================
# Threshold offset = -0.25 (slightly negative)
# Interpretation:
# - Large positive threshold: Overly sensitive, triggers on weak signals
# - Negative threshold: Conservative, demands substantial evidence
# - Our value (-0.25) creates moderate conservatism

threshold_offset = -0.25

print(f"\nThreshold Offset (Bias): {threshold_offset:.2f}")
print(f"  Meaning: Moderately conservative - requires meaningful evidence")
print(f"  Positive threshold → overly sensitive (false alarms)")
print(f"  Negative threshold → conservative (needs strong signals)")

# ============================================================================
# EVIDENCE ACCUMULATION: Computing Integrated Signal
# ============================================================================
integrated_signal = np.dot(feature_vector, connection_strengths) + threshold_offset

print(f"\nIntegrated Signal (Evidence Level): {integrated_signal:.4f}")
print(f"  Computation: ({tempo:.2f} × {connection_strengths[0]:.2f}) + "
      f"({freq_instability:.2f} × {connection_strengths[1]:.2f}) + "
      f"({silence_gaps:.2f} × {connection_strengths[2]:.2f}) + ({threshold_offset:.2f})")
print(f"  = {integrated_signal:.4f}")

# ============================================================================
# ACTIVATION TRANSFORMATIONS: Applying Different Response Functions
# ============================================================================

def logistic_response(z):
    """Logistic function: smooth S-shaped curve mapping to [0, 1]"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def hyperbolic_tangent(z):
    """Hyperbolic tangent: symmetric S-curve mapping to [-1, 1]"""
    return np.tanh(z)

def rectified_linear(z):
    """Rectified Linear Unit: identity for positive, zero for negative"""
    return np.maximum(0.0, z)

# Evaluate neuron response with each transformation
response_logistic = logistic_response(integrated_signal)
response_tanh = hyperbolic_tangent(integrated_signal)
response_relu = rectified_linear(integrated_signal)

print(f"\n" + "=" * 75)
print("ACTIVATION FUNCTION RESPONSES")
print("=" * 75)
print(f"\nLogistic Output: {response_logistic:.4f}")
print(f"Hyperbolic Tangent Output: {response_tanh:.4f}")
print(f"Rectified Linear Output: {response_relu:.4f}")

# ============================================================================
# VISUALIZATION: Response Function Comparison
# ============================================================================
signal_range = np.linspace(-5, 5, 1000)
logistic_curve = logistic_response(signal_range)
tanh_curve = hyperbolic_tangent(signal_range)
relu_curve = rectified_linear(signal_range)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Full range comparison
axes[0].plot(signal_range, logistic_curve, 'b-', linewidth=2.5, label='Logistic', alpha=0.9)
axes[0].plot(signal_range, tanh_curve, 'g-', linewidth=2.5, label='Tanh', alpha=0.9)
axes[0].plot(signal_range, relu_curve, 'r-', linewidth=2.5, label='ReLU', alpha=0.9)
axes[0].axvline(x=integrated_signal, color='purple', linestyle='--', linewidth=2,
                label=f'Our Signal (z={integrated_signal:.2f})', alpha=0.8)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.7, alpha=0.5)
axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.7, alpha=0.5)
axes[0].set_xlabel('Integrated Signal (z)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Neuron Response', fontsize=12, fontweight='bold')
axes[0].set_title('Activation Function Comparison', fontsize=13, fontweight='bold')
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Focused view around our signal
zoom_range = np.linspace(integrated_signal - 1.5, integrated_signal + 1.5, 250)
axes[1].plot(zoom_range, logistic_response(zoom_range), 'b-', linewidth=3, 
             marker='o', markersize=4, label='Logistic', alpha=0.8)
axes[1].plot(zoom_range, hyperbolic_tangent(zoom_range), 'g-', linewidth=3,
             marker='s', markersize=4, label='Tanh', alpha=0.8)
axes[1].plot(zoom_range, rectified_linear(zoom_range), 'r-', linewidth=3,
             marker='^', markersize=4, label='ReLU', alpha=0.8)
axes[1].axvline(x=integrated_signal, color='purple', linestyle='--', linewidth=2.5,
                label=f'Signal z={integrated_signal:.2f}', alpha=0.9)
axes[1].scatter([integrated_signal], [response_logistic], color='blue', s=150,
                zorder=6, edgecolors='black', linewidth=2, label='Logistic Point')
axes[1].scatter([integrated_signal], [response_tanh], color='green', s=150,
                zorder=6, edgecolors='black', linewidth=2, label='Tanh Point')
axes[1].scatter([integrated_signal], [response_relu], color='red', s=150,
                zorder=6, edgecolors='black', linewidth=2, label='ReLU Point')
axes[1].set_xlabel('Integrated Signal (z)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Neuron Response', fontsize=12, fontweight='bold')
axes[1].set_title('Focused View: Neuron Response', fontsize=13, fontweight='bold')
axes[1].legend(loc='best', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_response_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# INTERPRETATION: Understanding Response Differences
# ============================================================================
print(f"\n" + "=" * 75)
print("INTERPRETATION: Why Different Responses from Same Input?")
print("=" * 75)

print(f"\n1. GRADUAL VS ABRUPT TRANSITIONS:")
print(f"   - Logistic ({response_logistic:.4f}): Smooth, continuous response curve")
print(f"     → Provides gradual confidence scaling (0 to 1 probability range)")
print(f"   - Tanh ({response_tanh:.4f}): Also smooth but symmetric around zero")
print(f"     → Can represent both enhancement and suppression (-1 to +1 range)")
print(f"   - ReLU ({response_relu:.4f}): Sharp transition at zero threshold")
print(f"     → Either completely inactive (0) or linearly active (positive values)")

print(f"\n2. CONFIDENCE INTERPRETATION:")
print(f"   - Logistic: {response_logistic:.4f} = {response_logistic*100:.1f}% stress likelihood")
print(f"   - Tanh: {response_tanh:.4f} = moderate enhancement signal")
print(f"   - ReLU: {response_relu:.4f} = {response_relu*100:.1f}% of peak activation level")

print(f"\n3. SIGNAL MODULATION:")
print(f"   - Logistic: Unidirectional amplification (0→1), cannot reduce below zero")
print(f"   - Tanh: Bidirectional modulation (can enhance or suppress signals)")
print(f"   - ReLU: Selective activation (passes positive, blocks negative signals)")

print(f"\n4. NON-LINEAR TRANSFORMATION:")
print(f"   - Logistic: Highly curved S-shape → enables smooth probability mapping")
print(f"   - Tanh: Curved but balanced → symmetric response around neutral point")
print(f"   - ReLU: Piecewise linear → computationally efficient, gradient-friendly")

print(f"\n" + "=" * 75)
print("SUMMARY:")
print("=" * 75)
print("Identical evidence produces distinct outputs because each transformation")
print("interprets the signal differently:")
print("- Logistic: Probability interpretation (gradual confidence)")
print("- Tanh: Signed magnitude interpretation (can be negative)")
print("- ReLU: Raw activation interpretation (only if positive)")
