"""
Lab 3 - Task 1: Activation Functions Implementation and Visualization
Author: Zarmeena Jawad
Roll Number: B23F0115AI125
Section: AI Red

This script implements and visualizes four common activation functions:
1. Sigmoid (Logistic Function)
2. ReLU (Rectified Linear Unit)
3. Tanh (Hyperbolic Tangent)
4. Leaky ReLU (Parametric ReLU)

Each function is explained with mathematical formulas, use cases, and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# ACTIVATION FUNCTION IMPLEMENTATIONS
# ============================================================================

def sigmoid(x):
    """
    Sigmoid (Logistic) Activation Function
    
    Mathematical Formula: f(x) = 1 / (1 + e^(-x))
    Derivative: f'(x) = f(x) * (1 - f(x))
    
    Properties:
    - Output Range: [0, 1]
    - S-shaped curve (smooth transition)
    - Used in: Binary classification output layers
    - Problem: Suffers from vanishing gradient problem
    
    Args:
        x: Input value or array
        
    Returns:
        Sigmoid output in range [0, 1]
    """
    # Clip values to prevent overflow in exponential
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def relu(x):
    """
    ReLU (Rectified Linear Unit) Activation Function
    
    Mathematical Formula: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0, else 0
    
    Properties:
    - Output Range: [0, ∞)
    - Piecewise linear (identity for positive, zero for negative)
    - Used in: Hidden layers for efficient non-linearity
    - Advantages: Computationally efficient, gradient-friendly
    - Problem: "Dying ReLU" - neurons can become permanently inactive
    
    Args:
        x: Input value or array
        
    Returns:
        ReLU output (x if x > 0, else 0)
    """
    return np.maximum(0, x)


def tanh(x):
    """
    Tanh (Hyperbolic Tangent) Activation Function
    
    Mathematical Formula: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Derivative: f'(x) = 1 - (f(x))^2
    
    Properties:
    - Output Range: [-1, 1]
    - Zero-centered (symmetric around origin)
    - S-shaped curve (smooth transition)
    - Used in: Hidden layers (better than sigmoid for gradients)
    - Advantages: Better gradient properties than sigmoid
    
    Args:
        x: Input value or array
        
    Returns:
        Tanh output in range [-1, 1]
    """
    return np.tanh(x)


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU (Parametric ReLU) Activation Function
    
    Mathematical Formula: 
        f(x) = x if x > 0
        f(x) = alpha * x if x <= 0
    
    Properties:
    - Output Range: (-∞, ∞) for negative, [0, ∞) for positive
    - Small positive slope (alpha) for negative inputs
    - Used in: Hidden layers (addresses dying ReLU problem)
    - Advantages: Prevents neurons from dying, allows small gradients
    
    Args:
        x: Input value or array
        alpha: Small positive constant (default: 0.01)
        
    Returns:
        Leaky ReLU output
    """
    return np.where(x >= 0, x, alpha * x)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_sigmoid():
    """
    Plot Sigmoid activation function with detailed annotations.
    """
    # Generate input values
    x = np.linspace(-10, 10, 1000)
    y = sigmoid(x)
    
    # Create figure
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, 'b-', linewidth=3, label='Sigmoid Function', alpha=0.9)
    
    # Add key points
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='Midpoint (0.5)')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Mark key characteristics
    plt.plot(0, 0.5, 'ro', markersize=10, label='Center Point (0, 0.5)')
    
    # Labels and formatting
    plt.xlabel('Input (x)', fontsize=13, fontweight='bold')
    plt.ylabel('Sigmoid(x)', fontsize=13, fontweight='bold')
    plt.title('Sigmoid (Logistic) Activation Function', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.ylim(-0.1, 1.1)
    plt.xlim(-10, 10)
    
    # Add text annotation
    plt.text(6, 0.2, 'Range: [0, 1]\nUsed: Binary Classification\nIssue: Vanishing Gradient', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('task1_sigmoid.png', dpi=150, bbox_inches='tight')
    print("✓ Sigmoid plot saved as 'task1_sigmoid.png'")
    plt.close()


def plot_relu():
    """
    Plot ReLU activation function with detailed annotations.
    """
    # Generate input values
    x = np.linspace(-10, 10, 1000)
    y = relu(x)
    
    # Create figure
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, 'g-', linewidth=3, label='ReLU Function', alpha=0.9)
    
    # Add key lines
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label='Activation Threshold (x=0)')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Mark key point
    plt.plot(0, 0, 'ro', markersize=10, label='Threshold Point (0, 0)')
    
    # Labels and formatting
    plt.xlabel('Input (x)', fontsize=13, fontweight='bold')
    plt.ylabel('ReLU(x)', fontsize=13, fontweight='bold')
    plt.title('ReLU (Rectified Linear Unit) Activation Function', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.ylim(-1, 10)
    plt.xlim(-10, 10)
    
    # Add text annotation
    plt.text(6, 7, 'Range: [0, ∞)\nUsed: Hidden Layers\nIssue: Dying ReLU', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('task1_relu.png', dpi=150, bbox_inches='tight')
    print("✓ ReLU plot saved as 'task1_relu.png'")
    plt.close()


def plot_tanh():
    """
    Plot Tanh activation function with detailed annotations.
    """
    # Generate input values
    x = np.linspace(-10, 10, 1000)
    y = tanh(x)
    
    # Create figure
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, 'm-', linewidth=3, label='Tanh Function', alpha=0.9)
    
    # Add key lines
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='Zero Line')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.axhline(y=1, color='blue', linestyle=':', linewidth=1, alpha=0.5)
    plt.axhline(y=-1, color='blue', linestyle=':', linewidth=1, alpha=0.5)
    
    # Mark key point
    plt.plot(0, 0, 'ro', markersize=10, label='Center Point (0, 0)')
    
    # Labels and formatting
    plt.xlabel('Input (x)', fontsize=13, fontweight='bold')
    plt.ylabel('Tanh(x)', fontsize=13, fontweight='bold')
    plt.title('Tanh (Hyperbolic Tangent) Activation Function', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.ylim(-1.2, 1.2)
    plt.xlim(-10, 10)
    
    # Add text annotation
    plt.text(6, 0.3, 'Range: [-1, 1]\nZero-Centered\nBetter Gradients than Sigmoid', 
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('task1_tanh.png', dpi=150, bbox_inches='tight')
    print("✓ Tanh plot saved as 'task1_tanh.png'")
    plt.close()


def plot_leaky_relu(alpha=0.01):
    """
    Plot Leaky ReLU activation function with detailed annotations.
    
    Args:
        alpha: Slope for negative inputs (default: 0.01)
    """
    # Generate input values
    x = np.linspace(-10, 10, 1000)
    y = leaky_relu(x, alpha)
    
    # Create figure
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, 'orange', linewidth=3, label=f'Leaky ReLU (α={alpha})', alpha=0.9)
    
    # Add comparison with standard ReLU
    y_standard_relu = relu(x)
    plt.plot(x, y_standard_relu, 'g--', linewidth=2, alpha=0.5, 
             label='Standard ReLU (for comparison)')
    
    # Add key lines
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label='Threshold (x=0)')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Mark key point
    plt.plot(0, 0, 'ro', markersize=10, label='Origin (0, 0)')
    
    # Labels and formatting
    plt.xlabel('Input (x)', fontsize=13, fontweight='bold')
    plt.ylabel('Leaky ReLU(x)', fontsize=13, fontweight='bold')
    plt.title('Leaky ReLU (Parametric ReLU) Activation Function', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    plt.ylim(-0.5, 10)
    plt.xlim(-10, 10)
    
    # Add text annotation
    plt.text(6, 7, f'Range: (-∞, ∞) for x<0\n[0, ∞) for x>0\nSolves: Dying ReLU Problem\nα = {alpha}', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('task1_leaky_relu.png', dpi=150, bbox_inches='tight')
    print("✓ Leaky ReLU plot saved as 'task1_leaky_relu.png'")
    plt.close()


def plot_all_activations():
    """
    Plot all activation functions together for comparison.
    """
    # Generate input values
    x = np.linspace(-10, 10, 1000)
    
    # Compute all functions
    y_sigmoid = sigmoid(x)
    y_relu = relu(x)
    y_tanh = tanh(x)
    y_leaky = leaky_relu(x, alpha=0.01)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sigmoid
    axes[0, 0].plot(x, y_sigmoid, 'b-', linewidth=2.5, label='Sigmoid')
    axes[0, 0].set_title('Sigmoid Function', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Input (x)', fontsize=11)
    axes[0, 0].set_ylabel('Output', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(-0.1, 1.1)
    
    # ReLU
    axes[0, 1].plot(x, y_relu, 'g-', linewidth=2.5, label='ReLU')
    axes[0, 1].set_title('ReLU Function', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Input (x)', fontsize=11)
    axes[0, 1].set_ylabel('Output', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(-1, 10)
    
    # Tanh
    axes[1, 0].plot(x, y_tanh, 'm-', linewidth=2.5, label='Tanh')
    axes[1, 0].set_title('Tanh Function', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Input (x)', fontsize=11)
    axes[1, 0].set_ylabel('Output', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(-1.2, 1.2)
    
    # Leaky ReLU
    axes[1, 1].plot(x, y_leaky, 'orange', linewidth=2.5, label='Leaky ReLU (α=0.01)')
    axes[1, 1].set_title('Leaky ReLU Function', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Input (x)', fontsize=11)
    axes[1, 1].set_ylabel('Output', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(-0.5, 10)
    
    plt.suptitle('Activation Functions Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('task1_all_activations.png', dpi=150, bbox_inches='tight')
    print("✓ Combined comparison plot saved as 'task1_all_activations.png'")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("TASK 1: Activation Functions Implementation and Visualization")
    print("=" * 75)
    
    # Test each function with sample values
    print("\n" + "=" * 75)
    print("TESTING ACTIVATION FUNCTIONS WITH SAMPLE VALUES")
    print("=" * 75)
    
    test_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    print(f"\nTest Input Values: {test_values}")
    print(f"\n{'Input':<10} {'Sigmoid':<12} {'ReLU':<12} {'Tanh':<12} {'Leaky ReLU':<15}")
    print("-" * 75)
    
    for val in test_values:
        sig_val = sigmoid(val)
        relu_val = relu(val)
        tanh_val = tanh(val)
        leaky_val = leaky_relu(val)
        print(f"{val:>8.2f}   {sig_val:>10.4f}   {relu_val:>10.4f}   {tanh_val:>10.4f}   {leaky_val:>13.4f}")
    
    # Generate all plots
    print("\n" + "=" * 75)
    print("GENERATING VISUALIZATIONS")
    print("=" * 75)
    print()
    
    plot_sigmoid()
    plot_relu()
    plot_tanh()
    plot_leaky_relu()
    plot_all_activations()
    
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print("\n✓ All activation functions implemented successfully")
    print("✓ Individual plots generated for each function")
    print("✓ Comparison plot generated showing all functions together")
    print("\nKey Characteristics:")
    print("  • Sigmoid: Range [0,1], S-curve, used in binary classification")
    print("  • ReLU: Range [0,∞), piecewise linear, used in hidden layers")
    print("  • Tanh: Range [-1,1], zero-centered, better gradients than sigmoid")
    print("  • Leaky ReLU: Small slope for negatives, solves dying ReLU problem")
    print("\n" + "=" * 75)
