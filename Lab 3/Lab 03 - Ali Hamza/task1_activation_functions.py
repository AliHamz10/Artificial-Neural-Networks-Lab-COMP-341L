"""
Lab 3 - Task 1: Activation Functions Implementation and Visualization
Author: Ali Hamza
Roll Number: B23F0063AI106
Section: B.S AI - Red

This script implements and visualizes four common activation functions:
1. Sigmoid
2. ReLU (Rectified Linear Unit)
3. Tanh (Hyperbolic Tangent)
4. Leaky ReLU
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


# =============================================================================
# ACTIVATION FUNCTION IMPLEMENTATIONS
# =============================================================================

def sigmoid(x):
    """
    Sigmoid (Logistic) activation function.
    Transforms input to range [0, 1].
    Formula: f(x) = 1 / (1 + e^(-x))
    Use: Binary classification. Suffers from gradient vanishing.
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function.
    Outputs x if x > 0, else 0.
    Formula: f(x) = max(0, x)
    Use: Hidden layers for non-linearity with efficient computation.
    """
    return np.maximum(0, x)


def tanh(x):
    """
    Tanh (Hyperbolic Tangent) activation function.
    Transforms input to range [-1, 1]. Zero-centered.
    Formula: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Use: Better gradient properties than sigmoid.
    """
    return np.tanh(x)


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    For x > 0: outputs x. For x <= 0: outputs alpha * x.
    Formula: f(x) = x if x > 0 else alpha * x
    Use: Addresses 'dying ReLU' problem by allowing small gradient for negative inputs.
    """
    return np.where(x >= 0, x, alpha * x)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_sigmoid():
    """Plot Sigmoid activation function."""
    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Sigmoid(x)', fontsize=12)
    plt.title('Sigmoid Activation Function', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig('task1_sigmoid.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_relu():
    """Plot ReLU activation function."""
    x = np.linspace(-10, 10, 100)
    y = relu(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'g-', linewidth=2)
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('ReLU(x)', fontsize=12)
    plt.title('ReLU (Rectified Linear Unit) Activation Function', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('task1_relu.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_tanh():
    """Plot Tanh activation function."""
    x = np.linspace(-10, 10, 100)
    y = tanh(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'm-', linewidth=2)
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('tanh(x)', fontsize=12)
    plt.title('Tanh (Hyperbolic Tangent) Activation Function', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.ylim(-1.2, 1.2)
    plt.tight_layout()
    plt.savefig('task1_tanh.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_leaky_relu(alpha=0.01):
    """Plot Leaky ReLU activation function."""
    x = np.linspace(-10, 10, 100)
    y = leaky_relu(x, alpha=alpha)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'r-', linewidth=2)
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Leaky ReLU(x)', fontsize=12)
    plt.title(f'Leaky ReLU Activation Function (alpha={alpha})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('task1_leaky_relu.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_activations():
    """Plot all four activation functions in a 2x2 grid for comparison."""
    x = np.linspace(-10, 10, 200)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sigmoid
    axes[0, 0].plot(x, sigmoid(x), 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Input (x)')
    axes[0, 0].set_ylabel('Output')
    axes[0, 0].set_title('Sigmoid', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
    
    # ReLU
    axes[0, 1].plot(x, relu(x), 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Input (x)')
    axes[0, 1].set_ylabel('Output')
    axes[0, 1].set_title('ReLU', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
    
    # Tanh
    axes[1, 0].plot(x, tanh(x), 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Input (x)')
    axes[1, 0].set_ylabel('Output')
    axes[1, 0].set_title('Tanh', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
    
    # Leaky ReLU
    axes[1, 1].plot(x, leaky_relu(x), 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Input (x)')
    axes[1, 1].set_ylabel('Output')
    axes[1, 1].set_title('Leaky ReLU', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
    
    plt.suptitle('Activation Functions Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('task1_all_activations.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN: Test and Visualize
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lab 3 - Task 1: Activation Functions")
    print("=" * 60)
    
    # Test each function with sample values
    print("\n--- Function Tests ---")
    print(f"Sigmoid(0.8)    = {sigmoid(0.8):.6f}")
    print(f"Tanh(1.0)       = {tanh(1.0):.6f}")
    print(f"ReLU(-78)       = {relu(-78)}")
    print(f"ReLU(78)        = {relu(78)}")
    print(f"Leaky ReLU(-89) = {leaky_relu(-89):.2f}")
    print(f"Leaky ReLU(100) = {leaky_relu(100)}")
    
    # Plot individual functions
    print("\n--- Generating Plots ---")
    plot_sigmoid()
    plot_relu()
    plot_tanh()
    plot_leaky_relu()
    
    # Plot all together
    plot_all_activations()
    
    print("\nDone! All plots saved.")
