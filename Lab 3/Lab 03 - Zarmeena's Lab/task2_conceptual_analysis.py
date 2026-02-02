"""
Lab 3 - Task 2: Conceptual Analysis of Activation Functions
Author: Zarmeena Jawad
Roll Number: B23F0115AI125
Section: AI Red

This script provides detailed answers to conceptual questions about activation functions:
1. Which activation function is most suitable for hidden layers and why?
2. What is the vanishing gradient problem and which activation functions suffer from it?
3. What issue does ReLU face and how does Leaky ReLU solve it?
4. Why is Softmax preferred in output layers for classification?
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# ACTIVATION FUNCTIONS (for demonstration)
# ============================================================================

def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)


def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function"""
    return np.where(x >= 0, x, alpha * x)


def softmax(x):
    """Softmax activation function for multi-class classification"""
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


# ============================================================================
# QUESTION 1: Which activation function is most suitable for hidden layers?
# ============================================================================

def answer_question_1():
    """
    Question 1: Which activation function is most suitable for hidden layers and why?
    
    Answer: ReLU (Rectified Linear Unit) is most suitable for hidden layers.
    """
    print("=" * 75)
    print("QUESTION 1: Most Suitable Activation Function for Hidden Layers")
    print("=" * 75)
    
    print("\nANSWER: ReLU (Rectified Linear Unit) is most suitable for hidden layers.")
    
    print("\n" + "-" * 75)
    print("REASONS:")
    print("-" * 75)
    
    print("\n1. COMPUTATIONAL EFFICIENCY:")
    print("   • ReLU is simple: max(0, x) - just a threshold operation")
    print("   • No expensive exponential calculations (unlike sigmoid/tanh)")
    print("   • Faster forward and backward propagation")
    print("   • Enables training of deeper networks efficiently")
    
    print("\n2. GRADIENT PROPERTIES:")
    print("   • Derivative is either 0 or 1 (very simple)")
    print("   • No vanishing gradient for positive inputs")
    print("   • Allows gradients to flow through active neurons")
    print("   • Enables effective backpropagation in deep networks")
    
    print("\n3. SPARSITY:")
    print("   • ReLU naturally creates sparse representations")
    print("   • Negative inputs become zero (neurons 'turn off')")
    print("   • Only relevant features activate, reducing overfitting")
    print("   • More efficient use of network capacity")
    
    print("\n4. BIOLOGICAL PLAUSIBILITY:")
    print("   • Mimics biological neurons (fire or don't fire)")
    print("   • Threshold-based activation is natural")
    
    print("\n" + "-" * 75)
    print("COMPARISON WITH OTHER FUNCTIONS:")
    print("-" * 75)
    
    print("\n• SIGMOID:")
    print("  - Problem: Vanishing gradients, not zero-centered")
    print("  - Use: Output layer for binary classification only")
    
    print("\n• TANH:")
    print("  - Better than sigmoid (zero-centered, better gradients)")
    print("  - Still suffers from vanishing gradients")
    print("  - More computationally expensive than ReLU")
    
    print("\n• LEAKY ReLU:")
    print("  - Good alternative to ReLU")
    print("  - Solves dying ReLU problem")
    print("  - Slightly more complex than ReLU")
    
    print("\n" + "=" * 75)


# ============================================================================
# QUESTION 2: Vanishing Gradient Problem
# ============================================================================

def answer_question_2():
    """
    Question 2: What is the vanishing gradient problem and which activation 
    functions suffer from it?
    """
    print("\n" + "=" * 75)
    print("QUESTION 2: Vanishing Gradient Problem")
    print("=" * 75)
    
    print("\nWHAT IS VANISHING GRADIENT PROBLEM?")
    print("-" * 75)
    print("""
    The vanishing gradient problem occurs when gradients become extremely small
    (close to zero) during backpropagation in deep neural networks. This happens
    when gradients are multiplied through many layers, causing them to shrink
    exponentially.
    
    CONSEQUENCES:
    • Early layers receive very small gradient updates
    • Weights in early layers barely change during training
    • Network fails to learn meaningful features in early layers
    • Training becomes very slow or stops completely
    • Deep networks become difficult or impossible to train
    """)
    
    print("\nWHICH ACTIVATION FUNCTIONS SUFFER FROM IT?")
    print("-" * 75)
    
    print("\n1. SIGMOID FUNCTION:")
    print("   • Derivative: f'(x) = f(x) * (1 - f(x))")
    print("   • Maximum derivative value: 0.25 (at x=0)")
    print("   • Derivative approaches 0 for large |x|")
    print("   • When multiplied across layers, gradients vanish quickly")
    print("   • SEVERELY AFFECTED by vanishing gradients")
    
    # Demonstrate sigmoid derivative
    x = np.linspace(-10, 10, 1000)
    sigmoid_vals = sigmoid(x)
    sigmoid_derivative = sigmoid_vals * (1 - sigmoid_vals)
    
    print("\n   Visual Demonstration:")
    print(f"   • At x=0: derivative = {sigmoid_derivative[500]:.4f}")
    print(f"   • At x=5: derivative = {sigmoid_derivative[750]:.6f} (very small!)")
    print(f"   • At x=-5: derivative = {sigmoid_derivative[250]:.6f} (very small!)")
    
    print("\n2. TANH FUNCTION:")
    print("   • Derivative: f'(x) = 1 - (f(x))^2")
    print("   • Maximum derivative value: 1.0 (at x=0)")
    print("   • Better than sigmoid but still suffers from vanishing gradients")
    print("   • Derivative approaches 0 for large |x|")
    print("   • MODERATELY AFFECTED by vanishing gradients")
    
    # Demonstrate tanh derivative
    tanh_vals = tanh(x)
    tanh_derivative = 1 - tanh_vals**2
    
    print("\n   Visual Demonstration:")
    print(f"   • At x=0: derivative = {tanh_derivative[500]:.4f}")
    print(f"   • At x=5: derivative = {tanh_derivative[750]:.6f} (small)")
    print(f"   • At x=-5: derivative = {tanh_derivative[250]:.6f} (small)")
    
    print("\n3. ReLU FUNCTION:")
    print("   • Derivative: f'(x) = 1 if x > 0, else 0")
    print("   • No vanishing gradient for positive inputs")
    print("   • Constant gradient of 1 for active neurons")
    print("   • NOT AFFECTED by vanishing gradients (for positive inputs)")
    print("   • Problem: Dead neurons (gradient = 0 for negative inputs)")
    
    print("\n4. LEAKY ReLU:")
    print("   • Derivative: f'(x) = 1 if x > 0, else alpha")
    print("   • Small but non-zero gradient for negative inputs")
    print("   • NOT AFFECTED by vanishing gradients")
    print("   • Solves both vanishing gradient and dying ReLU problems")
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, sigmoid_derivative, 'b-', linewidth=2.5, label='Sigmoid Derivative')
    plt.axhline(y=0.25, color='r', linestyle='--', alpha=0.7, label='Max = 0.25')
    plt.xlabel('Input (x)', fontsize=11)
    plt.ylabel("f'(x)", fontsize=11)
    plt.title('Sigmoid Derivative (Vanishing Gradient)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.05, 0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(x, tanh_derivative, 'm-', linewidth=2.5, label='Tanh Derivative')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Max = 1.0')
    plt.xlabel('Input (x)', fontsize=11)
    plt.ylabel("f'(x)", fontsize=11)
    plt.title('Tanh Derivative (Moderate Vanishing)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.1, 1.1)
    
    plt.subplot(2, 2, 3)
    relu_derivative = np.where(x > 0, 1, 0)
    plt.plot(x, relu_derivative, 'g-', linewidth=2.5, label='ReLU Derivative')
    plt.xlabel('Input (x)', fontsize=11)
    plt.ylabel("f'(x)", fontsize=11)
    plt.title('ReLU Derivative (No Vanishing for x>0)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.1, 1.2)
    
    plt.subplot(2, 2, 4)
    leaky_derivative = np.where(x > 0, 1, 0.01)
    plt.plot(x, leaky_derivative, 'orange', linewidth=2.5, label='Leaky ReLU Derivative')
    plt.xlabel('Input (x)', fontsize=11)
    plt.ylabel("f'(x)", fontsize=11)
    plt.title('Leaky ReLU Derivative (No Vanishing)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-0.05, 1.2)
    
    plt.suptitle('Gradient Comparison: Vanishing Gradient Problem', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('task2_vanishing_gradient.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'task2_vanishing_gradient.png'")
    plt.close()
    
    print("\n" + "=" * 75)


# ============================================================================
# QUESTION 3: ReLU Issue and Leaky ReLU Solution
# ============================================================================

def answer_question_3():
    """
    Question 3: What issue does ReLU face and how does Leaky ReLU solve it?
    """
    print("\n" + "=" * 75)
    print("QUESTION 3: ReLU Issue and Leaky ReLU Solution")
    print("=" * 75)
    
    print("\nTHE PROBLEM: Dying ReLU Problem")
    print("-" * 75)
    print("""
    ReLU faces the "Dying ReLU" problem (also called "Dead Neuron" problem):
    
    WHAT HAPPENS:
    • When a neuron's input is consistently negative, ReLU outputs zero
    • The gradient for negative inputs is exactly zero
    • Once a neuron becomes inactive (outputs zero), it may never recover
    • The neuron "dies" and stops contributing to learning
    • This can happen to a significant portion of neurons in a network
    
    WHY IT HAPPENS:
    • Large negative bias values can push neurons into negative region
    • During training, if weights become too negative, neuron stays off
    • Learning rate too high can cause weights to jump into negative region
    • Once gradient is zero, no weight updates occur, neuron stays dead
    """)
    
    # Demonstrate dying ReLU
    x = np.linspace(-5, 5, 1000)
    relu_output = relu(x)
    leaky_output = leaky_relu(x, alpha=0.01)
    
    print("\nHOW LEAKY ReLU SOLVES IT:")
    print("-" * 75)
    print("""
    Leaky ReLU introduces a small positive slope (alpha) for negative inputs:
    
    FORMULA:
    • ReLU:      f(x) = max(0, x)           → gradient = 0 for x < 0
    • Leaky ReLU: f(x) = max(alpha * x, x)   → gradient = alpha for x < 0
    
    KEY DIFFERENCES:
    1. Small Gradient for Negatives:
       • ReLU: gradient = 0 for negative inputs (neuron dies)
       • Leaky ReLU: gradient = alpha (typically 0.01) for negative inputs
       • Neuron can still learn and recover even with negative inputs
    
    2. Prevents Permanent Death:
       • Even if input is negative, small gradient allows weight updates
       • Neuron can gradually move back to positive region
       • Network maintains more active neurons
    
    3. Better Gradient Flow:
       • Gradients can flow through all neurons (not just positive ones)
       • More neurons contribute to learning
       • Better utilization of network capacity
    """)
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ReLU
    axes[0].plot(x, relu_output, 'g-', linewidth=3, label='ReLU', alpha=0.9)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].fill_between(x[x < 0], relu_output[x < 0], 0, alpha=0.3, color='red', 
                         label='Dead Zone (gradient = 0)')
    axes[0].set_xlabel('Input (x)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Output', fontsize=12, fontweight='bold')
    axes[0].set_title('ReLU: Dying Neuron Problem', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(-0.5, 5)
    axes[0].text(2, 4, 'Problem:\nGradient = 0\nfor x < 0', 
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontsize=10)
    
    # Leaky ReLU
    axes[1].plot(x, leaky_output, 'orange', linewidth=3, label='Leaky ReLU (α=0.01)', alpha=0.9)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].fill_between(x[x < 0], leaky_output[x < 0], 0, alpha=0.3, color='green', 
                         label='Active Zone (gradient = α)')
    axes[1].set_xlabel('Input (x)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Output', fontsize=12, fontweight='bold')
    axes[1].set_title('Leaky ReLU: Solution to Dying Neurons', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(-0.5, 5)
    axes[1].text(2, 4, 'Solution:\nGradient = α\nfor x < 0', 
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7), fontsize=10)
    
    plt.suptitle('ReLU vs Leaky ReLU: Dying Neuron Problem', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('task2_dying_relu_solution.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'task2_dying_relu_solution.png'")
    plt.close()
    
    print("\nCOMPARISON TABLE:")
    print("-" * 75)
    print(f"{'Aspect':<25} {'ReLU':<30} {'Leaky ReLU':<30}")
    print("-" * 85)
    print(f"{'Gradient for x < 0':<25} {'0 (dead)':<30} {'α (small but active)':<30}")
    print(f"{'Neuron Recovery':<25} {'No (permanently dead)':<30} {'Yes (can recover)':<30}")
    print(f"{'Active Neurons':<25} {'Fewer (many die)':<30} {'More (stay active)':<30}")
    print(f"{'Learning Capacity':<25} {'Reduced':<30} {'Maintained':<30}")
    print(f"{'Computational Cost':<25} {'Very Low':<30} {'Very Low':<30}")
    
    print("\n" + "=" * 75)


# ============================================================================
# QUESTION 4: Why Softmax for Output Layers?
# ============================================================================

def answer_question_4():
    """
    Question 4: Why is Softmax preferred in output layers for classification?
    """
    print("\n" + "=" * 75)
    print("QUESTION 4: Why Softmax in Output Layers for Classification?")
    print("=" * 75)
    
    print("\nWHY SOFTMAX?")
    print("-" * 75)
    print("""
    Softmax is preferred in output layers for multi-class classification because:
    
    1. PROBABILITY DISTRIBUTION:
       • Converts raw scores (logits) into valid probabilities
       • All outputs sum to exactly 1.0
       • Each output is in range [0, 1]
       • Provides interpretable confidence scores for each class
    
    2. MULTI-CLASS CLASSIFICATION:
       • Designed specifically for problems with multiple classes
       • Handles competition between classes naturally
       • Winner-takes-all behavior with probability distribution
       • Better than using sigmoid for each class independently
    
    3. DIFFERENTIABLE AND SMOOTH:
       • Smooth function (important for gradient-based optimization)
       • Well-behaved gradients for backpropagation
       • Enables effective training with gradient descent
    
    4. NUMERICAL STABILITY:
       • Can be implemented with max subtraction trick
       • Prevents overflow in exponential calculations
       • Robust to large input values
    """)
    
    # Demonstrate softmax
    print("\nSOFTMAX EXAMPLE:")
    print("-" * 75)
    
    # Example: 3-class classification
    logits = np.array([2.3, 1.6, 0.7])  # Raw scores for 3 classes
    probabilities = softmax(logits)
    
    classes = ['Class A', 'Class B', 'Class C']
    
    print(f"\nRaw Logits (Network Outputs):")
    for i, (cls, logit) in enumerate(zip(classes, logits)):
        print(f"  {cls}: {logit:.2f}")
    
    print(f"\nSoftmax Probabilities:")
    for cls, prob in zip(classes, probabilities):
        print(f"  {cls}: {prob:.4f} ({prob*100:.2f}%)")
    
    print(f"\nVerification:")
    print(f"  Sum of probabilities: {np.sum(probabilities):.6f} (should be 1.0)")
    print(f"  All in [0, 1] range: {np.all((probabilities >= 0) & (probabilities <= 1))}")
    
    # Visualize softmax
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart comparison
    x_pos = np.arange(len(classes))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, logits, width, label='Raw Logits', alpha=0.8, color='lightblue')
    axes[0].bar(x_pos + width/2, probabilities, width, label='Softmax Probabilities', 
                alpha=0.8, color='coral')
    axes[0].set_xlabel('Classes', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Value', fontsize=12, fontweight='bold')
    axes[0].set_title('Logits vs Softmax Probabilities', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(classes)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Probability distribution
    axes[1].bar(classes, probabilities, color=['skyblue', 'lightgreen', 'salmon'], alpha=0.8)
    axes[1].set_ylabel('Probability', fontsize=12, fontweight='bold')
    axes[1].set_title('Softmax Probability Distribution', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add probability labels on bars
    for i, prob in enumerate(probabilities):
        axes[1].text(i, prob + 0.02, f'{prob:.2%}', ha='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('Softmax: Converting Logits to Probabilities', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('task2_softmax_output.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'task2_softmax_output.png'")
    plt.close()
    
    print("\nCOMPARISON WITH OTHER FUNCTIONS:")
    print("-" * 75)
    print("""
    • SIGMOID:
      - Used for binary classification (single output)
      - Outputs probability for one class
      - Not suitable for multi-class (doesn't ensure sum = 1)
    
    • SOFTMAX:
      - Used for multi-class classification (multiple outputs)
      - Ensures probabilities sum to 1.0
      - Natural competition between classes
      - Standard choice for multi-class output layers
    
    • ReLU/TANH:
      - Not suitable for output layers in classification
      - Don't produce probability distributions
      - Used only in hidden layers
    """)
    
    print("\nKEY ADVANTAGES:")
    print("-" * 75)
    print("  1. Interpretable: Probabilities tell us confidence in each class")
    print("  2. Normalized: Sum to 1.0 (valid probability distribution)")
    print("  3. Competitive: Increasing one class decreases others")
    print("  4. Differentiable: Enables gradient-based learning")
    print("  5. Standard: Industry standard for multi-class classification")
    
    print("\n" + "=" * 75)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("TASK 2: Conceptual Analysis of Activation Functions")
    print("=" * 75)
    print("\nThis script provides detailed answers to conceptual questions")
    print("about activation functions in neural networks.\n")
    
    # Answer all questions
    answer_question_1()
    answer_question_2()
    answer_question_3()
    answer_question_4()
    
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print("\n✓ All conceptual questions answered in detail")
    print("✓ Visualizations generated for better understanding")
    print("\nKey Takeaways:")
    print("  • ReLU is best for hidden layers (efficiency, no vanishing gradients)")
    print("  • Sigmoid/Tanh suffer from vanishing gradient problem")
    print("  • Leaky ReLU solves the dying ReLU problem")
    print("  • Softmax is essential for multi-class classification output layers")
    print("\n" + "=" * 75)
