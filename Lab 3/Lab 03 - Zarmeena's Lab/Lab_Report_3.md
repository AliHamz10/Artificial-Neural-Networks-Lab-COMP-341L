# Laboratory Report 3: Activation Functions in Neural Networks

---

**Course:** COMP-341L - Artificial Neural Networks Lab  
**Lab Assignment:** 3  
**Topic:** Activation Functions  
**Submission Date:** January 26, 2026

**Student Name:** Zarmeena Jawad  
**Roll Number:** B23F0115AI125  
**Section:** AI Red

---

## Executive Summary

This laboratory assignment focuses on understanding and implementing various activation functions used in artificial neural networks. The experiments involve implementing four key activation functions (Sigmoid, ReLU, Tanh, and Leaky ReLU), visualizing their behavior, and analyzing their properties, advantages, and limitations. The lab also explores conceptual questions about activation function selection, gradient problems, and their applications in different network layers.

---

## Learning Objectives

Upon completion of this lab, students should be able to:
- Implement common activation functions (Sigmoid, ReLU, Tanh, Leaky ReLU)
- Visualize and compare activation function behaviors
- Understand the vanishing gradient problem and its impact
- Identify suitable activation functions for different network layers
- Explain the dying ReLU problem and its solutions
- Understand why Softmax is preferred for multi-class classification

---

## Methodology

All experiments were conducted using Python 3 with NumPy and Matplotlib libraries. The implementations follow a systematic approach: implementing each activation function with proper mathematical formulations, testing with sample values, and generating visualizations to illustrate key characteristics. Conceptual analysis includes detailed explanations supported by visual demonstrations.

---

## Task 1: Activation Functions Implementation and Visualization

### Purpose
Implement and visualize four fundamental activation functions used in neural networks, understanding their mathematical properties, output ranges, and use cases.

### Implementation Details

#### 1. Sigmoid (Logistic) Function

**Mathematical Formula:** `f(x) = 1 / (1 + e^(-x))`

**Derivative:** `f'(x) = f(x) * (1 - f(x))`

**Properties:**
- Output Range: [0, 1]
- S-shaped smooth curve
- Maximum derivative: 0.25 (at x = 0)
- Used in: Binary classification output layers
- Problem: Suffers from vanishing gradient problem

**Implementation:**
```python
def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))
```

#### 2. ReLU (Rectified Linear Unit)

**Mathematical Formula:** `f(x) = max(0, x)`

**Derivative:** `f'(x) = 1 if x > 0, else 0`

**Properties:**
- Output Range: [0, ∞)
- Piecewise linear function
- Computationally efficient
- Used in: Hidden layers
- Problem: Dying ReLU problem (neurons can become permanently inactive)

**Implementation:**
```python
def relu(x):
    return np.maximum(0, x)
```

#### 3. Tanh (Hyperbolic Tangent)

**Mathematical Formula:** `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

**Derivative:** `f'(x) = 1 - (f(x))^2`

**Properties:**
- Output Range: [-1, 1]
- Zero-centered (symmetric around origin)
- Maximum derivative: 1.0 (at x = 0)
- Used in: Hidden layers (better gradients than sigmoid)
- Problem: Still suffers from vanishing gradients

**Implementation:**
```python
def tanh(x):
    return np.tanh(x)
```

#### 4. Leaky ReLU (Parametric ReLU)

**Mathematical Formula:** 
- `f(x) = x if x > 0`
- `f(x) = alpha * x if x <= 0` (typically alpha = 0.01)

**Derivative:** `f'(x) = 1 if x > 0, else alpha`

**Properties:**
- Output Range: (-∞, ∞) for negative, [0, ∞) for positive
- Small positive slope for negative inputs
- Used in: Hidden layers
- Advantage: Solves dying ReLU problem

**Implementation:**
```python
def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)
```

### Results

**Test Values and Outputs:**

| Input | Sigmoid | ReLU | Tanh | Leaky ReLU |
|-------|---------|------|------|------------|
| -2.0  | 0.1192  | 0.0000 | -0.9640 | -0.0200 |
| -1.0  | 0.2689  | 0.0000 | -0.7616 | -0.0100 |
| 0.0   | 0.5000  | 0.0000 | 0.0000  | 0.0000   |
| 1.0   | 0.7311  | 1.0000 | 0.7616  | 1.0000   |
| 2.0   | 0.8808  | 2.0000 | 0.9640  | 2.0000   |

### Visualizations

The following visualizations were generated:
- Individual plots for each activation function showing their characteristic shapes
- Combined comparison plot displaying all functions together
- Annotations highlighting key properties and ranges

**Key Observations:**
- Sigmoid: Smooth S-curve, maps all inputs to [0, 1] range
- ReLU: Linear for positive inputs, zero for negative (hard cutoff)
- Tanh: S-curve symmetric around zero, maps to [-1, 1] range
- Leaky ReLU: Similar to ReLU but with small slope for negative inputs

---

## Task 2: Conceptual Analysis of Activation Functions

### Purpose
Answer conceptual questions about activation functions, their properties, problems, and appropriate usage in neural network architectures.

### Question 1: Most Suitable Activation Function for Hidden Layers

**Answer:** ReLU (Rectified Linear Unit) is most suitable for hidden layers.

**Reasons:**

1. **Computational Efficiency**
   - Simple operation: `max(0, x)` - just a threshold
   - No expensive exponential calculations
   - Faster forward and backward propagation
   - Enables training of deeper networks efficiently

2. **Gradient Properties**
   - Derivative is either 0 or 1 (very simple)
   - No vanishing gradient for positive inputs
   - Allows gradients to flow through active neurons
   - Enables effective backpropagation in deep networks

3. **Sparsity**
   - Naturally creates sparse representations
   - Negative inputs become zero (neurons "turn off")
   - Only relevant features activate, reducing overfitting

4. **Biological Plausibility**
   - Mimics biological neurons (fire or don't fire)
   - Threshold-based activation is natural

**Comparison:**
- **Sigmoid**: Vanishing gradients, not zero-centered → Use only in output layer for binary classification
- **Tanh**: Better than sigmoid but still suffers from vanishing gradients → Less efficient than ReLU
- **Leaky ReLU**: Good alternative, solves dying ReLU problem → Slightly more complex than ReLU

### Question 2: Vanishing Gradient Problem

**What is the Vanishing Gradient Problem?**

The vanishing gradient problem occurs when gradients become extremely small (close to zero) during backpropagation in deep neural networks. When gradients are multiplied through many layers, they shrink exponentially.

**Consequences:**
- Early layers receive very small gradient updates
- Weights in early layers barely change during training
- Network fails to learn meaningful features in early layers
- Training becomes very slow or stops completely
- Deep networks become difficult or impossible to train

**Which Activation Functions Suffer From It?**

1. **Sigmoid** - SEVERELY AFFECTED
   - Maximum derivative: 0.25 (at x = 0)
   - Derivative approaches 0 for large |x|
   - When multiplied across layers, gradients vanish quickly
   - Example: At x = 5, derivative ≈ 0.0066 (very small!)

2. **Tanh** - MODERATELY AFFECTED
   - Maximum derivative: 1.0 (at x = 0)
   - Better than sigmoid but still suffers from vanishing gradients
   - Derivative approaches 0 for large |x|
   - Example: At x = 5, derivative ≈ 0.0002 (small)

3. **ReLU** - NOT AFFECTED (for positive inputs)
   - Constant gradient of 1 for active neurons
   - No vanishing gradient for positive inputs
   - Problem: Dead neurons (gradient = 0 for negative inputs)

4. **Leaky ReLU** - NOT AFFECTED
   - Small but non-zero gradient for negative inputs
   - Solves both vanishing gradient and dying ReLU problems

### Question 3: ReLU Issue and Leaky ReLU Solution

**The Problem: Dying ReLU**

ReLU faces the "Dying ReLU" problem (also called "Dead Neuron" problem):

**What Happens:**
- When a neuron's input is consistently negative, ReLU outputs zero
- The gradient for negative inputs is exactly zero
- Once a neuron becomes inactive (outputs zero), it may never recover
- The neuron "dies" and stops contributing to learning
- This can happen to a significant portion of neurons in a network

**Why It Happens:**
- Large negative bias values can push neurons into negative region
- During training, if weights become too negative, neuron stays off
- Learning rate too high can cause weights to jump into negative region
- Once gradient is zero, no weight updates occur, neuron stays dead

**How Leaky ReLU Solves It:**

Leaky ReLU introduces a small positive slope (alpha) for negative inputs:

**Key Differences:**

| Aspect | ReLU | Leaky ReLU |
|--------|------|------------|
| Gradient for x < 0 | 0 (dead) | α (small but active) |
| Neuron Recovery | No (permanently dead) | Yes (can recover) |
| Active Neurons | Fewer (many die) | More (stay active) |
| Learning Capacity | Reduced | Maintained |

**Solution Mechanism:**
- Even if input is negative, small gradient (alpha) allows weight updates
- Neuron can gradually move back to positive region
- Network maintains more active neurons
- Better gradient flow through all neurons

### Question 4: Why Softmax in Output Layers?

**Why Softmax is Preferred:**

1. **Probability Distribution**
   - Converts raw scores (logits) into valid probabilities
   - All outputs sum to exactly 1.0
   - Each output is in range [0, 1]
   - Provides interpretable confidence scores for each class

2. **Multi-Class Classification**
   - Designed specifically for problems with multiple classes
   - Handles competition between classes naturally
   - Winner-takes-all behavior with probability distribution
   - Better than using sigmoid for each class independently

3. **Differentiable and Smooth**
   - Smooth function (important for gradient-based optimization)
   - Well-behaved gradients for backpropagation
   - Enables effective training with gradient descent

4. **Numerical Stability**
   - Can be implemented with max subtraction trick
   - Prevents overflow in exponential calculations
   - Robust to large input values

**Example:**

Raw Logits: [2.3, 1.6, 0.7]  
Softmax Probabilities: [0.5888, 0.2924, 0.1189] (58.88%, 29.24%, 11.89%)

**Comparison:**
- **Sigmoid**: Used for binary classification (single output), not suitable for multi-class
- **Softmax**: Used for multi-class classification (multiple outputs), ensures probabilities sum to 1.0
- **ReLU/Tanh**: Not suitable for output layers in classification (don't produce probability distributions)

---

## Results Summary

### Task 1 Results
- Successfully implemented four activation functions (Sigmoid, ReLU, Tanh, Leaky ReLU)
- Generated individual visualizations for each function
- Created combined comparison plot showing all functions together
- Tested functions with sample values demonstrating different behaviors

### Task 2 Results
- Identified ReLU as most suitable for hidden layers with detailed justification
- Explained vanishing gradient problem and identified affected functions
- Analyzed dying ReLU problem and Leaky ReLU solution
- Demonstrated why Softmax is preferred for multi-class classification output layers
- Generated visualizations supporting conceptual explanations

---

## Conclusion

This laboratory assignment provided comprehensive understanding of activation functions in neural networks. Through implementing and visualizing different activation functions, I learned that each function has distinct characteristics that make it suitable for specific applications. Sigmoid functions provide smooth probability mappings but suffer from vanishing gradients, making them suitable only for output layers in binary classification. ReLU functions offer computational efficiency and solve vanishing gradient problems for positive inputs, making them ideal for hidden layers, though they face the dying neuron problem. Leaky ReLU addresses this limitation by allowing small gradients for negative inputs, maintaining network capacity. Tanh functions provide zero-centered outputs with better gradient properties than sigmoid but are still computationally more expensive than ReLU.

The conceptual analysis revealed critical insights into activation function selection. ReLU's dominance in hidden layers stems from its simplicity, efficiency, and ability to maintain gradient flow in deep networks. The vanishing gradient problem, which severely affects sigmoid and moderately affects tanh, demonstrates why these functions are less suitable for deep architectures. The dying ReLU problem illustrates how even effective functions can have limitations, and Leaky ReLU's solution shows the importance of maintaining gradient flow through all neurons.

Understanding Softmax's role in multi-class classification highlighted the importance of proper output layer design. Softmax's ability to convert raw logits into valid probability distributions provides interpretable confidence scores and enables effective multi-class learning. This understanding is crucial for designing neural network architectures that can handle complex classification tasks.

Overall, this lab demonstrated that activation function selection is not arbitrary but based on mathematical properties, computational efficiency, gradient behavior, and task requirements. The choice of activation function fundamentally affects network learning capability, training efficiency, and final performance. This knowledge is essential for designing effective neural network architectures.

---

## References

1. Lab Manual: Lab 03 - Activation Functions
2. NumPy Documentation: https://numpy.org/doc/stable/
3. Matplotlib Documentation: https://matplotlib.org/stable/contents.html

---

**Report Prepared By:** Zarmeena Jawad  
**Roll Number:** B23F0115AI125  
**Section:** AI Red  
**Date:** January 26, 2026
