# Lab Report 3: Activation Functions

---

**Course Code:** COMP-341L  
**Course Name:** Artificial Neural Networks Lab  
**Lab Number:** 3  
**Lab Title:** Activation Functions  
**Date:** February 2, 2026

**Name:** Ali Hamza  
**Roll Number:** B23F0063AI106  
**Section:** B.S AI - Red

---

## Objective

To implement and visualize common activation functions (Sigmoid, ReLU, Tanh, Leaky ReLU) in Python, and to understand their properties, use cases, and trade-offs through conceptual analysis.

---

## Introduction

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. They determine whether a neuron should fire based on the weighted sum of inputs and bias, and they make backpropagation possible by providing gradients for weight updates. This lab implements four activation functions and explores their characteristics through code and conceptual questions.

---

## Task 1: Implementation and Visualization of Activation Functions

### 1.1 Sigmoid Function

**Formula:** f(x) = 1 / (1 + e^(-x))

**Properties:**
- Output range: [0, 1]
- S-shaped curve
- Use: Binary classification
- Drawback: Suffers from vanishing gradient problem

**Implementation:**
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

### 1.2 ReLU (Rectified Linear Unit)

**Formula:** f(x) = max(0, x)

**Properties:**
- Output: x if x > 0, else 0
- Simple and computationally efficient
- Use: Hidden layers for non-linearity
- Drawback: Dying ReLU problem for negative inputs

**Implementation:**
```python
def relu(x):
    return np.maximum(0, x)
```

### 1.3 Tanh (Hyperbolic Tangent)

**Formula:** f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

**Properties:**
- Output range: [-1, 1]
- Zero-centered
- Better gradient properties than sigmoid
- Drawback: Still suffers from vanishing gradient for large |x|

**Implementation:**
```python
def tanh(x):
    return np.tanh(x)
```

### 1.4 Leaky ReLU

**Formula:** f(x) = x if x > 0, else α·x (typically α = 0.01)

**Properties:**
- Allows small gradient for negative inputs
- Addresses dying ReLU problem
- Maintains most benefits of ReLU

**Implementation:**
```python
def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)
```

### 1.5 Visualizations

Plots were generated for each activation function over the input range [-10, 10]:
- `task1_sigmoid.png` — Sigmoid activation
- `task1_relu.png` — ReLU activation
- `task1_tanh.png` — Tanh activation
- `task1_leaky_relu.png` — Leaky ReLU activation
- `task1_all_activations.png` — All four functions in a 2×2 comparison grid

---

## Task 2: Conceptual Questions

### Q1: Which activation function is most suitable for hidden layers and why?

**Answer:** ReLU is the most suitable for hidden layers because:
- It is computationally efficient (no exponentials)
- It mitigates the vanishing gradient problem (derivative = 1 for x > 0)
- It produces sparse activations, reducing overfitting
- It does not saturate for positive inputs
- It is widely used and performs well in practice

### Q2: What is the vanishing gradient problem and which activation functions suffer from it?

**Answer:** The vanishing gradient problem occurs when gradients become very small as they propagate backward through deep networks. Early layers receive almost no gradient signal, so weights barely update and learning stalls.

**Functions that suffer:** Sigmoid and Tanh. Their derivatives approach zero for large |x|, causing gradients to shrink exponentially with depth.

**Functions that mitigate:** ReLU and Leaky ReLU maintain non-zero gradients for positive (and for Leaky ReLU, negative) inputs.

### Q3: What issue does ReLU face and how does Leaky ReLU solve it?

**Answer:** ReLU faces the **dying ReLU** problem: neurons with consistently negative inputs output zero and have zero gradient, so their weights never update and they become permanently inactive.

**Leaky ReLU solution:** For x ≤ 0, Leaky ReLU outputs α·x (e.g., 0.01x) instead of 0. This gives a small non-zero derivative, so gradients can flow and "dead" neurons can recover.

### Q4: Why is Softmax preferred in output layers for classification?

**Answer:** Softmax is preferred because:
- It produces a valid probability distribution (outputs sum to 1)
- It supports multi-class classification
- It uses relative scores (competitive normalization)
- It pairs well with cross-entropy loss
- Outputs are interpretable as class probabilities

---

## Results and Observations

- All four activation functions were implemented and tested successfully
- Plots clearly show the different shapes: Sigmoid (S-curve 0–1), ReLU (flat then linear), Tanh (S-curve -1 to 1), Leaky ReLU (slight slope for negative inputs)
- The comparison plot highlights differences in range, saturation, and gradient behavior

---

## Conclusion

This lab reinforced how activation functions shape neural network behavior. Sigmoid and Tanh are useful for specific cases but suffer from vanishing gradients in deep networks. ReLU is the default choice for hidden layers due to its simplicity and gradient flow. Leaky ReLU addresses the dying ReLU problem by allowing small gradients for negative inputs. Softmax is the standard choice for multi-class classification outputs. Choosing the right activation function depends on the layer (hidden vs output), the task (classification vs regression), and the network depth.

---

## References

1. Lab Manual: Lab 03 - Activation Functions
2. NumPy Documentation: https://numpy.org/doc/
3. Matplotlib Documentation: https://matplotlib.org/

---

**Prepared by:** Ali Hamza  
**Roll Number:** B23F0063AI106  
**Section:** B.S AI - Red  
**Date:** February 2, 2026
