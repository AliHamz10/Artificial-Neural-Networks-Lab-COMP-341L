# Lab 3 - Task 2: Conceptual Questions on Activation Functions

**Author:** Ali Hamza  
**Roll Number:** B23F0063AI106  
**Section:** B.S AI - Red

---

## Question 1: Which activation function is most suitable for hidden layers and why?

**Answer:**

**ReLU (Rectified Linear Unit)** is the most suitable activation function for hidden layers.

**Reasons:**

1. **Computational efficiency**: ReLU is simple to compute — it only requires a max(0, x) operation. No exponentials like sigmoid or tanh, making forward and backward passes faster.

2. **Mitigates vanishing gradient**: ReLU has a derivative of 1 for positive inputs, so gradients flow through without shrinking. Sigmoid and tanh saturate for large |x|, causing gradients to vanish in deep networks.

3. **Sparse activation**: ReLU outputs zero for negative inputs, creating sparse representations. Fewer active neurons reduce overfitting and improve generalization.

4. **Avoids saturation for positive inputs**: Unlike sigmoid and tanh, ReLU does not saturate for positive values, allowing the network to learn faster.

5. **Empirical success**: ReLU is widely used in modern deep networks (CNNs, ResNets, etc.) and has shown strong performance in practice.

**Alternatives:** Leaky ReLU and Tanh are also used in hidden layers when ReLU causes "dying ReLU" or when zero-centered outputs are needed.

---

## Question 2: What is the vanishing gradient problem and which activation functions suffer from it?

**Answer:**

**Vanishing Gradient Problem:**

The vanishing gradient problem occurs in deep neural networks when gradients become very small (close to zero) as they propagate backward through layers. When gradients vanish:
- Earlier layers receive almost no gradient signal
- Weights in early layers barely update
- The network fails to learn useful representations from input data
- Training becomes extremely slow or stops

**Cause:** During backpropagation, gradients are multiplied by the derivative of the activation function at each layer. If these derivatives are small (< 1), the product shrinks exponentially with depth.

**Activation functions that suffer from it:**

1. **Sigmoid**: Derivative is f'(x) = f(x)(1 - f(x)), which is at most 0.25. For |x| large, the derivative approaches 0. Gradients vanish quickly in deep networks.

2. **Tanh**: Derivative is f'(x) = 1 - f(x)², which is at most 1 but approaches 0 for large |x|. Tanh saturates in both tails, causing vanishing gradients in deep networks.

**Activation functions that mitigate it:**

- **ReLU**: Derivative is 1 for x > 0, so no vanishing for positive inputs.
- **Leaky ReLU**: Small non-zero derivative for negative inputs, maintains gradient flow.

---

## Question 3: What issue does ReLU face and how does Leaky ReLU solve it?

**Answer:**

**Issue with ReLU: Dying ReLU Problem**

ReLU outputs zero for all negative inputs and has zero gradient for x ≤ 0. If a neuron's weighted sum is consistently negative during training:
- The neuron always outputs 0
- The gradient is always 0
- The weights never update
- The neuron becomes permanently "dead" and stops contributing to learning

This is called the **dying ReLU** or **dead ReLU** problem. It can affect a significant fraction of neurons, especially with high learning rates or improper initialization.

**How Leaky ReLU solves it:**

Leaky ReLU is defined as:
```
f(x) = x      if x > 0
f(x) = α·x    if x ≤ 0   (typically α = 0.01)
```

**Solution:**
- For negative inputs, Leaky ReLU outputs a small positive value (α·x) instead of zero
- The derivative for x < 0 is α (e.g., 0.01), not zero
- Gradients can still flow backward through "dead" neurons
- Weights can update and neurons can recover
- Prevents permanent neuron death while keeping most benefits of ReLU

---

## Question 4: Why is Softmax preferred in output layers for classification?

**Answer:**

**Softmax** is preferred in the output layer for **multi-class classification** for these reasons:

1. **Probability distribution**: Softmax converts raw scores (logits) into probabilities that sum to 1. Each output represents the probability of that class. This matches the interpretation needed for classification.

2. **Multi-class support**: Softmax handles multiple classes (e.g., 10 classes in MNIST) by producing one probability per class. Sigmoid is for binary classification; softmax generalizes to many classes.

3. **Competitive normalization**: Softmax uses relative scores. Increasing one class score automatically decreases others (zero-sum). This reflects the mutually exclusive nature of classes.

4. **Compatible with cross-entropy loss**: Softmax pairs naturally with cross-entropy loss. The gradient form is simple (predicted - actual), which makes training stable and efficient.

5. **Interpretable outputs**: Outputs like [0.7, 0.2, 0.1] directly show confidence per class, which is useful for decision-making and calibration.

**Formula:** softmax(z_i) = exp(z_i) / Σ exp(z_j)

**Note:** For binary classification, sigmoid is often used. For multi-class, softmax is the standard choice.
