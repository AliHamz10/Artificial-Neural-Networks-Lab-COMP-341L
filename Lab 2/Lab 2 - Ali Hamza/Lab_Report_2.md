# Lab Report 2: Understanding ANN from Scratch

---

**Course Code:** COMP-341L  
**Course Name:** Artificial Neural Networks Lab  
**Lab Number:** 2  
**Lab Title:** Understanding ANN from Scratch  
**Date:** January 26, 2026

**Name:** Ali Hamza  
**Roll Number:** B23F0063AI106  
**Section:** B.S AI - Red

---

## Objective

To understand the fundamental concepts of artificial neural networks by implementing a single neuron, exploring activation functions, analyzing threshold behavior through bias manipulation, understanding the need for multi-layer networks, and implementing softmax for multi-class classification.

---

## Introduction

Artificial Neural Networks (ANNs) are computational models inspired by biological neural networks. This lab focuses on understanding how individual neurons make decisions, how activation functions transform signals, how bias controls sensitivity, why depth is necessary for complex problems, and how softmax enables multi-class probability distributions. All experiments are conducted using NumPy and Matplotlib in a Python environment, simulating the behavior of neurons for stress detection from speech signals.

---

## Prerequisites

- Python 3.8 or higher
- NumPy library
- Matplotlib library
- Understanding of basic linear algebra and probability

---

## Task 1: Build a Neuron That Thinks, Not Just Calculates

### Objective
Design a single artificial neuron that decides stress levels from speech characteristics, exploring how different activation functions transform the same input differently.

### Procedure

#### Step 1: Choose Input Features
Three speech-related features were selected for stress detection:
- **Speech Rate** (0-1 scale): Measures how fast a person speaks
- **Pitch Variation** (0-1 scale): Measures instability in voice pitch
- **Pause Duration** (0-1 scale): Measures average pause length between words

For demonstration, we simulate:
- Speech Rate: 0.85 (high - fast speech)
- Pitch Variation: 0.72 (moderate-high - somewhat shaky)
- Pause Duration: 0.25 (short pauses)

#### Step 2: Assign Weights
Weights were assigned based on feature importance:
- **Speech Rate Weight: 0.4** (Moderate importance) - Fast speech can indicate stress but not always reliable
- **Pitch Variation Weight: 0.5** (Highest importance) - Key indicator of stress, as stressed speech typically shows pitch instability
- **Pause Duration Weight: 0.1** (Lowest importance) - Less reliable compared to pitch and rate

#### Step 3: Add Bias (Sensitivity Control)
Bias was set to -0.3 (moderately negative):
- **High positive bias**: Neuron fires easily, too sensitive
- **Low/negative bias**: Neuron is strict, requires strong evidence
- Our bias makes the neuron moderately strict, requiring evidence before activating

#### Step 4: Apply Different Activation Functions
The same weighted sum was passed through three activation functions:
- **Sigmoid**: `σ(z) = 1 / (1 + e^(-z))`
- **Tanh**: `tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))`
- **ReLU**: `ReLU(z) = max(0, z)`

### Code Description

The implementation computes a weighted sum of inputs plus bias, then applies each activation function:

```python
# Weighted sum calculation
weighted_sum = np.dot(inputs, weights) + bias

# Activation functions
output_sigmoid = 1 / (1 + np.exp(-weighted_sum))
output_tanh = np.tanh(weighted_sum)
output_relu = np.maximum(0, weighted_sum)
```

### Results

For the given inputs and weights:
- **Weighted Sum (z)**: 0.435
- **Sigmoid Output**: 0.6071 (60.71% confidence in stress)
- **Tanh Output**: 0.4106 (moderate positive activation)
- **ReLU Output**: 0.4350 (43.5% of maximum activation)

### Analysis: Why Does the Same Neuron Behave Differently?

#### 1. Soft Margin
- **Sigmoid (0.6071)**: Provides a soft, gradual response with no hard cutoff, giving a confidence level between 0-1
- **Tanh (0.4106)**: Also soft but centered at zero, can express both positive and negative responses
- **ReLU (0.4350)**: Hard threshold at zero - either activates (positive) or stays silent (zero)

#### 2. Confidence
- **Sigmoid**: Interprets output as probability (60.71% confidence in stress)
- **Tanh**: Shows moderate positive activation, indicating some evidence for stress
- **ReLU**: Shows raw activation strength (43.5% of maximum)

#### 3. Suppression vs Amplification
- **Sigmoid**: Only amplifies (0 to 1), cannot suppress negative signals
- **Tanh**: Can both amplify (positive) and suppress (negative) signals
- **ReLU**: Amplifies positive signals, completely suppresses negative ones

#### 4. Non-linearity
- **Sigmoid**: Highly non-linear S-curve enabling smooth transitions
- **Tanh**: Non-linear but symmetric, providing balanced response
- **ReLU**: Piecewise linear, simple but effective for deep networks

**Conclusion**: The same weighted sum produces different outputs because each activation function transforms evidence differently - sigmoid interprets as probability, tanh as signed strength, and ReLU as raw activation.

---

## Task 2: The Threshold Experiment

### Objective
Control when the neuron fires by systematically varying bias while keeping inputs and weights constant, observing how different activation functions respond to bias changes.

### Procedure

#### Step 1: Fix Everything Except Bias
- Inputs: [0.85, 0.72, 0.25] (fixed)
- Weights: [0.4, 0.5, 0.1] (fixed)
- Weighted sum (without bias): 0.735

#### Step 2: Sweep Bias Values
Bias was gradually changed from -3.0 to +3.0 (300 points) to create smooth curves.

#### Step 3: Plot Behavior
For each bias value, the neuron output was computed using:
- Sigmoid activation
- ReLU activation
- Tanh activation (for comparison)

### Code Description

The implementation sweeps bias values and computes outputs:

```python
bias_range = np.linspace(-3.0, 3.0, 300)
z_values = weighted_sum_no_bias + bias_range

sigmoid_outputs = sigmoid(z_values)
tanh_outputs = tanh(z_values)
relu_outputs = relu(z_values)
```

### Results

The plots reveal distinct "wake-up" behaviors:

#### Sigmoid - Gradual Wake-Up
- **Wake-up bias**: Approximately -0.735 (where output = 0.5)
- **Behavior**: Smooth S-curve with gradual transition
- **Why gradual**: No sharp transition; output changes smoothly from 0 to 1, providing soft responses even before full activation

#### ReLU - Sudden Wake-Up
- **Wake-up bias**: -0.735 (where weighted sum z becomes positive)
- **Behavior**: Hard threshold at z = 0
- **Why sudden**: Binary-like behavior - completely silent (output = 0) for z < 0, then linear activation (output = z) for z > 0

#### Tanh - Neutral Zone
- **Neutral point bias**: -0.735 (where output = 0)
- **Behavior**: Outputs range from -1 to +1, centered at zero
- **Why neutral zone**: Can express suppression (negative), neutral (zero), or activation (positive)

### Critical Thinking

**At what bias does the neuron "wake up"?**
- **Sigmoid**: Gradual wake-up around -0.735, providing soft confidence levels
- **ReLU**: Sudden wake-up at -0.735, switching from OFF to ON instantly
- **Tanh**: Crosses neutral point at -0.735, transitioning from suppression to activation

**Key Insight**: Bias controls WHEN the neuron activates, but HOW it activates depends on the activation function. Sigmoid provides probabilistic thinking, ReLU provides binary-like decisions, and Tanh provides signed activation with a neutral zone.

---

## Task 3: Why One Neuron Is Not Enough (The Depth Problem)

### Objective
Demonstrate why a single neuron fails for complex patterns and how adding a hidden layer enables the network to learn non-linear decision boundaries.

### Scenario
The system fails to distinguish between:
- **Type 1**: Calm but fast speakers (high speech rate, low pitch variation)
- **Type 2**: Stressed but slow speakers (low speech rate, high pitch variation)

### Part A: Try With ONE Neuron

#### Procedure
A single neuron with weights [0.3, 0.6, 0.1] and bias -0.2 was tested on multiple examples.

#### Results
The single neuron produced similar outputs for both types:
- Calm but fast: Output ≈ 0.45-0.50
- Stressed but slow: Output ≈ 0.50-0.55

**Problem**: The neuron cannot distinguish between these patterns because it can only create a linear decision boundary. Both patterns produce overlapping outputs.

### Part B: Add a Hidden Layer (Manual Forward Pass)

#### Network Architecture
- **Input layer**: 3 features (speech rate, pitch variation, pause duration)
- **Hidden layer**: 2 neurons
  - Neuron 1: Specialized for fast speech patterns (weights: [0.5, 0.2, 0.1])
  - Neuron 2: Specialized for pitch instability (weights: [0.1, 0.6, 0.2])
- **Output layer**: 1 neuron (combines hidden layer outputs)

#### Code Description

```python
def mlp_forward_pass(inputs):
    # Hidden layer
    z1 = np.dot(inputs, W_hidden_1) + b_hidden_1
    z2 = np.dot(inputs, W_hidden_2) + b_hidden_2
    a1 = sigmoid(z1)  # Fast speech detector
    a2 = sigmoid(z2)  # Pitch instability detector
    
    # Output layer
    z_out = np.dot([a1, a2], W_output) + b_output
    output = sigmoid(z_out)
    return [a1, a2], output
```

#### Results
The MLP with hidden layer produces distinct outputs:
- Calm but fast: Lower output (correctly identified as calm)
- Stressed but slow: Higher output (correctly identified as stressed)

The hidden layer learns intermediate features:
- Neuron 1 detects "fast speech patterns"
- Neuron 2 detects "pitch instability patterns"
- Output neuron combines these to make the final decision

### Reflection: Why Does Adding a Layer Help?

#### 1. Feature Transformation
The hidden layer transforms raw inputs into new feature representations:
- Low-level: Raw features (speech rate, pitch, pauses)
- Mid-level: Learned patterns (fast speech pattern, pitch instability)
- High-level: Final decision (stress vs calm)

#### 2. Space Bending
- **Single neuron**: Can only create a straight line (linear boundary)
- **Hidden layer**: Creates curved boundaries (non-linear)
- The combination of neurons allows the network to "bend" the decision space

#### 3. Combination of Soft Decisions
Each hidden neuron makes a soft decision (0 to 1), and the output neuron combines these:
- Logic emerges: "If fast speech AND low pitch variation → calm"
- This logic is learned through weight combinations

#### 4. Non-linear Composition
- **Single neuron**: `f(x) = sigmoid(W·x + b)` → still linear in x
- **MLP**: `f(x) = sigmoid(W2 · sigmoid(W1·x + b1) + b2)` → nested sigmoids create non-linearity

**Key Insight**: Depth allows hierarchical learning - low-level features → mid-level patterns → high-level decisions. This is why deep networks can solve complex problems that single neurons cannot.

---

## Task 4: Softmax

### Objective
Implement softmax activation for multi-class classification, converting raw network scores into probability distributions across three emotional states: Calm, Anxious, and Stressed.

### Scenario
The network now predicts three emotional states with raw scores (logits):
- Calm: z₁ = 2.5
- Anxious: z₂ = 1.8
- Stressed: z₃ = 0.9

### Step 1: Apply Softmax Manually

#### Procedure
Softmax was computed step-by-step:

1. **Subtract maximum** (for numerical stability):
   - max(z₁, z₂, z₃) = 2.5
   - Shifted scores: [0.0, -0.7, -1.6]

2. **Compute exponentials**:
   - exp(0.0) = 1.0000
   - exp(-0.7) = 0.4966
   - exp(-1.6) = 0.2019

3. **Compute sum**: 1.0000 + 0.4966 + 0.2019 = 1.6985

4. **Divide each by sum**:
   - P(Calm) = 1.0000 / 1.6985 = 0.5889
   - P(Anxious) = 0.4966 / 1.6985 = 0.2925
   - P(Stressed) = 0.2019 / 1.6985 = 0.1186

### Code Description

```python
def softmax(z):
    z_shifted = z - np.max(z)  # Numerical stability
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)
```

### Step 2: Verify Probability Behavior

#### Verification Results
- ✓ **All outputs in [0, 1]**: All probabilities are valid (between 0 and 1)
- ✓ **Sum equals 1**: Total probability = 1.0000 (valid probability distribution)

### Interpretation: Why Does Increasing One Score Reduce Others?

#### Experiment
When the Calm score (z₁) was increased from 2.5 to 3.5:
- **Original**: P(Calm)=0.5889, P(Anxious)=0.2925, P(Stressed)=0.1186
- **Modified**: P(Calm)=0.7311 (+0.1422), P(Anxious)=0.1978 (-0.0947), P(Stressed)=0.0711 (-0.0475)

#### Explanation

1. **Competition**: Softmax is a zero-sum game. Total probability must equal 1.0. If P(Calm) increases, remaining probabilities must decrease.

2. **Relative Confidence**: Softmax compares relative scores, not absolute values. Increasing z₁ makes it relatively larger compared to z₂ and z₃, so P(Calm) increases while others decrease proportionally.

3. **Soft Margins Across Classes**: Unlike hard classification (winner takes all), softmax gives probabilities to ALL classes, reflecting uncertainty: "I'm 58.89% sure it's Calm, 29.25% Anxious, 10.86% Stressed."

**Conclusion**: Softmax converts raw scores into a probability distribution where classes compete for probability mass, creating interpretable confidence scores for each class.

---

## Results and Observations

### Task 1 Results
- Successfully implemented a single neuron with three activation functions
- Demonstrated how the same input produces different outputs based on activation function
- Sigmoid provides probabilistic interpretation, Tanh provides signed activation, ReLU provides raw activation

### Task 2 Results
- Systematically explored bias effects on neuron activation
- Identified distinct "wake-up" behaviors for each activation function
- Sigmoid wakes up gradually, ReLU wakes up suddenly, Tanh has a neutral zone

### Task 3 Results
- Demonstrated limitations of single neuron (linear boundaries)
- Showed how hidden layers enable non-linear decision boundaries
- MLP successfully distinguishes complex patterns that single neuron cannot

### Task 4 Results
- Implemented softmax for multi-class classification
- Verified probability properties (sum to 1, all in [0,1])
- Demonstrated competition between classes in probability space

---

## Conclusion

This lab provided comprehensive hands-on experience with the fundamental building blocks of artificial neural networks. Through implementing a single neuron, I learned that neurons are not mere calculators but decision-makers that answer "Is the signal strong enough to matter?" The choice of activation function dramatically changes how a neuron interprets and responds to evidence - sigmoid provides soft probabilistic thinking, ReLU provides binary-like decisions, and tanh provides balanced signed activation.

The threshold experiment revealed how bias acts as a sensitivity control mechanism. By systematically varying bias, I observed that different activation functions "wake up" differently - sigmoid gradually, ReLU suddenly, and tanh with a neutral zone. This understanding is crucial for designing networks that respond appropriately to different types of signals.

The depth problem task demonstrated why single neurons fail for complex patterns. A single neuron can only create linear boundaries, but real-world problems often require non-linear decision surfaces. By adding a hidden layer, the network learned to transform raw features into intermediate patterns (fast speech detection, pitch instability detection) and combine them hierarchically to make final decisions. This hierarchical learning - from low-level features to high-level abstractions - is the essence of deep learning.

The softmax implementation showed how raw network scores can be converted into interpretable probability distributions. The zero-sum nature of softmax creates competition between classes, where increasing confidence in one class automatically decreases confidence in others. This provides a natural way to express uncertainty and relative confidence across multiple classes.

Overall, this lab taught me that neural networks work not because computers are powerful, but because they are structured decision systems. Each neuron makes micro-decisions about what to keep, what to suppress, when to fire, and when to stay silent. These decisions, when combined across layers, create intelligence that emerges from mathematics. The experience of implementing these concepts from scratch gave me deep insight into how activation functions turn rigid mathematical rules into human-like decision-making through soft margins, non-linearity, sensitivity control, and confidence expression.

---

## References

1. Lab Manual: Lab 02 - Understanding ANN from Scratch
2. NumPy Documentation: https://numpy.org/doc/
3. Matplotlib Documentation: https://matplotlib.org/

---

**Prepared by:** Ali Hamza  
**Roll Number:** B23F0063AI106  
**Section:** B.S AI - Red  
**Date:** January 26, 2026
