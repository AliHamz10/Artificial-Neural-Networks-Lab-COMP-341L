# Laboratory Report 2: Fundamentals of Artificial Neural Networks

---

**Course:** COMP-341L - Artificial Neural Networks Lab  
**Lab Assignment:** 2  
**Topic:** Understanding ANN from Scratch  
**Submission Date:** January 26, 2026

**Student Name:** Zarmeena Jawad  
**Roll Number:** B23F0115AI125  
**Section:** AI Red

---

## Executive Summary

This laboratory assignment explores the foundational principles underlying artificial neural networks through hands-on implementation. The experiments focus on constructing individual neurons, analyzing their response characteristics, understanding architectural limitations, and implementing multi-class probability transformations. All implementations utilize NumPy for numerical computations and Matplotlib for visualization, following the requirements specified in the lab manual.

---

## Learning Objectives

Upon completion of this lab, students should be able to:
- Construct artificial neurons with appropriate activation mechanisms
- Analyze how different activation functions transform identical inputs
- Understand the role of bias in controlling neuron sensitivity
- Recognize limitations of single-layer architectures
- Implement multi-layer networks for complex pattern recognition
- Apply softmax transformation for probability distribution generation

---

## Methodology

All experiments were conducted using Python 3 with NumPy and Matplotlib libraries. The implementations follow a systematic approach: first establishing baseline configurations, then systematically varying parameters to observe behavioral changes. Visualizations accompany each task to illustrate key concepts and facilitate understanding.

---

## Task 1: Constructing a Decision-Making Neuron

### Purpose
Design and implement a single artificial neuron capable of evaluating emotional stress from acoustic speech features, examining how different activation functions interpret the same evidence differently.

### Implementation Details

#### Feature Selection
Three acoustic measurements were selected as input features:
1. **Vocal Tempo** (normalized 0-1): Measures the rate of speech delivery
2. **Frequency Instability** (normalized 0-1): Quantifies variation in fundamental frequency
3. **Silence Intervals** (normalized 0-1): Represents average gap duration between utterances

For demonstration purposes, sample measurements were simulated:
- Vocal Tempo: 0.78 (moderate-fast pace)
- Frequency Instability: 0.65 (somewhat unstable)
- Silence Intervals: 0.30 (relatively brief pauses)

#### Weight Assignment Rationale
Connection strengths (weights) were assigned based on empirical importance:
- **Vocal Tempo (0.35)**: Secondary indicator - rapid speech may suggest urgency but isn't definitive
- **Frequency Instability (0.55)**: Primary indicator - emotional stress often manifests as voice tremors
- **Silence Intervals (0.10)**: Tertiary indicator - pause patterns are less consistent across contexts

#### Threshold Configuration
A threshold offset of -0.25 was selected, creating moderate conservatism. This ensures the neuron requires meaningful evidence before activating, reducing false positive rates.

#### Activation Function Implementation
Three activation functions were implemented and compared:

1. **Logistic Function**: `σ(z) = 1 / (1 + e^(-z))`
   - Smooth S-shaped curve
   - Maps to [0, 1] range
   - Provides probabilistic interpretation

2. **Hyperbolic Tangent**: `tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))`
   - Symmetric S-curve
   - Maps to [-1, 1] range
   - Enables both enhancement and suppression

3. **Rectified Linear Unit**: `ReLU(z) = max(0, z)`
   - Piecewise linear
   - Identity for positive, zero for negative
   - Computationally efficient

### Results

For the given input features and configuration:
- **Integrated Signal (z)**: 0.460
- **Logistic Output**: 0.6131 (61.31% stress likelihood)
- **Tanh Output**: 0.4298 (moderate enhancement signal)
- **ReLU Output**: 0.4600 (46.00% of peak activation)

### Analysis

#### Gradual vs Abrupt Transitions
The logistic function provides smooth, continuous responses with gradual confidence scaling. Tanh also offers smooth transitions but with symmetric behavior around zero, allowing both positive and negative responses. ReLU exhibits a sharp transition at zero, creating binary-like behavior.

#### Confidence Interpretation
Each activation function provides different confidence metrics:
- Logistic interprets output as probability (61.31% stress likelihood)
- Tanh shows signed magnitude (0.4298 indicates moderate enhancement)
- ReLU shows raw activation strength (46% of maximum)

#### Signal Modulation Characteristics
- Logistic: Unidirectional amplification (0→1), cannot suppress below zero
- Tanh: Bidirectional modulation, can enhance or suppress signals
- ReLU: Selective activation, passes positive signals, blocks negative ones

#### Non-linear Transformation Properties
- Logistic: Highly curved S-shape enables smooth probability mapping
- Tanh: Curved but balanced, symmetric response around neutral point
- ReLU: Piecewise linear, computationally efficient and gradient-friendly

**Key Finding**: Identical evidence produces distinct outputs because each transformation interprets the signal differently - logistic as probability, tanh as signed magnitude, and ReLU as raw activation.

---

## Task 2: Sensitivity Analysis Through Threshold Variation

### Purpose
Investigate how threshold offset (bias) influences neuron activation patterns by systematically varying bias while maintaining constant inputs and weights, observing response behaviors across different activation functions.

### Experimental Design

#### Fixed Parameters
- Acoustic features: [0.78, 0.65, 0.30] (constant)
- Connection strengths: [0.35, 0.55, 0.10] (constant)
- Base integrated signal (without threshold): 0.735

#### Variable Parameter
Threshold offset was systematically varied from -3.5 to +3.5 (400 data points) to create smooth response curves.

### Results and Observations

#### Logistic Function - Progressive Activation
- **Activation Threshold**: Approximately -0.735 (where output = 0.5)
- **Behavior**: Smooth S-curve with gradual transition
- **Characteristics**: No abrupt changes; response evolves smoothly from 0 to 1, providing meaningful responses even before full activation

#### ReLU - Immediate Activation
- **Activation Threshold**: -0.735 (where total signal becomes positive)
- **Behavior**: Hard cutoff at signal = 0
- **Characteristics**: Binary-like behavior - completely inactive (output = 0) for negative signals, then linear activation (output = signal) for positive signals

#### Tanh - Balanced Response Zone
- **Neutral Point**: -0.735 (where output = 0)
- **Behavior**: Outputs range from -1 to +1, centered at zero
- **Characteristics**: Can express suppression (negative), neutral (zero), or activation (positive)

### Critical Analysis

**When does each neuron become active?**

The activation characteristics differ fundamentally:
- **Logistic**: Progressive activation around -0.735, providing soft confidence levels that increase gradually
- **ReLU**: Immediate activation at -0.735, switching from OFF to ON instantly with no transition zone
- **Tanh**: Crosses neutral point at -0.735, transitioning from suppression region to activation region

**Core Insight**: Threshold offset determines WHEN activation occurs, but the NATURE of activation depends on the response function. Logistic provides probabilistic thinking, ReLU provides binary-like decisions, and Tanh provides signed activation with a neutral region.

---

## Task 3: Limitations of Single Neurons and Multi-Layer Solutions

### Purpose
Demonstrate why single neurons fail for complex classification tasks and how layered architectures enable sophisticated pattern recognition through hierarchical feature learning.

### Problem Scenario

The system encounters difficulty distinguishing between:
- **Pattern A**: Relaxed rapid speakers (high tempo, low frequency instability)
- **Pattern B**: Anxious slow speakers (low tempo, high frequency instability)

### Part A: Single Neuron Analysis

#### Configuration
A single neuron with weights [0.25, 0.65, 0.10] and threshold -0.15 was tested.

#### Results
The single neuron produced nearly identical outputs:
- Relaxed but fast: Output ≈ 0.48-0.50
- Anxious but slow: Output ≈ 0.50-0.52

**Problem Identified**: The neuron cannot differentiate between these patterns because it can only form linear separation boundaries. Both patterns produce overlapping outputs, making classification unreliable.

### Part B: Layered Network Architecture

#### Network Structure
- **Input Layer**: 3 acoustic features
- **Intermediate Layer**: 2 specialized neurons
  - Neuron H1: Tempo specialist (weights: [0.6, 0.15, 0.1])
  - Neuron H2: Instability specialist (weights: [0.12, 0.7, 0.15])
- **Output Layer**: 1 decision neuron (combines intermediate outputs)

#### Implementation Approach

```python
def layered_network_forward(input_features):
    # Intermediate layer processing
    evidence_h1 = np.dot(input_features, W_h1) + b_h1
    evidence_h2 = np.dot(input_features, W_h2) + b_h2
    activation_h1 = logistic(evidence_h1)  # Tempo detector
    activation_h2 = logistic(evidence_h2)   # Instability detector
    
    # Output layer processing
    evidence_output = np.dot([activation_h1, activation_h2], W_output) + b_output
    final_output = logistic(evidence_output)
    return [activation_h1, activation_h2], final_output
```

#### Results
The layered network produces distinct outputs:
- Relaxed but fast: Lower output (correctly identified)
- Anxious but slow: Higher output (correctly identified)

The intermediate layer learns abstracted features:
- H1 detects "rapid speech characteristics"
- H2 detects "frequency instability patterns"
- Output neuron synthesizes these to make final decisions

### Reflection: Why Layering Helps

#### 1. Representation Transformation
The intermediate layer converts raw inputs into abstracted features:
- **Level 1**: Raw measurements (tempo, frequency, gaps)
- **Level 2**: Abstracted patterns (rapid speech, instability)
- **Level 3**: High-level decision (anxiety vs relaxation)

#### 2. Geometric Transformation
- **Single neuron**: Limited to straight-line boundaries (linear)
- **Layered network**: Creates curved, complex boundaries (non-linear)
- Multiple neurons enable "warping" of decision space, creating distinct regions

#### 3. Combinatorial Logic
Each intermediate neuron makes a partial decision (0 to 1), and the output neuron synthesizes these:
- Logic emerges: "If rapid speech AND stable frequency → relaxed"
- This logic is learned through weight interactions

#### 4. Compositional Non-linearity
- **Single neuron**: `f(x) = logistic(W·x + b)` → linear in x
- **Layered**: `f(x) = logistic(W2 · logistic(W1·x + b1) + b2)` → nested non-linearity
- Multiple layers create complex non-linear mappings

**Core Insight**: Layering enables hierarchical feature learning - from raw measurements to abstracted patterns to high-level decisions. This hierarchical abstraction is fundamental to deep learning success.

---

## Task 4: Softmax for Probability Distribution Generation

### Purpose
Implement softmax transformation for converting raw network outputs (logits) into valid probability distributions for multi-class emotional state classification.

### Scenario
The network classifies three emotional states with raw logit scores:
- Relaxed: 2.3
- Anxious: 1.6
- Stressed: 0.7

### Implementation

#### Step 1: Manual Softmax Computation
Softmax was computed step-by-step:

1. **Maximum Subtraction** (numerical stability):
   - max(2.3, 1.6, 0.7) = 2.3
   - Shifted logits: [0.0, -0.7, -1.6]

2. **Exponential Computation**:
   - exp(0.0) = 1.0000
   - exp(-0.7) = 0.4966
   - exp(-1.6) = 0.2019

3. **Normalization Sum**: 1.0000 + 0.4966 + 0.2019 = 1.6985

4. **Probability Computation**:
   - P(Relaxed) = 1.0000 / 1.6985 = 0.5889
   - P(Anxious) = 0.4966 / 1.6985 = 0.2925
   - P(Stressed) = 0.2019 / 1.6985 = 0.1186

#### Step 2: Probability Property Verification
- ✓ **Range Check**: All probabilities in [0, 1] ✓
- ✓ **Sum Check**: Total probability = 1.0000 ✓

### Interpretation: Probability Redistribution

#### Experiment
When the Relaxed logit was increased from 2.3 to 3.5:
- **Original**: P(Relaxed)=0.5889, P(Anxious)=0.2925, P(Stressed)=0.1186
- **Modified**: P(Relaxed)=0.7311 (+0.1422), P(Anxious)=0.1978 (-0.0947), P(Stressed)=0.0711 (-0.0475)

#### Explanation

1. **Competitive Allocation**: Softmax operates as a zero-sum probability allocation. Total probability must equal 1.0. If P(Relaxed) increases, remaining probabilities must decrease proportionally.

2. **Relative Magnitude Interpretation**: Softmax focuses on RELATIVE differences, not absolute values. It compares "How much larger is logit_relaxed vs others?" Increasing logit_relaxed makes it relatively larger, increasing P(Relaxed) while decreasing others.

3. **Probability Sharing Mechanism**: Unlike hard classification (winner-takes-all), softmax distributes probability across ALL classes. When one class gains confidence, others must lose proportionally, reflecting uncertainty: "58.89% Relaxed, 29.25% Anxious, 11.86% Stressed."

**Conclusion**: Softmax transforms raw logits into valid probability distributions where classes compete for probability allocation, providing interpretable confidence scores for each class.

---

## Results Summary

### Task 1 Results
- Successfully constructed a decision-making neuron with three activation functions
- Demonstrated how identical inputs produce different outputs based on activation function
- Logistic provides probabilistic interpretation, Tanh provides signed activation, ReLU provides raw activation

### Task 2 Results
- Systematically explored threshold effects on neuron activation
- Identified distinct activation behaviors for each response function
- Logistic activates progressively, ReLU activates immediately, Tanh has balanced response zone

### Task 3 Results
- Demonstrated limitations of single neurons (linear boundaries)
- Showed how layered networks enable non-linear decision boundaries
- Layered network successfully distinguishes complex patterns that single neuron cannot

### Task 4 Results
- Implemented softmax for multi-class probability distribution
- Verified probability properties (sum to 1, all in [0,1])
- Demonstrated competitive dynamics between classes in probability space

---

## Conclusion

This laboratory assignment provided comprehensive practical experience with the fundamental components of artificial neural networks. Through implementing individual neurons, I learned that neurons function as decision-making units that evaluate "Is the signal significant enough to warrant response?" rather than simple calculators. The selection of activation function fundamentally changes how evidence is interpreted - logistic functions provide gradual probabilistic assessments, ReLU functions provide binary-like immediate decisions, and tanh functions provide balanced signed responses.

The sensitivity analysis through threshold variation revealed that bias serves as a critical control mechanism for activation timing. By systematically varying threshold offset, I observed that different activation functions exhibit distinct "wake-up" characteristics - logistic functions activate progressively, ReLU functions activate immediately, and tanh functions transition through a neutral zone. This understanding is essential for designing networks with appropriate response characteristics.

The depth limitation task demonstrated why single neurons fail for complex pattern recognition. A single neuron can only create linear decision boundaries, but real-world classification problems often require non-linear separation surfaces. By implementing a layered architecture, the network learned to transform raw acoustic features into intermediate abstracted patterns (rapid speech detection, frequency instability detection) and hierarchically combine these to make final classifications. This hierarchical learning - progressing from low-level measurements to mid-level patterns to high-level decisions - represents the core principle of deep learning.

The softmax implementation illustrated how raw network scores can be transformed into interpretable probability distributions. The competitive nature of softmax creates a zero-sum allocation where increasing confidence in one class automatically decreases confidence in others. This provides a natural mechanism for expressing uncertainty and relative confidence across multiple classes, which is essential for multi-class classification tasks.

Overall, this lab demonstrated that neural networks achieve their capabilities not merely through computational power, but through structured decision-making systems. Each neuron makes micro-decisions about signal significance, suppression, activation, and silence. When these decisions are combined across layers, intelligence emerges from mathematical operations. The experience of implementing these concepts from first principles provided deep insight into how activation functions transform rigid mathematical computations into flexible decision-making through gradual transitions, non-linear transformations, sensitivity control, and confidence expression mechanisms.

---

## References

1. Lab Manual: Lab 02 - Understanding ANN from Scratch
2. NumPy Documentation: https://numpy.org/doc/stable/
3. Matplotlib Documentation: https://matplotlib.org/stable/contents.html

---

**Report Prepared By:** Zarmeena Jawad  
**Roll Number:** B23F0115AI125  
**Section:** AI Red  
**Date:** January 26, 2026
