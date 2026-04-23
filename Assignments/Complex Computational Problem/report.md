# Complex Computational Problem (CCP) — Artificial Neural Network (MNIST)

**Class:** BSAI F23 Red/Blue  
**Due date:** 25-04-2026 (MS Teams)  
**Weightage:** 5%  
**Course:** Artificial Neural Network  
**Department:** IT & Computer Science — Pak-Austria Fachhochschule: Institute of Applied Sciences & Technology  

**Title:** Neural Computing System for Handwritten Digit Recognition (MNIST)  
**Students:** Ali Hamza (B23F0063AI106) & Zarmeena Jawad (B23F0115AI125)  
**Section:** B.S AI Red  
**Instructor:** Dr. Abid Ali  
**Submission date:** 25-04-2026  

> Report formatting target (Word/PDF): Times New Roman, 12pt, 1.5 line spacing, justified, page numbers bottom-center.

---

## Abstract

This project implements a **complete neural computing pipeline** for recognizing handwritten digits (0–9) from the MNIST dataset, using **NumPy-only** model implementations (external libraries are used strictly for *loading* MNIST). We sequentially implement and evaluate four foundational paradigms: (1) a **single-layer Perceptron** as a baseline binary classifier, (2) an **Adaline** (Adaptive Linear Neuron) trained by the **Delta Rule** via gradient descent, (3) a **Kohonen Self-Organizing Map (SOM)** for unsupervised clustering and topology-preserving feature learning, and (4) a **multi-layer perceptron (MLP)** trained with **Backpropagation** for full 10-class digit classification. Experimental outputs include decision boundary evolution for the Perceptron, MSE loss surfaces and learning-rate convergence curves for Adaline, SOM visualizations (U-matrix, prototype maps, class hit maps) and quality metrics (Quantization Error, Topographic Error), anomaly detection using per-sample quantization error, and MLP training/validation curves with activation and regularization comparisons. On our MNIST setup, the final MLP achieved **97.14% test accuracy**, while the Perceptron reached **99.34%** accuracy on a binary (0 vs 1) PCA-2D visualization task. The SOM produced **QE = 5.535** and **TE = 0.1078** on a representative subset, enabling anomaly detection with **precision = 0.8299**, **recall ≈ 1.0**, and **F1 = 0.9070** at a 95th-percentile threshold. Overall, the four paradigms form a unified learning system illustrating how linear decision rules evolve into gradient-based multi-layer representations and how unsupervised topological maps can support analysis and auxiliary feature design.

---

## Reproducibility (what to run)

- Main notebook (run top → bottom): `ccp_ann_project.ipynb`
- Alternate notebooks: `notebooks/ccp_ann.ipynb`, `notebooks/ccp_ann_colab.ipynb`
- NumPy-only implementations: `src/` (`perceptron.py`, `adaline.py`, `som.py`, `mlp.py`)
- Figures used in this report: `figures/`

**Important note:** The notebook uses TensorFlow/Keras (or equivalent) **only for loading MNIST**. All models are from scratch in NumPy.

---

## Course Learning Outcomes (CLOs)

By the end of this project, students will be able to:

- Understand the biological basis of neural computation and implement a Perceptron model for binary classification tasks.
- Apply the Delta Rule to train Adaline networks through gradient descent and analyze learning rate effects on convergence.
- Design and train Kohonen Self-Organizing Maps for unsupervised feature learning, clustering, and topology-preserving dimensionality reduction.
- Implement a multi-layer Backpropagation network from scratch to solve complex non-linear classification problems.
- Evaluate and compare neural network models using performance metrics including accuracy, confusion matrices, and loss curves.
- Integrate multiple neural learning paradigms into a unified system and critically analyze their strengths and limitations.

---

## Project overview (what we built)

This project implements a complete MNIST digit-recognition pipeline by building four paradigms sequentially:

1. **Perceptron (Part A):** baseline binary classifier (0 vs 1) with PCA-2D decision boundary visualizations.
2. **Adaline (Part B):** Delta Rule derivation + MSE surface + learning-rate convergence study.
3. **SOM (Part C):** unsupervised competitive learning with U-matrix, prototype maps, hit maps, QE/TE, and anomaly detection.
4. **MLP Backprop (Part D):** 784→128→64→10 classifier with activation comparison, regularization, and SOM-feature integration.

---

## 1. Introduction

### 1.1 Motivation and objectives

Neural computation is inspired by biological neurons and their emergent behavior when organized into networks. Historically, early neural models such as the **Perceptron** introduced the core idea that a weighted combination of inputs can implement a decision boundary. Over time, limitations of single-layer models motivated differentiable learning rules (e.g., the **Delta Rule** for Adaline) and, eventually, the generalization to deep, multi-layer networks trained by **Backpropagation**. Separately, unsupervised paradigms such as **Kohonen Self-Organizing Maps (SOMs)** provide a different learning lens: they compress high-dimensional structure into a topology-preserving grid that supports clustering, visualization, and anomaly detection.

**Objective of this project:** Build a comprehensive neural computing system for MNIST digit recognition by implementing four paradigms and comparing them using consistent metrics and visual evidence.

### 1.2 Dataset: MNIST

MNIST contains 28×28 grayscale images of handwritten digits (0–9). Each image is flattened into a 784-dimensional vector:

\[
\mathbf{x}\in\mathbb{R}^{784}, \quad x_i \in [0,1]
\]

We use standard train/test splits. For the MLP, standardization (mean removal and variance scaling) stabilizes gradient descent, but this preprocessing does not violate the “NumPy-only model” constraint because it is a deterministic transformation applied to inputs.

### 1.3 Paradigms overview (how the pipeline is structured)

1. **Perceptron (Part A):** A baseline supervised model for binary linear classification; used to study linear separability and convergence.
2. **Adaline + Delta Rule (Part B):** A supervised linear model trained with gradient descent on MSE; serves as a bridge to backpropagation.
3. **SOM (Part C):** An unsupervised, competitive learning map; used for clustering, visualization, and anomaly detection via quantization error.
4. **Backpropagation MLP (Part D):** A supervised multi-layer classifier for 10-class MNIST recognition; includes activation, regularization, and SOM-feature integration analysis.

---

## 2. Part A — Perception and the Perceptron Model

### 2.1 A(i) Biological mapping and historical context

The Perceptron is motivated by the biological neuron. A simplified mapping is:

- **Dendrites** → input features \(x_1, x_2, \dots, x_d\)
- **Synapses** → weights \(w_1, w_2, \dots, w_d\) (synaptic strengths)
- **Soma** → weighted sum + bias \(net = \sum_i w_i x_i + b\)
- **Axon firing** → threshold activation producing an output spike (binary decision)

**Figure A1. Biological mapping diagram.**  
Insert: `figures/partA_biological_mapping.png`

Historically, the Perceptron offered an early computational model of learning by updating weights from errors. However, it is limited to **linearly separable** problems because it implements a single hyperplane decision boundary. This limitation is famously illustrated by the XOR problem, motivating multi-layer networks.

### 2.2 A(ii) Perceptron formulation

For input \(\mathbf{x}\in\mathbb{R}^d\) with bias \(b\):

\[
net = \mathbf{w}^\top \mathbf{x} + b
\]

Step activation:

\[
y = \mathbb{1}[net \ge 0]
\]

Perceptron update (for one sample):

\[
\mathbf{w} \leftarrow \mathbf{w} + \eta (d - y)\mathbf{x},\quad
b \leftarrow b + \eta(d - y)
\]

where \(d\in\{0,1\}\) is the desired label and \(\eta\) is the learning rate.

### 2.3 A(ii) Perceptron implementation (NumPy only)

The implementation used in this project is in `src/perceptron.py`. The model stores a history of misclassifications and weight snapshots per epoch.

**Listing A1. Core Perceptron update loop (from `src/perceptron.py`).**

```python
for _ in range(self.epochs):
    miscls = 0
    for xi, di in zip(xb, y):
        yi = 1 if (xi @ self.w) >= 0 else 0
        err = di - yi
        if err != 0:
            miscls += 1
            self.w = self.w + self.lr * err * xi
    self.history_["miscls"].append(int(miscls))
    self.history_["w"].append(self.w.copy())
```

### 2.4 A(ii) MNIST binary experiment and decision boundary plots

Because MNIST is 784-dimensional, **direct decision boundary plotting is not possible**. To visualize learning, we project MNIST digits into 2D using PCA and train a Perceptron on the PCA-2D space for a binary task (digits **0 vs 1**). This preserves the interpretability of a linear boundary while still using real MNIST samples.

**Figure A2. Misclassifications per epoch (MNIST 0 vs 1 in PCA-2D).**  
Insert: `figures/partA_perceptron_miscls.png`

**Figures A3–A8. Decision boundary evolution during training.**  
Insert:
- `figures/partA_perceptron_boundary_epoch_01.png`
- `figures/partA_perceptron_boundary_epoch_02.png`
- `figures/partA_perceptron_boundary_epoch_03.png`
- `figures/partA_perceptron_boundary_epoch_06.png`
- `figures/partA_perceptron_boundary_epoch_11.png`
- `figures/partA_perceptron_boundary_epoch_35.png`

**Observed result:** Test accuracy on this binary task (PCA-2D) was:

- **Perceptron test accuracy (0 vs 1, PCA-2D): 0.99338**

This high value is expected because digits 0 and 1 are relatively separable, and PCA-2D still preserves strong class structure for these digits.

### 2.5 A(iii) Convergence analysis: separable vs non-separable

**Perceptron convergence (linear separability).** When a dataset is linearly separable, there exists a hyperplane that classifies all training samples correctly. The Perceptron convergence theorem states that repeated updates will find a separating hyperplane in a finite number of steps (given a fixed margin and bounded inputs). Empirically, we demonstrate convergence on a synthetic linearly separable dataset.

**Figure A9. Convergence on linearly separable data.**  
Insert: `figures/partA_perceptron_convergence_linearsep.png`

**Perceptron failure (XOR).** XOR is not linearly separable: no single line can separate the classes. Consequently, the Perceptron cannot reduce misclassifications to zero; the updates oscillate.

**Figure A10. XOR failure visualization.**  
Insert: `figures/partA_perceptron_xor_failure.png`

**Key takeaway:** The Perceptron is a powerful conceptual baseline, but its representational capacity is fundamentally limited by linear separability.

---

## 3. Part B — The Delta Rule and Adaptive Linear Neurons (Adaline)

### 3.1 B(i) Deriving the Delta Rule from MSE

Unlike the Perceptron (which uses a hard step output), Adaline uses a **continuous linear output during training**:

\[
\hat{y} = \mathbf{w}^\top \mathbf{x} + b
\]

For a dataset \(\{(\mathbf{x}^{(n)}, d^{(n)})\}_{n=1}^N\), define the Mean Squared Error objective:

\[
E(\mathbf{w},b)=\frac{1}{2N}\sum_{n=1}^{N}\left(d^{(n)}-\hat{y}^{(n)}\right)^2
\]

For a single sample, the error term is:

\[
e = d - \hat{y}
\]

Differentiate \(E\) with respect to weight \(w_i\):

\[
\frac{\partial E}{\partial w_i}
=\frac{\partial}{\partial w_i}\left(\frac{1}{2N}\sum_n e_n^2\right)
=\frac{1}{N}\sum_n e_n\frac{\partial e_n}{\partial w_i}
\]

Since \(e_n = d_n - \hat{y}_n\) and \(\hat{y}_n = \sum_j w_j x_{n,j} + b\),

\[
\frac{\partial e_n}{\partial w_i}
= -\frac{\partial \hat{y}_n}{\partial w_i}
= -x_{n,i}
\]

So:

\[
\frac{\partial E}{\partial w_i} = -\frac{1}{N}\sum_n e_n x_{n,i}
\]

Gradient descent update:

\[
w_i \leftarrow w_i - \eta \frac{\partial E}{\partial w_i}
= w_i + \eta\left(\frac{1}{N}\sum_n e_n x_{n,i}\right)
\]

For online (stochastic) gradient descent, the per-sample Delta Rule becomes:

\[
\Delta w_i = \eta(d-\hat{y})x_i
\]

### 3.2 B(ii) Adaline implementation (NumPy only)

Our batch gradient descent Adaline is implemented in `src/adaline.py` as `AdalineGD`. A key difference from the Perceptron is that Adaline computes the gradient of MSE and performs a smooth update, enabling visualization of loss surfaces and gradient descent trajectories.

**Listing B1. Batch gradient descent update (from `src/adaline.py`).**

```python
y_hat = xb @ self.w
err = y - y_hat
grad = -(xb.T @ err) / n
self.w = self.w - self.lr * grad
mse = float((err**2).mean() / 2.0)
```

### 3.3 B(ii) Loss surface and gradient descent trajectory

To explicitly show the geometry of optimization, we train Adaline on a 2D task and compute the MSE surface over a grid of weight values. This produces a convex “bowl-shaped” objective typical of least-squares problems.

**Figure B1. Adaline MSE surface (2D toy dataset).**  
Insert: `figures/partB_adaline_mse_surface.png`

**Figure B2. Gradient descent trajectory on the MSE surface.**  
Insert: `figures/partB_adaline_gd_trajectory.png`

**Figure B3. MSE vs epoch (2D toy dataset).**  
Insert: `figures/partB_adaline_mse_toy.png`

### 3.4 B(iii) Learning rate analysis

We compare learning rates:

\[
\eta \in \{0.0001, 0.001, 0.01, 0.1\}
\]

**Figure B4. Adaline MSE convergence for multiple learning rates.**  
Insert: `figures/partB_adaline_lr_comparison.png`

**Observed behavior (Figure B4):**

- \(\eta=0.0001\): almost no improvement over 50 epochs (very slow).
- \(\eta=0.001\): stable but still slow.
- \(\eta=0.01\): clear improvement with decent speed.
- \(\eta=0.1\): fastest convergence in this specific setup, reaching the lowest MSE early.

**Interpretation:** In gradient descent, larger learning rates can accelerate convergence but may also risk divergence if the objective curvature is steep. On our toy setup (convex MSE with moderate curvature), \(\eta=0.1\) remained stable and performed best.

---

## 4. Part C — Kohonen Self-Organizing Maps (SOM)

### 4.1 C(i) Architecture and competitive learning

A SOM maps high-dimensional inputs into a 2D grid of neurons while preserving neighborhood relationships. Each neuron \(i\) has a weight vector:

\[
\mathbf{w}_i \in \mathbb{R}^{784}
\]

**Best Matching Unit (BMU):**

\[
\text{BMU} = \arg\min_i \lVert \mathbf{x} - \mathbf{w}_i \rVert
\]

SOM updates not only the BMU but also its neighbors according to a Gaussian neighborhood function:

\[
h_i(t) = \exp\left(-\frac{\lVert r_i - r_{\text{BMU}}\rVert^2}{2\sigma(t)^2}\right)
\]

with decays:

\[
\eta(t)=\eta_0 e^{-t/\lambda}, \quad \sigma(t)=\sigma_0 e^{-t/\lambda}
\]

and weight update:

\[
\mathbf{w}_i(t+1) = \mathbf{w}_i(t) + \eta(t)\, h_i(t)\, (\mathbf{x}-\mathbf{w}_i(t))
\]

### 4.2 C(ii) SOM training algorithm (NumPy only)

The SOM implementation is in `src/som.py`. Because MNIST is large, a naive full-grid update per sample is slow. Our implementation updates only a **local neighborhood window** around the BMU (approximately a \(3\sigma\) radius), which is a standard performance optimization that retains the learning behavior.

**Listing C1. SOM BMU selection and local neighborhood update (from `src/som.py`).**

```python
bmu_r, bmu_c = self.bmu(xi)
lr_t, sigma_t = self._decay(t, max_t)
rad = int(np.ceil(3.0 * sigma_t))
# ... slice local window ...
h = np.exp(-dist2 / (2.0 * (sigma_t**2 + 1e-12))).astype(np.float32)
self.weights[r0:r1+1, c0:c1+1] = w_local + lr_t * h[..., None] * (xi - w_local)
```

### 4.3 C(iii) Visualization and evaluation

After training, SOMs can be analyzed visually and quantitatively:

1. **U-matrix:** shows average distances between neighboring neurons (high values suggest cluster boundaries).
2. **Prototype maps:** reshape neuron weights into 28×28 images to see learned digit-like prototypes.
3. **Class hit maps:** count which digit labels (from MNIST) activate each neuron most strongly.

**Figure C1. U-matrix.**  
Insert: `figures/partC_som_umatrix.png`

**Figure C2. Prototype (weight) maps.**  
Insert: `figures/partC_som_weight_maps.png`

**Figure C3. Winner class hit map (dominant digit per neuron).**  
Insert: `figures/partC_som_class_hitmap_winner.png`

**Figure C4. Hit strength map.**  
Insert: `figures/partC_som_class_hitmap_strength.png`

#### Quality metrics: QE and TE

Two widely used SOM quality metrics are:

- **Quantization Error (QE):** average distance from each sample to its BMU’s weight.
- **Topographic Error (TE):** fraction of samples whose best and second-best BMUs are not adjacent (indicating topology violations).

From the main notebook run:

- **QE = 5.535064**
- **TE = 0.10775**

These values indicate that the SOM provides a meaningful quantization of the data distribution, while some topology violations occur (as expected for complex, high-dimensional digit manifolds).

### 4.4 C(iv) Anomaly detection using quantization error

The intuition is: if a digit is corrupted, ambiguous, or out-of-distribution, its distance to the closest SOM prototype (BMU) should be unusually high. We use per-sample QE:

\[
qe(\mathbf{x}) = \lVert \mathbf{x} - \mathbf{w}_{\text{BMU}} \rVert
\]

**Thresholding approach:** Let \(T\) be the 95th percentile of the “clean” QE distribution. Predict anomaly if \(qe(\mathbf{x}) > T\).

**Figure C5. QE histogram with threshold.**  
Insert: `figures/partC_som_anomaly_qe_hist.png`

From the notebook:

- **Threshold (95th percentile clean): 7.481676**
- **Precision: 0.829876**
- **Recall: ~1.0**
- **F1: 0.907029**

**Interpretation:** Recall near 1.0 means the method detects almost all anomalies, while precision under 1.0 indicates some clean samples are flagged as anomalies (false positives). In practice, threshold tuning allows choosing between higher precision (fewer false alarms) and higher recall (fewer misses).

---

## 5. Part D — Backpropagation Networks (MLP)

### 5.1 D(i) Backpropagation theory (3-layer derivation)

Consider a 3-layer MLP (input → hidden → output). For each layer \(l\):

\[
net^{(l)} = W^{(l)}o^{(l-1)} + b^{(l)}, \quad o^{(l)} = f(net^{(l)})
\]

The key backpropagation idea is to compute gradients via the chain rule using **error signals** (deltas).

#### Output-layer delta

For output neuron \(k\):

\[
\delta_k = \frac{\partial E}{\partial net_k}
= (d_k - y_k)\, f'(net_k)
\]

#### Hidden-layer delta

For hidden neuron \(j\):

\[
\delta_j = f'(net_j)\sum_k \delta_k w_{kj}
\]

#### Weight updates

If \(o_i\) is the activation from the previous layer entering weight \(w_{ij}\):

\[
\Delta w_{ij} = \eta\, \delta_j \, o_i
\]

**Practical simplification:** For **softmax + cross-entropy**, the gradient at logits simplifies to:

\[
\frac{\partial E}{\partial logits} = (p - y)
\]

This avoids explicitly multiplying by a softmax derivative and improves numerical stability.

### 5.2 D(ii) MLP implementation (NumPy only)

Our MLP is implemented in `src/mlp.py` with architecture:

**784 → 128 → 64 → 10**

Key implementation details:

- **Initialization:** He initialization for ReLU, Xavier-like scaling for sigmoid/tanh.
- **Optimizer:** mini-batch SGD + momentum.
- **Loss:** softmax cross-entropy.
- **Regularization options:** L2 weight decay and dropout.

**Listing D1. Core backprop updates for `W3` and propagation into hidden layers (from `src/mlp.py`).**

```python
dlogits = (cache["probs"] - yb_oh).astype(np.float32) / xb.shape[0]
grads["W3"] = cache["a2"].T @ dlogits + self.l2 * self.params["W3"]
grads["b3"] = dlogits.sum(axis=0, keepdims=True)
da2 = dlogits @ self.params["W3"].T
dz2 = da2 * self.act.df(cache["z2"]).astype(np.float32)
```

### 5.3 D(ii) Training curves, accuracy, and confusion matrix

**Figure D1. MLP training/validation loss curves.**  
Insert: `figures/partD_mlp_loss_curves.png`

**Figure D2. MLP training/validation accuracy curves.**  
Insert: `figures/partD_mlp_accuracy_curves.png`

**Figure D3. Confusion matrix on test set.**  
Insert: `figures/partD_confusion_matrix.png`

**Observed result (main notebook run):**

- **MLP test accuracy: 0.9714 (97.14%)**

This exceeds the ≥95% target and demonstrates the advantage of multi-layer non-linear representations over single-layer linear models.

### 5.4 D(iii) Activation function comparison (Sigmoid vs Tanh vs ReLU)

Activations affect gradient flow and convergence. We compare:

- **Sigmoid:** saturates for large \(|net|\), causing vanishing gradients; can also overflow in naive implementations of \(\exp(\cdot)\).
- **Tanh:** zero-centered but still saturates.
- **ReLU:** mitigates vanishing gradients for positive activations and typically converges faster, though it can “die” if many activations become negative.

**Figure D4. Activation curves and gradient magnitudes.**  
Insert: `figures/partD_activation_curves_and_gradients.png`

**Figure D5. Validation accuracy vs epoch across activations.**  
Insert: `figures/partD_activation_comparison_val_acc.png`

**Observed behavior (Figure D5):** ReLU achieved the highest and fastest validation accuracy, tanh was close, and sigmoid converged slowest with a lower final accuracy. This supports selecting ReLU as the default activation for MNIST-scale MLPs.

### 5.5 D(iv) Regularization (L2 + Dropout)

Regularization reduces overfitting by constraining model capacity:

- **L2 weight decay:** adds \(\frac{\lambda}{2}\lVert W\rVert_2^2\) to the loss, discouraging large weights.
- **Dropout:** randomly drops hidden units during training, reducing co-adaptation and acting like an implicit ensemble.

**Figure D6. Regularization effect on loss curves.**  
Insert: `figures/partD_regularization_loss.png`

**Interpretation:** Regularization typically increases training loss (harder to fit) but can reduce the generalization gap between train and validation curves. In our run, regularized curves are smoother and less over-optimized on the training set, though optimal tuning of \( \lambda \) and dropout rate is important because too much regularization can increase validation loss.

### 5.6 D(v) SOM integration (auxiliary features)

We integrate SOM-based topology information into the MLP by appending BMU coordinates as 2 auxiliary features:

1. Train SOM on MNIST pixels.
2. For each image, compute BMU \((r,c)\).
3. Normalize coordinates to \([0,1]\) and concatenate:

\[
\tilde{\mathbf{x}} = [\mathbf{x};\ r/H;\ c/W]
\]

**Figure D7. SOM integration comparison (validation accuracy).**  
Insert: `figures/partD_som_integration_val_acc.png`

From the notebook output:

- **Baseline final val acc:** 0.964833
- **SOM-aug final val acc:** 0.964333

**Interpretation:** SOM features slightly improved early-epoch validation accuracy in some epochs but did not increase final validation accuracy in this run. This suggests that simple BMU-coordinate features alone may be insufficient; richer SOM-derived features (e.g., one-hot BMU index, distances to multiple prototypes, or U-matrix-based local density measures) could be explored as future work.

---

## 6. Comparative Analysis (All four paradigms)

### 6.1 Summary table

| Paradigm | Supervision | Objective | Output non-linearity | Strengths | Limitations | Best use case |
|---|---|---|---|---|---|---|
| Perceptron | Supervised | Misclassification | Step | Very fast, interpretable baseline | Only linearly separable problems | Quick linear baseline (binary) |
| Adaline (Delta Rule) | Supervised | MSE (least squares) | Linear (train), threshold (infer) | Differentiable learning, convex objective | Still linear decision boundary | Bridge from perceptron to backprop |
| SOM (Kohonen) | Unsupervised | Topology-preserving quantization | Competitive | Clustering + visualization + anomaly detection | No direct class objective | Feature learning, cluster analysis |
| MLP (Backprop) | Supervised | Cross-entropy | Hidden activations + softmax | High accuracy, non-linear representations | More compute, tuning needed | Full 10-class digit recognition |

### 6.2 Complexity and representational power

- **Perceptron / Adaline:** Both are linear in parameters. Training is lightweight (\(O(Nd)\) per epoch) and memory requirements are small. However, their representational power is limited to linear boundaries.
- **SOM:** Training cost is higher because BMU search is \(O(Md)\) per sample (with \(M=H\times W\) neurons). Local neighborhood updates reduce constant factors but the method is still more expensive than a single linear model. In return, SOM provides topology-preserving prototypes that help interpret the data distribution.
- **MLP:** Training requires repeated forward and backward passes and more hyperparameter tuning. It is computationally heavier but provides non-linear decision boundaries that match the complexity of MNIST digit manifolds, leading to the best accuracy.

### 6.3 Suitability for MNIST recognition

MNIST is not linearly separable in raw pixel space for 10 classes. Therefore:

- Single-layer models (Perceptron/Adaline) are valuable for **learning theory** and **binary baselines** but cannot compete with multi-layer networks on 10-class recognition.
- SOM is ideal for **unsupervised structure discovery** and **anomaly detection**, and it can support feature engineering.
- MLP with backpropagation provides the best trade-off between accuracy and conceptual clarity when implementing “from scratch” without deep learning frameworks.

---

## 7. Conclusion

This project demonstrates a progressive path from biologically inspired, single-layer threshold models to modern multi-layer gradient-based learning. The Perceptron illustrates linear classification and the crucial concept of linear separability, while its XOR failure motivates the need for hidden layers. Adaline extends learning into differentiable optimization by minimizing MSE, making it a direct conceptual bridge to backpropagation. SOM offers a distinct unsupervised perspective, producing interpretable 2D maps of digit structure and enabling anomaly detection via quantization error (QE). Finally, the NumPy-only MLP trained by backpropagation achieves strong MNIST performance (97.14% test accuracy), validating the representational power of multi-layer networks. Overall, integrating these paradigms into a unified system clarifies their relative strengths: simplicity and interpretability (Perceptron/Adaline), topology-preserving clustering (SOM), and high-accuracy non-linear recognition (MLP).

---

## References (APA 7th style)

Kohonen, T. (2001). *Self-Organizing Maps* (3rd ed.). Springer.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278–2324.

Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.

Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review, 65*(6), 386–408.

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*(6088), 533–536.

Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits. In *1960 IRE WESCON Convention Record* (pp. 96–104).

---

## Appendix A — Where each requirement is satisfied (quick checklist)

- Perceptron biological mapping + diagram: Section 2.1, Figure A1.
- Perceptron implementation (NumPy only) + update rule + boundary plots: Sections 2.2–2.4, Listing A1, Figures A2–A8.
- Convergence + XOR failure analysis: Section 2.5, Figures A9–A10.
- Delta Rule derivation from MSE: Section 3.1.
- Adaline implementation + MSE surface + GD trajectory + learning rates: Sections 3.2–3.4, Listing B1, Figures B1–B4.
- SOM architecture + BMU + training + decay: Sections 4.1–4.2, Listing C1.
- SOM visualization + QE/TE: Section 4.3, Figures C1–C4, metrics reported.
- SOM anomaly detection + threshold + precision/recall: Section 4.4, Figure C5, metrics reported.
- Backprop derivation + MLP implementation (NumPy only): Sections 5.1–5.2, Listing D1.
- MLP training curves + confusion matrix + test accuracy: Section 5.3, Figures D1–D3.
- Activation comparison + curves: Section 5.4, Figures D4–D5.
- Regularization analysis: Section 5.5, Figure D6.
- SOM integration evaluation: Section 5.6, Figure D7, metrics reported.
