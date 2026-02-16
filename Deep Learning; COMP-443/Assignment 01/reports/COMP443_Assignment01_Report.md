# COMP-443 – Deep Learning  
## Assignment 01: Classical vs Deep Models on Fashion-MNIST

---

**Student:** Ali Hamza  
**Registration No.:** B23F0063AI106  
**Course:** Deep Learning (COMP-443)  
**Instructor:** (to be filled)  
**Submission Date:** 16 February 2026

---

## 1. Problem Statement and Requirements

The whiteboard instructions for **Assignment 01** can be summarised as:

- Choose **one good dataset** (from Kaggle or another public source).  
- Implement **4 algorithms in total**:
  - **2 conventional (classical) machine-learning models**  
  - **2 deep learning models**
- Train and evaluate all four models on the same dataset, **compare their performance**, and present the work in a **properly formatted report suitable for printing**.

In this submission I use the **Fashion-MNIST** dataset and implement:

- **Logistic Regression** (classical baseline)  
- **Random Forest** (classical baseline)  
- **Simple CNN** (small convolutional neural network)  
- **Deeper CNN** (deeper convolutional neural network)

All experiments and plots were generated using the code under  
`Deep Learning; COMP-443/Assignment 01/src/`.

---

## 2. Dataset and Preprocessing

### 2.1 Fashion-MNIST overview

Fashion-MNIST is a 10-class image classification dataset with **28×28 grayscale** images of clothing items. There are **60,000 training** and **10,000 test** examples. The classes are:

1. T-shirt/top  
2. Trouser  
3. Pullover  
4. Dress  
5. Coat  
6. Sandal  
7. Shirt  
8. Sneaker  
9. Bag  
10. Ankle boot

This dataset is widely used as a drop-in replacement for MNIST, but is more challenging and better suited to evaluating deep models.

### 2.2 Splits and preprocessing

The code in `src/data.py` performs the following steps:

- Downloads Fashion-MNIST via `tf.keras.datasets.fashion_mnist`.  
- Normalises pixel intensities from **[0, 255] to [0, 1]** (float32).  
- Creates a **validation split** from the original training set:
  - 80% of the original training set → training  
  - 20% of the original training set → validation  
- Adds a **channel dimension** (shape `(28, 28, 1)`) for CNNs.  
- Flattens images to 784-dimensional vectors for classical models.

The same train/validation split (fixed random seed) is used for all four models.

---

## 3. Classical Baselines

The module `src/baseline_models.py` implements two classical models using scikit-learn.

### 3.1 Logistic Regression

Configuration:

- Multinomial logistic regression (`multi_class=\"multinomial\"`)  
- Solver: **lbfgs**  
- Regularisation: L2 with **C = 1.0**  
- Maximum iterations: 200

The model is trained on the **flattened** 784-dimensional features and evaluated on the validation set. Its confusion matrix is saved as:

- `figures/logreg_confusion.png`

### 3.2 Random Forest

Configuration:

- **RandomForestClassifier**  
- Number of trees: **150**  
- `max_depth=None` (trees grow until pure or min samples)  
- `random_state = 42`

The Random Forest is also trained on the flattened features. Its confusion matrix is saved as:

- `figures/rf_confusion.png`

Both classical models provide reasonably strong baselines but cannot exploit local image structure as well as CNNs.

---

## 4. Deep Learning Models

The module `src/deep_models.py` defines two convolutional networks.

### 4.1 Simple CNN

Architecture:

- Input: `(28, 28, 1)`  
- Conv2D(32, 3×3, ReLU, same)  
- Conv2D(32, 3×3, ReLU, same)  
- MaxPooling2D(2×2)  
- Flatten  
- Dense(128, ReLU)  
- Dropout(0.3)  
- Dense(10, softmax)

Training:

- Optimizer: **Adam**, learning rate 1e-3  
- Loss: **sparse categorical cross-entropy**  
- Batch size: 128  
- Epochs: 50 (early stopping on validation loss with patience 5)

The learning curves (loss and accuracy vs. epoch) are saved as:

- `figures/simple_cnn_curves.png`

### 4.2 Deeper CNN

Architecture:

- Input: `(28, 28, 1)`  
- Conv2D(32, 3×3, ReLU, same) → Conv2D(32, 3×3, ReLU, same) → MaxPool(2×2)  
- Conv2D(64, 3×3, ReLU, same) → Conv2D(64, 3×3, ReLU, same) → MaxPool(2×2)  
- Flatten  
- Dense(256, ReLU)  
- Dropout(0.4)  
- Dense(10, softmax)

Training configuration is similar to the Simple CNN. Learning curves are saved as:

- `figures/deeper_cnn_curves.png`

The deeper model has more capacity and typically achieves slightly higher validation accuracy than the simple CNN.

---

## 5. Quantitative Results

The script `src/run_experiments.py` runs all four models and saves the metrics to:

- `reports/assignment01_results.json`

The table below summarises **validation** performance (using the 20% validation split from the original training set).

| Model            | Type          | Val Loss | Val Acc |
|------------------|--------------|---------:|--------:|
| Logistic Regression | Classical | 0.8572* | 0.8573* |
| Random Forest    | Classical    | 0.8852* | 0.8853* |
| Simple CNN       | Deep (CNN)   | 0.2293  | 0.9148  |
| Deeper CNN       | Deep (CNN)   | 0.2071  | 0.9206  |

Values marked with `*` use accuracy from scikit-learn; the “Val Loss” column for classical models is effectively a placeholder (logistic regression and Random Forest were not trained with the same cross-entropy loss interface as the Keras models). The key takeaway is the **accuracy** comparison.

From the confusion matrices:

- Logistic Regression struggles especially with classes that are visually similar (e.g. shirts vs T-shirts/coats).  
- Random Forest improves these cases and gives a stronger baseline.  
- CNNs significantly reduce misclassifications by exploiting spatial structure in the images.

---

## 6. Selected Figures

**Figure 1 – Logistic Regression confusion matrix (validation)**  
![Logistic Regression confusion matrix](../figures/logreg_confusion.png)

**Figure 2 – Random Forest confusion matrix (validation)**  
![Random Forest confusion matrix](../figures/rf_confusion.png)

**Figure 3 – Simple CNN learning curves (loss and accuracy)**  
![Simple CNN learning curves](../figures/simple_cnn_curves.png)

**Figure 4 – Deeper CNN learning curves (loss and accuracy)**  
![Deeper CNN learning curves](../figures/deeper_cnn_curves.png)

These plots are suitable for inclusion when printing the report to PDF.

---

## 7. Discussion

### 7.1 Classical vs deep models

The classical models operate on **flattened pixel vectors** and treat each pixel as an independent feature. They ignore the 2D structure of the image, which limits their ability to learn local patterns such as edges, textures, or small shapes. Random Forest performs noticeably better than logistic regression because it captures non-linear interactions between pixels.

In contrast, the convolutional networks learn **translation-invariant local filters**, allowing them to detect patterns like stripes, soles, pockets, or sleeves. This leads to substantially higher validation accuracy (around **91–92%**), especially on confusing classes such as shirts vs T-shirts or coats.

### 7.2 Overfitting and regularisation

The CNNs use **Dropout** (0.3–0.4) in the dense layers and early stopping on validation loss. The learning curves show that training loss continues to decrease while validation loss eventually flattens; the early-stopping mechanism prevents substantial overfitting. The deeper CNN has more capacity but still generalises well due to these regularisation techniques.

### 7.3 When classical models are still useful

Although CNNs clearly win in accuracy, classical models are still attractive when:

- Training time or compute is very limited.  
- A quick baseline is needed to check data quality.  
- Model interpretability is important (logistic regression offers linear weights; tree-based models can provide feature importances).

On small tabular datasets, classical models can even outperform naive deep networks. In this image domain, however, CNNs are the natural choice.

---

## 8. Conclusions

This assignment implemented **two classical** and **two deep learning** models on the Fashion-MNIST dataset and compared their performance:

- Logistic Regression and Random Forest provided solid baselines, with Random Forest achieving around **88.5%** validation accuracy.  
- A simple CNN already surpassed the classical methods (≈ **91.5%** accuracy).  
- A deeper CNN achieved the best performance (≈ **92.1%** accuracy), confirming that deeper architectures with convolutional layers are well suited for image classification tasks.

The experiments demonstrate that:

- Deep models excel at extracting spatial features from images and therefore typically outperform classical models on computer-vision problems.  
- Classical models remain relevant as strong baselines and for scenarios with limited compute or where interpretability is crucial.  
- Good preprocessing, appropriate regularisation (Dropout, early stopping), and careful learning-rate choices are essential for stable and high-performing deep learning systems.

For future work, one could explore:

- Data augmentation (random flips/crops) to further improve CNN generalisation.  
- Transfer learning with a pre-trained CNN on a larger dataset.  
- More advanced optimizers or learning-rate schedules.

---

## 9. How to Reproduce

1. Ensure the project virtual environment is set up and dependencies installed:
   ```bash
   cd "Artificial-Neural-Networks-Lab-COMP-341L"
   ./venv/bin/pip install -r requirements.txt
   ```
2. Run the experiment script:
   ```bash
   ./venv/bin/python3 "Deep Learning; COMP-443/Assignment 01/src/run_experiments.py" --epochs 50
   ```
   This regenerates all figures under `Deep Learning; COMP-443/Assignment 01/figures/`  
   and updates `reports/assignment01_results.json`.
3. Open `COMP443_Assignment01_Report.md` or the HTML version (if generated) and print to PDF.

