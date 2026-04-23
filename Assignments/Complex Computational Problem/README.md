# CCP — Artificial Neural Network (MNIST)

This folder contains a **NumPy-from-scratch** implementation of four neural learning paradigms on MNIST:

- Part A: Perceptron (binary classification baseline)
- Part B: Adaline (Delta rule / Widrow–Hoff learning)
- Part C: Kohonen Self-Organizing Map (SOM) + anomaly detection
- Part D: Backpropagation MLP (784→128→64→10) + activation/regularization + SOM integration

## Contents

- `ccp_ann_project.ipynb` — main Colab-ready notebook (run top→bottom)
- `src/` — all model implementations (NumPy only)
- `figures/` — outputs saved by the notebook (use these in the report)
- `report.md` — report template (Markdown)

## Google Colab usage

1. Upload `Assignments/Complex Computational Problem/` to Colab (or copy it to Google Drive and mount Drive).
2. Open `ccp_ann_project.ipynb`.
3. Run cells top→bottom.

Notes:

- The notebook uses TensorFlow/Keras **only to load MNIST**. All models are implemented in `src/` using NumPy only.
- Set `FAST_RUN = True` near the top for a quick demo run.
- The first setup cell is **self-contained**: if the runtime cannot see `src/` (common with Cursor “Colab server” extensions that only sync the notebook), it will recreate `src/` from embedded sources automatically.
