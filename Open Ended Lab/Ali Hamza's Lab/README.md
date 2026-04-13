# Open Ended Lab Files

This folder now contains a **Google Colab-ready notebook** for the open-ended lab:

- `open_ended_lab_waste_classification_colab.ipynb`
- `build_open_ended_lab_notebook.py`

## What the notebook does

1. Loads a **CIFAR-10 subset** as allowed by the manual
2. Trains the baseline CNN for 10 epochs
3. Diagnoses overfitting using training vs validation accuracy
4. Modifies at least two parameters and compares performance
5. Replaces the CNN with **MobileNetV2**
6. Fine-tunes the last 10 layers with a smaller learning rate
7. Saves plots, metrics, Markdown report, and HTML report

## How to use in Google Colab

1. Open Google Colab
2. Upload `open_ended_lab_waste_classification_colab.ipynb`
3. Run all cells in order
4. The notebook will save outputs to:
   `/content/drive/MyDrive/COMP-341L/Open Ended Lab/Ali Hamza's Lab`
5. Export `Lab_Report_OpenEnded.html` to PDF if needed
