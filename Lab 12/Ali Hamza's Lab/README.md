# Lab 12 - Ali Hamza

## Files
- `lab12_gan_mnist_colab.ipynb`: main Google Colab notebook for Lab 12 (GAN on MNIST)
- `build_lab12_gan_mnist_notebook.py`: notebook generator used to create the `.ipynb`

## Dataset (Kaggle)
This lab is designed to use:
- `oddrationale/mnist-in-csv` (MNIST in CSV)

### Kaggle token required (Colab)
To download from Kaggle in Colab, upload your `kaggle.json` file to the runtime (Files panel).

### Fallback dataset
If `kaggle.json` is not provided, the notebook automatically falls back to `torchvision.datasets.MNIST` so you can still complete the lab end-to-end.

## How to run in Colab
1. Open `lab12_gan_mnist_colab.ipynb` in Google Colab.
2. Run the first setup cell (Google Drive mount).
3. (Optional) Upload `kaggle.json` to use the Kaggle MNIST-from-CSV dataset.
4. Run all cells in order.

## Outputs (Saved to Google Drive)
This notebook saves everything to:
- `/content/drive/MyDrive/COMP-341L/Lab 12/Ali Hamza's Lab`

It creates:
- `plots/` (real samples, loss curves, real vs fake)
- `samples/` (generated digits per epoch)
- `Lab_Report_12.md` and `Lab_Report_12.html`

