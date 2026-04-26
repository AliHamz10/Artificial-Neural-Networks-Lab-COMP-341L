# Lab 12 - Zarmeena Jawad

## Files
- `lab12_gan_mnist_colab.ipynb`: Google Colab notebook (Vanilla GAN / MLP) for MNIST digit generation
- `build_lab12_gan_mnist_notebook.py`: script used to generate the notebook

## Dataset
Primary: Kaggle **MNIST in CSV** dataset:
- `oddrationale/mnist-in-csv`

### Kaggle token (Colab)
If you want the Kaggle CSV version, upload `kaggle.json` in Colab (Files panel).  
If you don't upload it, the notebook automatically uses `torchvision.datasets.MNIST` so the lab still completes end-to-end.

## How to run in Colab
1. Open `lab12_gan_mnist_colab.ipynb` in Colab.
2. Run the setup cell (mounts Drive and sets output folders).
3. (Optional) Upload `kaggle.json` for Kaggle download.
4. Run all cells in order.

## Outputs (Saved to Drive)
Saved under:
- `/content/drive/MyDrive/COMP-341L/Lab 12/Zarmeena's Lab`

The notebook writes everything inside:
- `outputs/plots/` (loss plot + comparisons)
- `outputs/samples/` (generated digits each epoch)
- `Lab_Report_12.md` and `Lab_Report_12.html`

