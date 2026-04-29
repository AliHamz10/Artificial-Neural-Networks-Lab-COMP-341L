# Abdul Basit — Assignment 3
## Image Classification (Transfer Learning): ResNet50 vs InceptionV3

**Notebook:** `notebooks/assignment3_resnet50_vs_inceptionv3_cifar10.ipynb`

### Run (recommended in Google Colab + Google Drive outputs)
- Upload this folder to Google Drive, open the notebook in Colab, and run cells top-to-bottom.
- Set runtime to GPU: Runtime → Change runtime type → GPU.
- The notebook mounts Drive and saves everything under `MyDrive/Abdul_Basit_Assignment_03/` (you can rename by editing `PROJECT_ROOT` in the first code cell).

### Outputs (auto-saved by the notebook)
- Curves: `figures/*.png`
- Confusion matrices: `figures/confusion_matrix_*.png` and `results/confusion_matrix_*.npy`
- Per-stage histories: `results/history_*_head.csv`, `results/history_*_finetune.csv`
- Comparison table: `results/summary_table.csv` and `results/metrics.json`
- Saved models: `models/*_cifar10_transfer.keras`
- Report starter: `report/report.md`
