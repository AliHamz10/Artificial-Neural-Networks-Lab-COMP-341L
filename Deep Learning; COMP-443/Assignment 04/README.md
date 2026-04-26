# Assignment 04 (Colab)

Open `notebooks/Assignment_04_Lightweight_CNNs_MobileNetV2_DenseNet121_EfficientNetB0_CIFAR10.ipynb` in Google Colab and run top-to-bottom.

## Drive saving
The notebook mounts Google Drive and saves all artifacts to:
- `MyDrive/COMP-443/Assignment_04_Efficient_CNNs` (default)

## Streamlit deployment
`streamlit_app/` contains a small Streamlit app that loads the trained `.keras` models and runs CIFAR-10 inference on an uploaded image.

1) Train in Colab and download the exported models from Drive (`OUTPUT_DIR`):
- `mobilenetv2_cifar10_transfer.keras`
- `densenet121_cifar10_transfer.keras`
- `efficientnetb0_cifar10_transfer.keras`

2) Place them in:
- `models/`

3) Run:
```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

## Directory layout
- `notebooks/` - training and evaluation notebook
- `docs/` - assignment handout PDF
- `reports/` - generated report files (`report.md`, `report.html`)
- `models/` - exported trained models (`*.keras`)
- `results/plots/` - comparison and training curve plots (`*.png`)
- `results/tables/` - comparison tables (`comparison_table.csv`)

## What you get
- Transfer learning on CIFAR-10 using:
  - MobileNetV2
  - DenseNet121
  - EfficientNetB0
- Fair comparison (same training settings, same input size)
- Comparison table: accuracy, loss, params, model size, training time, inference speed
- CPU-only inference test for MobileNetV2
- Saved graphs + HTML report in Drive, then organized in this folder structure

## Speed tips
- If slow, reduce `BATCH_SIZE` to 32 and/or reduce `EPOCHS_HEAD` / `EPOCHS_FT`.
