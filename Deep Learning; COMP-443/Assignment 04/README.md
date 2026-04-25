# Assignment 04 (Colab)

Open `Assignment_04_Lightweight_CNNs_MobileNetV2_DenseNet121_EfficientNetB0_CIFAR10.ipynb` in Google Colab and run top-to-bottom.

## Drive saving
The notebook mounts Google Drive and saves all artifacts to:
- `MyDrive/COMP-443/Assignment_04_Efficient_CNNs` (default)

## What you get
- Transfer learning on CIFAR-10 using:
  - MobileNetV2
  - DenseNet121
  - EfficientNetB0
- Fair comparison (same training settings, same input size)
- Comparison table: accuracy, loss, params, model size, training time, inference speed
- CPU-only inference test for MobileNetV2
- Saved graphs + HTML report (`report.html`) in Drive

## Speed tips
- If slow, reduce `BATCH_SIZE` to 32 and/or reduce `EPOCHS_HEAD` / `EPOCHS_FT`.
