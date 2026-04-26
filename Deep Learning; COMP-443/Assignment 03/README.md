# Assignment 03 (Colab) — ResNet50 vs InceptionV3 (CIFAR-10)

Open `Assignment_03_Image_Classification_ResNet50_vs_InceptionV3_CIFAR10.ipynb` in Google Colab and run top-to-bottom.

## Recommended Colab settings
- Runtime: GPU
- Memory: High-RAM (optional)

## What the notebook does
- Loads CIFAR-10
- Builds `tf.data` pipelines (resize + preprocess + augmentation)
- Trains transfer-learning models:
  - ResNet50 (224×224)
  - InceptionV3 (299×299)
- Uses 2-stage training:
  - head training (backbone frozen)
  - fine-tuning (unfreeze last N layers, smaller LR)

## Folder structure (organized)
- `report.html`: IEEE-style one-column report
- `report.md`: short markdown summary
- `figures/`: curves + confusion matrices
- `results/`: metrics JSON/CSV, histories, classification reports
- `models/`: saved `.keras` models
- `reports/`: archived old report files

## Notes
- CIFAR-10 images are 32×32; resizing can blur fine details (often affects animal classes).
- InceptionV3 uses 299×299 input, which typically increases compute.
