# Assignment 03 (Colab)

Open `Assignment_03_Image_Classification_ResNet50_vs_InceptionV3_CIFAR10.ipynb` in Google Colab and run top-to-bottom.

## Recommended Colab settings
- Runtime: GPU
- Memory: High-RAM (optional)

## What the notebook produces
- ResNet50 and InceptionV3 transfer learning on CIFAR-10
- Accuracy/Loss vs epoch plots
- Confusion matrix + classification report (per model)
- Comparison table: accuracy, loss, training time, model size, params

## Notes
- InceptionV3 uses 299×299 input, so it may take longer than ResNet50 (224×224).
- Epoch counts are set to 8 (head) + 7 (fine-tune). You can adjust in the notebook if needed.
