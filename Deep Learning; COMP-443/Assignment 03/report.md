# Assignment 03 Report — ResNet50 vs InceptionV3 (CIFAR-10)

**Course:** COMP-443 (Deep Learning)  
**Student:** Ali Hamza  
**Date:** April 27, 2026

## Folder structure
- `report.html` (IEEE one-column report)
- `figures/` (curves + confusion matrices)
- `results/` (metrics, histories, classification reports)
- `models/` (saved `.keras` models)
- `reports/` (archived old reports)

## Key results (test set)
From `results/summary_table.csv`:
- ResNet50: accuracy 0.9398, loss 0.1936
- InceptionV3: accuracy 0.8988, loss 0.3015

## What to review
- Curves: `figures/resnet50_accuracy_loss.png`, `figures/inceptionv3_accuracy_loss.png`
- Confusion matrices: `figures/confusion_matrix_resnet50.png`, `figures/confusion_matrix_inceptionv3.png`
- Classification reports: `results/classification_report_resnet50.txt`, `results/classification_report_inceptionv3.txt`
