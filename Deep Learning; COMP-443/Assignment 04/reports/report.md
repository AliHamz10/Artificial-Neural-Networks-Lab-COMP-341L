# Assignment 04 Report - Efficient CNN Architectures

## Models

- MobileNetV2
- DenseNet121
- EfficientNetB0

## Metrics Summary

- Best accuracy: EfficientNetB0 (0.9488)
- Best loss: EfficientNetB0 (0.1618)
- Fastest GPU inference: MobileNetV2 (0.021823 sec/batch)
- CPU-only benchmark (required): MobileNetV2 (0.285467 sec/batch)
- Smallest model size: MobileNetV2 (22.1755 MB)

## Memory Evidence

- Notebook global RSS (benchmark stage):
  - Before: 114416.1367 MB
  - After: 114495.9375 MB
  - Delta: 79.8008 MB
- Note: RSS values are session-level benchmark memory readings, not isolated per-model runtime memory.

## Key Files

- Table: `../results/tables/comparison_table.csv`
- Final report: `report.html`
- Figures:
  - `../results/plots/bar_accuracy.png`
  - `../results/plots/bar_params.png`
  - `../results/plots/bar_gpu_speed.png`
  - `../results/plots/bar_model_size.png`
  - `../results/plots/mobilenetv2_curves.png`
  - `../results/plots/densenet121_curves.png`
  - `../results/plots/efficientnetb0_curves.png`

## Deployment Recommendations

- Mobile phone: MobileNetV2
- Cloud server: EfficientNetB0
- Real-time system: MobileNetV2 (EfficientNetB0 when accuracy is prioritized)
  