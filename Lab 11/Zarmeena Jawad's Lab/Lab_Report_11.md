# Lab 11: LSTM for Time Series Forecasting (Regression)

**Student:** Zarmeena Jawad  
**Roll Number:** B23F0115AI125  
**Section:** B.S AI - Red  
**Date:** April 23, 2026

## Dataset
- Detected date column: `Month`
- Detected value column: `Sales`
- Samples (after cleaning): `108`

## Baseline vs Modified (final losses)
| Model    |   Window |   LSTM Units |   Dropout |   Train Loss (final) |   Val Loss (final) |
|:---------|---------:|-------------:|----------:|---------------------:|-------------------:|
| Base     |       12 |           50 |         0 |            0.050946  |          0.0529487 |
| Modified |       24 |          100 |         0 |            0.0557468 |          0.053419  |

## Plots
- `plots/task1_time_series.png`
- `plots/task1_trend_rolling_mean.png`
- `plots/task3_baseline_loss.png`
- `plots/task3_baseline_actual_vs_pred.png`
- `plots/task4_modified_loss.png`
- `plots/task4_modified_actual_vs_pred.png`
- `plots/task5_future_forecast.png`
