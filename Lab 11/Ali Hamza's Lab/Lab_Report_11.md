# Lab 11: LSTM for Time Series Forecasting (Regression)

**Course:** COMP-341L - Artificial Neural Networks Lab  
**Student:** Ali Hamza  
**Roll Number:** B23F0063AI106  
**Section:** B.S AI - Red  
**Date:** April 23, 2026

## Task 1: Data Understanding
- Detected date column: `Month`
- Detected value column: `Sales`
- Total samples (after cleaning): `108`

Saved plots:
- `plots/task1_time_series.png`
- `plots/task1_trend_rolling_mean.png`
- `plots/task1_seasonality_month_avg.png` (if datetime detected)

## Task 2: Preprocessing
- Normalization: `MinMaxScaler` (fit on training split only)
- Baseline window size: `12`

## Task 3: Baseline LSTM
- Units: `50`
- Dropout: `0.0`
- Epochs: `20`

Loss curve:
- `plots/task3_baseline_loss.png`

Actual vs Predicted:
- `plots/task3_baseline_actual_vs_pred.png`

## Task 4: Modified LSTM (ANY TWO changes)
Modified parameters:
- Window: `24`
- Units: `100`
- Dropout: `0.0`

Loss curve:
- `plots/task4_modified_loss.png`
Actual vs Predicted:
- `plots/task4_modified_actual_vs_pred.png`

### Loss Comparison (final epoch)
| Model    |   Window |   LSTM Units |   Dropout |   Train Loss (final) |   Val Loss (final) |
|:---------|---------:|-------------:|----------:|---------------------:|-------------------:|
| Base     |       12 |           50 |         0 |            0.050946  |          0.0529487 |
| Modified |       24 |          100 |         0 |            0.0557468 |          0.053419  |

## Task 5: Future Forecast
- Forecast steps: `12`
- Plot: `plots/task5_future_forecast.png`

## Notes
LSTM learns temporal dependencies by maintaining an internal memory cell and gated updates, enabling it to model trend + seasonality patterns from past windows.
