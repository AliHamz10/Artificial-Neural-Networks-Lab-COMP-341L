# Lab 11 - Ali Hamza

## Files
- `lab11_lstm_timeseries_forecasting_colab.ipynb`: main Google Colab notebook for Lab 11
- `build_lab11_lstm_timeseries_notebook.py`: notebook generator used to create the `.ipynb`

## Dataset
This lab is written to work with **any univariate time series CSV** (retail sales / demand / etc.).

Default expected CSV (you can change this inside the notebook):
- `Sales Forecasting Dataset.csv`

Recommended columns (you can also change these inside the notebook):
- date column: `Date`
- value column: `Sales`

### Fallback dataset (if your CSV is missing)
If the dataset CSV is not found in your lab folder, the notebook automatically downloads a small public time-series CSV (monthly sales / passengers) so you can still run the full pipeline end-to-end.

## How to run in Colab
1. Open `lab11_lstm_timeseries_forecasting_colab.ipynb` in Google Colab.
2. Keep Google Drive enabled in the first setup cell.
3. Run all cells in order.
4. Outputs will be saved to:
   - `/content/drive/MyDrive/COMP-341L/Lab 11/Ali Hamza's Lab`
5. Plots are saved into `plots/`.
6. At the end it exports:
   - `Lab_Report_11.md`
   - `Lab_Report_11.html`

## Google Drive note
This notebook is configured to save **everything** (dataset copy, plots, and reports) into the Google Drive lab folder. If the Drive mount fails, rerun the first setup cell.
