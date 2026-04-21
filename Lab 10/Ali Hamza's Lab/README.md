# Lab 10 - Ali Hamza

## Files
- `lab10_lstm_sentiment_colab.ipynb`: main Google Colab notebook for Lab 10
- `build_lab10_lstm_notebook.py`: notebook generator used to create the `.ipynb`

## Dataset
The notebook now downloads the IMDb dataset automatically from Hugging Face and saves a CSV copy in your lab folder on Google Drive.

Saved file name:
- `IMDB Dataset.csv`

Online source used by the notebook:
- Hugging Face dataset: `scikit-learn/imdb`

## How to run in Colab
1. Open `lab10_lstm_sentiment_colab.ipynb` in Google Colab.
2. Keep Google Drive enabled in the first setup cell.
3. Run all cells in order.
4. The notebook will mount Google Drive and save everything to:
   - `/content/drive/MyDrive/COMP-341L/Lab 10/Ali Hamza's Lab`
5. If the IMDb CSV is missing, the notebook will download it online automatically.
6. The notebook will save plots in `plots/`.
7. At the end it will generate:
- `Lab_Report_10.md`
- `Lab_Report_10.html`

## Important setup note
- The notebook now uses Google Drive by default.
- Default setting:
  - `USE_GOOGLE_DRIVE = True`
- If Drive mount fails temporarily, rerun the first cell after refreshing the notebook tab.

## What is included
- CSV loading and cleaning
- tokenization and padding
- original LSTM model
- modified LSTM model with two changes
- training and validation plots
- model comparison table
- custom prediction test
- reflection question answers
- exportable report files
