# Lab 10 - Zarmeena Jawad

## Included Files
- `lab10_lstm_sentiment_colab.ipynb`: Google Colab notebook for the complete Lab 10 workflow
- `build_lab10_lstm_notebook.py`: script used to generate the notebook file
- `Lab_Report_10.md`: Markdown version of the lab report
- `Lab_Report_10.html`: formatted academic HTML version of the report

## Dataset
This lab uses the IMDb movie review dataset. The notebook is configured to download the dataset automatically from Hugging Face if the CSV file is not already available in the lab folder.

Saved CSV file:
- `IMDB Dataset.csv`

Online source used in the notebook:
- Hugging Face dataset: `scikit-learn/imdb`

## How To Run In Colab
1. Open `lab10_lstm_sentiment_colab.ipynb` in Google Colab.
2. Keep Google Drive enabled in the setup cell.
3. Run the notebook cells in sequence.
4. Outputs will be written to:
   - `/content/drive/MyDrive/COMP-341L/Lab 10/Zarmeena Jawad's Lab`
5. If the IMDb CSV is missing, the notebook will fetch it automatically.
6. The notebook stores figures inside `plots/`.
7. It also exports:
   - `Lab_Report_10.md`
   - `Lab_Report_10.html`

## What The Notebook Covers
- text cleaning and review normalization
- tokenization and sequence padding
- baseline LSTM model training
- modified LSTM model with parameter changes
- validation and testing analysis
- custom review prediction
- report generation and saved plots
