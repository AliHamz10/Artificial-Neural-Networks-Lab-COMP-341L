# Lab 13 - Ali Hamza

## Files
- `lab13_llm_finetuning_academic_assistant_colab.ipynb`: main Google Colab notebook for Lab 13 (LLM fine-tuning + LoRA)
- `build_lab13_llm_finetune_notebook.py`: script used to generate the notebook

## What this lab does
- Builds a small instruction dataset in JSONL (manual rewriting required)
- Fine-tunes a small GPT-style model (`distilgpt2`) using **LoRA (PEFT)**
- Compares **base vs fine-tuned** outputs on the same prompt
- Saves evaluation outputs + failure cases to Drive
- Exports `Lab_Report_13.md` and `Lab_Report_13.html`

## How to run in Colab
1. Open `lab13_llm_finetuning_academic_assistant_colab.ipynb` in Google Colab.
2. Run the first cell to mount Google Drive.
3. Edit the dataset file created in Drive:
   - `/content/drive/MyDrive/COMP-341L/Lab 13/Ali Hamza's Lab/data/academic_assistant_instructions.jsonl`
4. Run all cells in order.

## Outputs (Saved to Google Drive)
Saved to:
- `/content/drive/MyDrive/COMP-341L/Lab 13/Ali Hamza's Lab`

Includes:
- `data/academic_assistant_instructions.jsonl`
- `outputs/before_after_dynamic_programming.txt`
- `outputs/failure_cases.txt`
- `Lab_Report_13.md`, `Lab_Report_13.html`

