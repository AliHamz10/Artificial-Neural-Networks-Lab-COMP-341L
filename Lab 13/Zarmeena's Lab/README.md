# Lab 13 - Zarmeena Jawad

## Files
- `lab13_llm_finetuning_ta_style_colab.ipynb`: Google Colab notebook for Lab 13 (LLM fine-tuning + LoRA)
- `build_lab13_llm_finetune_notebook.py`: script used to generate the notebook

## Highlights (distinct setup)
- Uses `gpt2` (small, Colab-friendly) + **LoRA (PEFT)**
- Uses a different prompt template:
  - `<INSTRUCTION> ... </INSTRUCTION>` and `<RESPONSE> ... </RESPONSE>`
- Report export uses **A4** layout and **black code blocks with white text**

## How to run in Colab
1. Open `lab13_llm_finetuning_ta_style_colab.ipynb` in Colab.
2. Run the setup cell (mounts Drive).
3. Edit the dataset in Drive (manual work required):
   - `/content/drive/MyDrive/COMP-341L/Lab 13/Zarmeena's Lab/data/academic_assistant.jsonl`
4. Run training + evaluation cells.

## Outputs (Saved to Google Drive)
Saved to:
- `/content/drive/MyDrive/COMP-341L/Lab 13/Zarmeena's Lab`

Includes:
- `data/academic_assistant.jsonl`
- `outputs/before_after_dp.txt`
- `outputs/failure_cases.txt`
- `Lab_Report_13.md`, `Lab_Report_13.html`

