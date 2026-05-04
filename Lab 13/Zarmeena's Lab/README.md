# Lab 13 - Zarmeena

## Included files
- `lab13_llm_finetune_academic_assistant_colab.ipynb`: Google Colab notebook for Lab 13
- `build_lab13_llm_finetune_notebook.py`: script used to generate the notebook
- `data/dataset_cs_instructions_v1.jsonl`: manually curated instruction dataset (v1)
- `data/dataset_cs_instructions_v2.jsonl`: improved dataset after failure analysis (v2)

## What this lab does
You will fine-tune a **small LLM (DistilGPT2)** to behave like a domain-specific academic assistant for Computer Science concepts.

The notebook includes:
- Manual dataset loading (instruction → response)
- Teaching style definition and enforcement
- LoRA fine-tuning (PEFT)
- Evaluation: base vs fine-tuned on the same prompt (`Explain dynamic programming`)
- Failure analysis (2 cases) + 1 manual fix + re-evaluation

## How to run in Colab
1. Open `lab13_llm_finetune_academic_assistant_colab.ipynb` in Google Colab.
2. Keep Google Drive enabled in the first setup cell.
3. Run all cells in order.
4. Outputs will be written to:
   - `/content/drive/MyDrive/COMP-341L/Lab 13/Zarmeena's Lab`
5. At the end it exports:
   - `Lab_Report_13.md`
   - `Lab_Report_13.html`

## Google Drive note
This notebook is configured to save **everything** (datasets copy, model adapters, outputs, and reports) into the Google Drive lab folder.

