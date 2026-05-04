import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "lab13_llm_finetune_academic_assistant_colab.ipynb"


def lines(text: str):
    return dedent(text).lstrip("\n").splitlines(keepends=True)


def md_cell(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


STUDENT_NAME = "Ali Hamza"
STUDENT_ROLL = "B23F0063AI106"
STUDENT_SECTION = "B.S AI - Red"
STUDENT_FOLDER_NAME = "Ali Hamza's Lab"


cells = [
    md_cell(
        f"""
        # Lab 13: Fine-Tuning a Domain-Specific Academic Assistant (LLMs)

        **Course:** COMP-341L - Artificial Neural Networks Lab  
        **Student:** {STUDENT_NAME}  
        **Roll Number:** {STUDENT_ROLL}  
        **Section:** {STUDENT_SECTION}  
        **Execution Environment:** Google Colab

        ## Objective
        Build a small, domain-specific academic assistant that explains CS concepts in a **custom teaching style**.

        This notebook is designed to satisfy the lab requirements:
        - Manual dataset (instruction → output) in a strict teaching style
        - Model selection + justification (small model)
        - Fine-tuning strategy (LoRA)
        - Evaluation (base vs fine-tuned) using the same prompt
        - Failure analysis (2 cases) and 1 manual improvement + re-evaluation
        """
    ),
    code_cell(
        f"""
        import os
        from datetime import datetime

        try:
            from google.colab import drive  # type: ignore
            IN_COLAB = True
        except Exception:
            IN_COLAB = False

        STUDENT_NAME = {STUDENT_NAME!r}
        STUDENT_ROLL = {STUDENT_ROLL!r}
        STUDENT_SECTION = {STUDENT_SECTION!r}
        STUDENT_FOLDER_NAME = {STUDENT_FOLDER_NAME!r}
        USE_GOOGLE_DRIVE = True

        if IN_COLAB:
            if not USE_GOOGLE_DRIVE:
                raise RuntimeError("Set USE_GOOGLE_DRIVE=True to save everything on Google Drive.")
            drive.mount("/content/drive", force_remount=True)
            BASE_DIR = f"/content/drive/MyDrive/COMP-341L/Lab 13/{{STUDENT_FOLDER_NAME}}"
            print("Google Drive mounted successfully.")
        else:
            BASE_DIR = os.environ.get("LAB13_BASE_DIR", ".")

        DATA_DIR = os.path.join(BASE_DIR, "data")
        OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
        ADAPTER_DIR = os.path.join(OUTPUTS_DIR, "lora_adapter")
        ADAPTER_DIR_V2 = os.path.join(OUTPUTS_DIR, "lora_adapter_v2")

        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

        print("IN_COLAB      :", IN_COLAB)
        print("USE_GOOGLE_DRIVE:", USE_GOOGLE_DRIVE)
        print("BASE_DIR      :", os.path.abspath(BASE_DIR))
        print("DATA_DIR      :", os.path.abspath(DATA_DIR))
        print("OUTPUTS_DIR   :", os.path.abspath(OUTPUTS_DIR))
        """
    ),
    md_cell(
        """
        ## Part 1 + Part 2 — Dataset (Manual) + Teaching Style

        The dataset is stored as JSONL in the lab folder:
        - `data/dataset_cs_instructions_v1.jsonl`
        - `data/dataset_cs_instructions_v2.jsonl` (after manual improvement)

        **Teaching style (strict):**
        1. Step-by-step explanation
        2. A real-life analogy
        3. A tiny example (1–3 lines)
        4. A common mistake / misconception
        5. A one-line recap

        The goal of fine-tuning is not to “teach new facts” to the model, but to **make outputs clearer, more consistent, and less likely to guess**.
        """
    ),
    code_cell(
        r"""
        import json
        from pathlib import Path

        v1_path = Path(DATA_DIR) / "dataset_cs_instructions_v1.jsonl"
        v2_path = Path(DATA_DIR) / "dataset_cs_instructions_v2.jsonl"

        # If you opened the notebook directly from the Drive folder, the dataset should already exist.
        # This fallback is only to keep the notebook runnable end-to-end.
        if not v1_path.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {v1_path}. Make sure you copied the lab folder to Google Drive."
            )

        def read_jsonl(path: Path):
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows

        data_v1 = read_jsonl(v1_path)
        data_v2 = read_jsonl(v2_path) if v2_path.exists() else []

        print("Loaded v1 samples:", len(data_v1))
        print("Loaded v2 samples:", len(data_v2))
        print("Example v1 item keys:", data_v1[0].keys())
        """
    ),
    md_cell(
        """
        ## Part 3 — Model Selection (Small LLM)

        **Chosen model:** `distilgpt2`

        **Why this model (reasoning):**
        - Small enough to fine-tune quickly on Colab
        - Good for demonstrating fine-tuning behavior with limited resources
        - Still produces coherent text for short explanations

        **Limitations (important):**
        - Weaker factual reliability than larger models
        - More likely to hallucinate if the prompt is vague
        - Limited context window and reasoning depth
        """
    ),
    md_cell(
        """
        ## Part 4 — Fine-Tuning Strategy (LoRA)

        We use **LoRA (Low-Rank Adaptation)** to train only a small number of additional parameters (adapters),
        instead of updating all model weights.

        **Why LoRA is suitable here:**
        - Faster training on limited hardware
        - Lower memory usage
        - Keeps base model intact; we can compare base vs adapter easily
        """
    ),
    code_cell(
        r"""
        import sys
        import subprocess

        def pip_install(pkgs):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)

        # Pin versions to avoid common Colab dependency mismatches (peft/transformers/huggingface_hub).
        pip_install(
            [
                "transformers==4.41.2",
                "datasets==2.19.1",
                "peft==0.11.1",
                "accelerate==0.31.0",
                "huggingface_hub==0.23.4",
                "safetensors",
            ]
        )

        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from peft import LoraConfig, get_peft_model, PeftModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        """
    ),
    code_cell(
        r"""
        BASE_MODEL_NAME = "distilgpt2"

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def format_example(instruction: str, output: str) -> str:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}\n"

        def to_dataset(rows):
            texts = [format_example(r["instruction"], r["output"]) for r in rows]
            return Dataset.from_dict({"text": texts})

        ds_v1 = to_dataset(data_v1)
        print(ds_v1)
        print(ds_v1[0]["text"][:250])
        """
    ),
    code_cell(
        r"""
        def tokenize_batch(batch):
            out = tokenizer(
                batch["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )
            out["labels"] = out["input_ids"].copy()
            return out

        tokenized_v1 = ds_v1.map(tokenize_batch, batched=True, remove_columns=["text"])
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        """
    ),
    code_cell(
        r"""
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
        base_model.resize_token_embeddings(len(tokenizer))
        base_model.to(device)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj"],
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        """
    ),
    md_cell(
        """
        ## Fine-tune (v1 dataset)

        Notes:
        - This is a tiny dataset; the goal is to learn *style and format*, not broad knowledge.
        - Keep epochs small to avoid overfitting / repetition.
        """
    ),
    code_cell(
        r"""
        training_args = TrainingArguments(
            output_dir=os.path.join(OUTPUTS_DIR, "trainer_runs_v1"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=8,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_v1,
            data_collator=data_collator,
        )

        trainer.train()
        model.save_pretrained(ADAPTER_DIR)
        tokenizer.save_pretrained(ADAPTER_DIR)
        print("Saved LoRA adapter to:", ADAPTER_DIR)
        """
    ),
    md_cell(
        """
        ## Part 5 — Evaluation (Base vs Fine-Tuned)

        Test prompt (fixed): **Explain dynamic programming**
        """
    ),
    code_cell(
        r"""
        import textwrap

        def generate_text(model_, prompt: str, max_new_tokens: int = 220):
            model_.eval()
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = model_.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(out_ids[0], skip_special_tokens=True)

        eval_instruction = "Explain dynamic programming in a step-by-step, analogy-based teaching style."
        eval_prompt = f"### Instruction:\n{eval_instruction}\n\n### Response:\n"

        base_eval_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device)
        tuned_eval_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device),
            ADAPTER_DIR,
        ).to(device)

        base_out = generate_text(base_eval_model, eval_prompt)
        tuned_out = generate_text(tuned_eval_model, eval_prompt)

        print("\n" + "=" * 40 + "\nBASE MODEL OUTPUT\n" + "=" * 40)
        print(base_out)
        print("\n" + "=" * 40 + "\nFINE-TUNED OUTPUT (v1)\n" + "=" * 40)
        print(tuned_out)

        (Path(OUTPUTS_DIR) / "eval_base.txt").write_text(base_out, encoding="utf-8")
        (Path(OUTPUTS_DIR) / "eval_finetuned_v1.txt").write_text(tuned_out, encoding="utf-8")
        print("\nSaved eval outputs to outputs/: eval_base.txt, eval_finetuned_v1.txt")
        """
    ),
    md_cell(
        """
        ## Part 6 — Failure Analysis (at least 2 cases)

        Below are two **intentionally chosen** prompts that often expose weak behavior in small LLMs:
        1) **Precision failure:** Ask for a strict, technical definition (model may get details wrong).
        2) **Overconfidence/hallucination:** Ask for details the model may “guess”.
        """
    ),
    code_cell(
        r"""
        failure_prompts = [
            "Explain dynamic programming and prove why it always gives the optimal answer.",
            "Explain the exact time complexity of the best-known algorithm for the traveling salesman problem and cite the year it was discovered.",
        ]

        def run_failures(tag: str, model_):
            outs = {}
            for i, instr in enumerate(failure_prompts, start=1):
                prompt = f"### Instruction:\n{instr}\n\n### Response:\n"
                out = generate_text(model_, prompt, max_new_tokens=220)
                outs[f"failure_{i}"] = out
                print("\n" + "-" * 30)
                print(f"{tag} | failure_{i} prompt:\n{instr}\n")
                print(out)
            return outs

        base_fail = run_failures("BASE", base_eval_model)
        tuned_fail_v1 = run_failures("TUNED_V1", tuned_eval_model)

        (Path(OUTPUTS_DIR) / "failures_base.json").write_text(json.dumps(base_fail, indent=2), encoding="utf-8")
        (Path(OUTPUTS_DIR) / "failures_finetuned_v1.json").write_text(json.dumps(tuned_fail_v1, indent=2), encoding="utf-8")
        print("\nSaved failure outputs to outputs/: failures_base.json, failures_finetuned_v1.json")
        """
    ),
    md_cell(
        """
        ## Part 7 — Manual Improvement (Fix ONE failure)

        Manual improvement strategy used here:
        - Add higher-quality examples that **teach the model how to respond when unsure** and how to define DP state clearly.
        - Train a second adapter on `dataset_cs_instructions_v2.jsonl`.

        Then we re-run the same evaluation prompt and compare.
        """
    ),
    code_cell(
        r"""
        if not v2_path.exists():
            raise FileNotFoundError(
                f"Missing improved dataset file: {v2_path}. Make sure it exists in your lab folder."
            )

        ds_v2 = to_dataset(read_jsonl(v2_path))
        tokenized_v2 = ds_v2.map(tokenize_batch, batched=True, remove_columns=["text"])

        base_model_v2 = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
        base_model_v2.resize_token_embeddings(len(tokenizer))
        base_model_v2.to(device)
        model_v2 = get_peft_model(base_model_v2, lora_config)

        training_args_v2 = TrainingArguments(
            output_dir=os.path.join(OUTPUTS_DIR, "trainer_runs_v2"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=6,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        )

        trainer_v2 = Trainer(
            model=model_v2,
            args=training_args_v2,
            train_dataset=tokenized_v2,
            data_collator=data_collator,
        )

        trainer_v2.train()
        model_v2.save_pretrained(ADAPTER_DIR_V2)
        tokenizer.save_pretrained(ADAPTER_DIR_V2)
        print("Saved LoRA adapter v2 to:", ADAPTER_DIR_V2)
        """
    ),
    code_cell(
        r"""
        tuned_eval_model_v2 = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(device),
            ADAPTER_DIR_V2,
        ).to(device)

        tuned_out_v2 = generate_text(tuned_eval_model_v2, eval_prompt)
        print("\n" + "=" * 40 + "\nFINE-TUNED OUTPUT (v2)\n" + "=" * 40)
        print(tuned_out_v2)

        (Path(OUTPUTS_DIR) / "eval_finetuned_v2.txt").write_text(tuned_out_v2, encoding="utf-8")

        tuned_fail_v2 = run_failures("TUNED_V2", tuned_eval_model_v2)
        (Path(OUTPUTS_DIR) / "failures_finetuned_v2.json").write_text(json.dumps(tuned_fail_v2, indent=2), encoding="utf-8")
        print("\nSaved outputs/: eval_finetuned_v2.txt, failures_finetuned_v2.json")
        """
    ),
    md_cell(
        """
        ## Reflection Report Export (Markdown + HTML)

        This section writes a structured report file to your Drive folder.
        It pulls in the saved model outputs to ensure the report contains **before/after evidence**.
        """
    ),
    code_cell(
        rf"""
        from pathlib import Path
        import html

        report_md_path = Path(BASE_DIR) / "Lab_Report_13.md"
        report_html_path = Path(BASE_DIR) / "Lab_Report_13.html"

        def safe_read(path: Path):
            return path.read_text(encoding="utf-8") if path.exists() else "(missing)"

        eval_base = safe_read(Path(OUTPUTS_DIR) / "eval_base.txt")
        eval_v1 = safe_read(Path(OUTPUTS_DIR) / "eval_finetuned_v1.txt")
        eval_v2 = safe_read(Path(OUTPUTS_DIR) / "eval_finetuned_v2.txt")
        failures_base = safe_read(Path(OUTPUTS_DIR) / "failures_base.json")
        failures_v1 = safe_read(Path(OUTPUTS_DIR) / "failures_finetuned_v1.json")
        failures_v2 = safe_read(Path(OUTPUTS_DIR) / "failures_finetuned_v2.json")

        today = datetime.now().strftime("%B %d, %Y")

        report_md = f\"\"\"# Lab 13 — Fine-Tuning a Domain-Specific Academic Assistant (LLMs)

        **Student:** {STUDENT_NAME}  
        **Roll Number:** {STUDENT_ROLL}  
        **Section:** {STUDENT_SECTION}  
        **Date:** {{today}}

        ## Part 1 — Dataset Creation (Manual)
        - v1 dataset: `data/dataset_cs_instructions_v1.jsonl` ({{len(data_v1)}} samples)
        - v2 dataset: `data/dataset_cs_instructions_v2.jsonl` ({{len(read_jsonl(v2_path))}} samples)
        - Format: JSONL with fields `instruction`, `output`
        - Manual constraint: Explanations written in a single consistent teaching style

        ## Part 2 — Teaching Style Definition
        **Style rules used in every output:**
        1) Step-by-step explanation  
        2) Analogy  
        3) Tiny example  
        4) Common mistake  
        5) One-line recap

        ## Part 3 — Model Selection
        **Selected:** `distilgpt2`
        - Advantages: small, fast to fine-tune, works on Colab
        - Limitations: more hallucination risk, weaker reasoning depth, limited context

        ## Part 4 — Fine-Tuning Strategy
        **Selected:** LoRA (PEFT)
        - Why: trains few parameters, faster + memory efficient
        - Effect: base weights frozen; only low-rank adapter matrices updated

        ## Part 5 — Evaluation (Same Prompt)
        **Prompt:** Explain dynamic programming

        ### Base model output
        ```text
        {{eval_base}}
        ```

        ### Fine-tuned output (v1)
        ```text
        {{eval_v1}}
        ```

        ### Fine-tuned output (v2, after manual improvement)
        ```text
        {{eval_v2}}
        ```

        **Critical comparison (write your analysis in your own words):**
        - Clarity: …
        - Hallucination: …
        - Style consistency: …

        ## Part 6 — Failure Analysis (2 cases)
        **Base failures (raw):**
        ```json
        {{failures_base}}
        ```

        **Fine-tuned v1 failures (raw):**
        ```json
        {{failures_v1}}
        ```

        **Fine-tuned v2 failures (raw):**
        ```json
        {{failures_v2}}
        ```

        **Your analysis (why it failed; data vs model vs training):**
        - Failure 1: …
        - Failure 2: …

        ## Part 7 — Manual Improvement (Fix one failure)
        - Change made: switched to v2 dataset with improved examples (DP state + uncertainty-aware answers)
        - Observed effect: …

        ## Reflection (Key Insights)
        - What the model learned: …
        - Where it still failed: …
        - What you would try next (more data / better prompts / eval metrics): …
        \"\"\"

        report_md_path.write_text(dedent(report_md).strip() + \"\\n\", encoding=\"utf-8\")

        # Minimal HTML export (keeps it simple and offline-friendly)
        html_body = \"<pre>\" + html.escape(report_md_path.read_text(encoding=\"utf-8\")) + \"</pre>\"
        report_html_path.write_text(
            \"<html><head><meta charset='utf-8'><title>Lab Report 13</title></head><body>\" + html_body + \"</body></html>\",
            encoding=\"utf-8\",
        )

        print(\"Wrote:\", report_md_path)
        print(\"Wrote:\", report_html_path)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
        "colab": {"name": NOTEBOOK_PATH.name, "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print("Wrote notebook:", NOTEBOOK_PATH)
