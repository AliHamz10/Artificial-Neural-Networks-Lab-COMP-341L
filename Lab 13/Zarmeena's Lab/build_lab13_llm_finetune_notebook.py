import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "lab13_llm_finetuning_ta_style_colab.ipynb"


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


cells = [
    md_cell(
        """
        # Lab 13 — Fine-Tuning LLMs: Building a Course-Style Academic Assistant

        **Course:** COMP-341L — Artificial Neural Networks Lab  
        **Student:** Zarmeena Jawad  
        **Roll No:** B23F0115AI125  
        **Section:** B.S AI - Red  
        **Platform:** Google Colab

        ## Why this lab?
        Pretrained models often answer in a generic style. The goal is to adapt a small model so it explains CS concepts like a teaching assistant:
        - clear structure
        - beginner-friendly but accurate
        - consistent tone

        ## Deliverables saved to Drive
        - curated dataset (manual)
        - before vs after outputs
        - failure cases + analysis notes
        - exported report (`Lab_Report_13.md`, `Lab_Report_13.html`)
        """
    ),
    code_cell(
        """
        import os
        from datetime import datetime

        try:
            from google.colab import drive  # type: ignore
            IN_COLAB = True
        except Exception:
            IN_COLAB = False

        STUDENT_NAME = "Zarmeena Jawad"
        STUDENT_ROLL = "B23F0115AI125"
        STUDENT_SECTION = "B.S AI - Red"
        STUDENT_FOLDER_NAME = "Zarmeena's Lab"
        USE_GOOGLE_DRIVE = True

        if IN_COLAB:
            if not USE_GOOGLE_DRIVE:
                raise RuntimeError("Set USE_GOOGLE_DRIVE=True so outputs save to Drive.")
            drive.mount("/content/drive", force_remount=True)
            BASE_DIR = f"/content/drive/MyDrive/COMP-341L/Lab 13/{STUDENT_FOLDER_NAME}"
            print("Drive mounted.")
        else:
            BASE_DIR = os.environ.get("LAB13_BASE_DIR", ".")

        DATA_DIR = os.path.join(BASE_DIR, "data")
        OUT_DIR = os.path.join(BASE_DIR, "outputs")
        PLOTS_DIR = os.path.join(OUT_DIR, "plots")

        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)

        print("BASE_DIR:", os.path.abspath(BASE_DIR))
        print("DATA_DIR:", os.path.abspath(DATA_DIR))
        print("OUT_DIR :", os.path.abspath(OUT_DIR))
        """
    ),
    md_cell(
        """
        ## Step 1 — Install Libraries
        We will fine-tune using HuggingFace + PEFT (LoRA).
        """
    ),
    code_cell(
        """
        import sys
        import subprocess

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)

        # Full clean reinstall to avoid NumPy ABI mismatch.
        # IMPORTANT: After this cell finishes, do Runtime → Restart runtime.
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "numpy", "pandas", "pyarrow", "transformers", "huggingface_hub", "peft", "accelerate"],
            check=False,
        )

        # Aggressive cleanup (Colab can retain stray numpy folders that cause ABI mismatch)
        import os
        import glob
        import shutil
        import site

        for sp in site.getsitepackages():
            for pat in ["numpy", "numpy-*.dist-info", "numpy-*.data"]:
                for p in glob.glob(os.path.join(sp, pat)):
                    shutil.rmtree(p, ignore_errors=True)

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade", "--force-reinstall", "numpy==2.0.2"],
            check=True,
        )

        # Pin compatible versions (PEFT 0.12.0 expects Transformers v4.x).
        pkgs = [
            "huggingface_hub==0.24.6",
            "transformers==4.45.2",
            "peft==0.12.0",
            "accelerate==0.33.0",
        ]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--upgrade",
                "--force-reinstall",
                "--no-cache-dir",
            ]
            + pkgs,
            check=True,
        )
        print("Installed/Pinned:", pkgs)
        print("Note: This notebook avoids `datasets` to prevent pandas/pyarrow dependency issues on Colab.")

        import numpy as np
        import transformers
        import huggingface_hub

        print("numpy:", np.__version__)
        print("transformers:", transformers.__version__)
        print("huggingface_hub:", huggingface_hub.__version__)
        print("has is_offline_mode:", hasattr(huggingface_hub, "is_offline_mode"))
        assert transformers.__version__.startswith("4."), "Transformers must be v4.x for this notebook."

        import numpy as np
        import transformers
        import huggingface_hub

        print("numpy:", np.__version__)
        print("transformers:", transformers.__version__)
        print("huggingface_hub:", huggingface_hub.__version__)
        print("has is_offline_mode:", hasattr(huggingface_hub, "is_offline_mode"))
        print("numpy file:", np.__file__)
        assert transformers.__version__.startswith("4."), "Transformers must be v4.x for this notebook."

        from transformers import DataCollatorForLanguageModeling
        print("DataCollatorForLanguageModeling import: OK")

        print("NEXT STEP: Runtime → Restart runtime, then run from the top again.")
        """
    ),
    md_cell(
        """
        ## Step 2 — Teaching Style (You define this)
        This style must be consistent across the dataset.
        """
    ),
    code_cell(
        r"""
        import json
        from pathlib import Path

        STYLE = {
            "tone": "friendly-academic (TA), avoids slang",
            "rules": [
                "Start with a 1-line definition",
                "Then give 3–5 bullet steps",
                "Then give 1 small example",
                "End with 'Common Pitfall' + 1 line",
            ],
        }

        style_path = Path(DATA_DIR) / "teaching_style.json"
        style_path.write_text(json.dumps(STYLE, indent=2), encoding="utf-8")
        print("Saved style:", style_path)
        """
    ),
    md_cell(
        """
        ## Step 3 — Dataset Creation (Manual)
        Create/edit `academic_assistant.jsonl` with 20–30 examples:
        - `instruction`
        - `output` (your own words, consistent with the style above)
        """
    ),
    code_cell(
        r"""
        import json
        from pathlib import Path

        dataset_path = Path(DATA_DIR) / "academic_assistant.jsonl"

        seed_items = [
            {
                "instruction": "Explain recursion using an everyday analogy.",
                "output": "Definition: Recursion is when a function solves a problem by calling itself on a smaller input.\n\nSteps:\n- Identify the repeating subproblem.\n- Define a base case that stops the recursion.\n- Ensure each call reduces the problem size.\n\nExample: factorial(n) = n * factorial(n-1) with factorial(0)=1.\n\nCommon Pitfall: Missing a base case leads to infinite recursion.",
            },
            {
                "instruction": "Explain dynamic programming in a structured teaching-assistant style.",
                "output": "Definition: Dynamic programming (DP) solves problems by storing solutions of overlapping subproblems.\n\nSteps:\n- Check for optimal substructure.\n- Define the state (what you store).\n- Write the recurrence (transition).\n- Compute using memoization or tabulation.\n\nExample: Fibonacci with memoization avoids recomputing F(n-1), F(n-2).\n\nCommon Pitfall: A wrong state definition makes the recurrence incorrect.",
            },
            {
                "instruction": "Explain what a stack is and where it is used.",
                "output": "Definition: A stack is a LIFO data structure (Last-In, First-Out).\n\nSteps:\n- push inserts on top.\n- pop removes from top.\n- top/peek reads the top element.\n\nExample: Function call stack stores return addresses and local variables.\n\nCommon Pitfall: Popping an empty stack causes underflow.",
            },
        ]

        if not dataset_path.exists():
            with open(dataset_path, "w", encoding="utf-8") as f:
                for row in seed_items:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("Wrote starter dataset:", dataset_path)
        else:
            print("Dataset exists (edit it manually):", dataset_path)

        # sanity-check
        rows = [json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print("Samples:", len(rows))
        print("Example instruction:", rows[0]["instruction"])
        """
    ),
    md_cell(
        """
        ## Step 4 — Model Selection
        We use **GPT-2** (small) so it can run on Colab easily.
        """
    ),
    md_cell(
        """
        ## Step 5 — Fine-Tuning Strategy: LoRA (PEFT)
        We adapt the base model using low-rank updates (few trainable parameters).
        """
    ),
    code_cell(
        """
        import json
        import torch

        import random
        from torch.utils.data import Dataset as TorchDataset
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
            set_seed,
        )
        from peft import LoraConfig, TaskType, get_peft_model

        set_seed(7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", device)

        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))

        if len(data) < 8:
            print("WARNING: add more items (target 20–30) for the final submission. Current:", len(data))

        rng = random.Random(7)
        idx = list(range(len(data)))
        rng.shuffle(idx)
        split = int(0.8 * len(idx))
        train_rows = [data[i] for i in idx[:split]]
        eval_rows = [data[i] for i in idx[split:]]
        print("train:", len(train_rows), "eval:", len(eval_rows))

        MODEL_NAME = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            base_model.config.pad_token_id = tokenizer.eos_token_id

        def pack(instruction: str, output: str):
            # Different prompt template than Ali's to keep notebooks distinct
            return (
                "<INSTRUCTION>\\n"
                + instruction.strip()
                + "\\n</INSTRUCTION>\\n"
                + "<RESPONSE>\\n"
                + output.strip()
                + "\\n</RESPONSE>"
                + tokenizer.eos_token
            )

        def tokenize_rows(rows):
            texts = [pack(r["instruction"], r["output"]) for r in rows]
            return tokenizer(texts, max_length=384, truncation=True, padding=False)

        train_enc = tokenize_rows(train_rows)
        eval_enc = tokenize_rows(eval_rows)

        class EncodedTextDataset(TorchDataset):
            def __init__(self, enc):
                self.enc = enc

            def __len__(self):
                return len(self.enc["input_ids"])

            def __getitem__(self, i):
                item = {k: torch.tensor(v[i]) for k, v in self.enc.items() if k in ["input_ids", "attention_mask"]}
                return item

        train_tok = EncodedTextDataset(train_enc)
        eval_tok = EncodedTextDataset(eval_enc)

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.08,
            bias="none",
            target_modules=["c_attn"],  # keep target smaller and distinct
        )

        model = get_peft_model(base_model, lora)
        model.print_trainable_parameters()
        """
    ),
    md_cell(
        """
        ## Step 6 — Train
        """
    ),
    code_cell(
        """
        args = TrainingArguments(
            output_dir=os.path.join(OUT_DIR, "lora_gpt2"),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=10,
            learning_rate=2e-4,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            report_to=[],
            fp16=torch.cuda.is_available(),
            optim="adamw_torch",
            seed=7,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            data_collator=collator,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        print("Eval:", eval_metrics)

        save_dir = os.path.join(OUT_DIR, "lora_gpt2", "final_adapter")
        os.makedirs(save_dir, exist_ok=True)
        trainer.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("Saved adapter:", save_dir)
        """
    ),
    md_cell(
        """
        ## Step 7 — Before vs After (same prompt)
        Prompt: **Explain dynamic programming**
        """
    ),
    code_cell(
        """
        from transformers import pipeline
        from peft import PeftModel

        def gen(m, prompt: str, max_new_tokens: int = 180):
            pipe = pipeline(
                "text-generation",
                model=m,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
            return pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.75,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )[0]["generated_text"]

        prompt = "<INSTRUCTION>\\nExplain dynamic programming.\\n</INSTRUCTION>\\n<RESPONSE>\\n"

        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        if base.config.pad_token_id is None:
            base.config.pad_token_id = tokenizer.eos_token_id

        tuned = PeftModel.from_pretrained(base, save_dir).to(device)

        base_out = gen(base, prompt)
        tuned_out = gen(tuned, prompt)

        print("=== Base ===\\n", base_out)
        print("\\n=== Tuned ===\\n", tuned_out)

        out_path = os.path.join(OUT_DIR, "before_after_dp.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("PROMPT:\\n" + prompt + "\\n\\n")
            f.write("=== BASE ===\\n" + base_out + "\\n\\n")
            f.write("=== TUNED ===\\n" + tuned_out + "\\n")
        print("Saved:", out_path)
        """
    ),
    md_cell(
        """
        ## Step 8 — Failure Cases + Manual Fix
        Run 3 tough prompts, analyze failures, then fix ONE case by improving your dataset and retraining briefly.
        """
    ),
    code_cell(
        """
        hard_prompts = [
            "<INSTRUCTION>\\nExplain concurrency vs parallelism with one example.\\n</INSTRUCTION>\\n<RESPONSE>\\n",
            "<INSTRUCTION>\\nExplain why quicksort worst-case is O(n^2) and how to avoid it.\\n</INSTRUCTION>\\n<RESPONSE>\\n",
            "<INSTRUCTION>\\nExplain backpropagation in a small neural network without skipping steps.\\n</INSTRUCTION>\\n<RESPONSE>\\n",
        ]

        path = os.path.join(OUT_DIR, "failure_cases.txt")
        with open(path, "w", encoding="utf-8") as f:
            for idx, hp in enumerate(hard_prompts, start=1):
                b = gen(base, hp, 200)
                t = gen(tuned, hp, 200)
                f.write(f"\\n--- Case {idx} ---\\n")
                f.write("PROMPT:\\n" + hp + "\\n\\n")
                f.write("BASE:\\n" + b + "\\n\\n")
                f.write("TUNED:\\n" + t + "\\n")
        print("Saved:", path)
        """
    ),
    md_cell(
        """
        ## Export (Lab Report)
        Exports an A4, one-column HTML report with black code blocks (white text) to match the lab submission style.
        """
    ),
    code_cell(
        """
        import json

        report_md = f\"\"\"# Lab 13 — Fine-Tuning LLMs (Academic Assistant)

**Student:** {STUDENT_NAME}  
**Roll No:** {STUDENT_ROLL}  
**Section:** {STUDENT_SECTION}  
**Date:** {datetime.now().strftime('%B %d, %Y')}

## Dataset
- `data/academic_assistant.jsonl`
- Samples: `{len(data)}` (target 20–30)

## Model + Method
- Model: `{MODEL_NAME}`
- Strategy: LoRA (PEFT)

## Before vs After
- `outputs/before_after_dp.txt`

## Failure Cases
- `outputs/failure_cases.txt`

## Reflection (write your own)
- what improved
- what failed and why (data/model/training)
\"\"\"

        html = f\"\"\"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>COMP-341L — Lab 13 Report — {STUDENT_NAME}</title>
    <style>
      * {{ box-sizing: border-box; }}
      html, body {{ background: #fff; color: #000; }}
      body {{
        margin: 0;
        font-family: "Times New Roman", Times, serif;
        font-size: 10pt;
        line-height: 1.25;
      }}
      .paper {{
        width: 210mm;
        max-width: 210mm;
        margin: 0 auto;
        padding: 20mm 18mm 22mm;
      }}
      .title {{
        text-align: center;
        font-size: 18pt;
        font-weight: bold;
        margin: 0 0 6px;
      }}
      .meta {{
        text-align: center;
        font-size: 10pt;
        margin: 0 0 14px;
      }}
      h2 {{
        font-size: 10pt;
        font-weight: bold;
        text-transform: uppercase;
        margin: 12px 0 6px;
      }}
      p {{ margin: 0 0 8px; text-align: justify; }}
      pre {{
        border: 1px solid #000;
        padding: 8px;
        margin: 8px 0 10px;
        white-space: pre;
        overflow-x: auto;
        font-family: "Courier New", Courier, monospace;
        font-size: 9pt;
        line-height: 1.2;
        background: #000;
        color: #fff;
      }}
      code {{ font-family: "Courier New", Courier, monospace; font-size: 9pt; }}
      @media print {{
        @page {{ size: A4; margin: 20mm 18mm 22mm; }}
        .paper {{ padding: 0; margin: 0; width: auto; max-width: none; }}
      }}
    </style>
  </head>
  <body>
    <div class="paper">
      <h1 class="title">Lab 13 Report: Fine-Tuning LLMs (Academic Assistant)</h1>
      <p class="meta">{STUDENT_NAME} ({STUDENT_ROLL}) — {STUDENT_SECTION} • Submission Date: {datetime.now().strftime('%B %d, %Y')}</p>

      <h2>Abstract</h2>
      <p>
        A small GPT-style model is adapted to produce structured course-style explanations for computer science concepts. LoRA is used to update a small set
        of parameters while keeping most model weights fixed.
      </p>

      <h2>Teaching Style</h2>
      <pre><code>{json.dumps(STYLE, indent=2)}</code></pre>

      <h2>Prompt Template</h2>
      <pre><code>&lt;INSTRUCTION&gt;
...your instruction...
&lt;/INSTRUCTION&gt;
&lt;RESPONSE&gt;
...assistant response...
&lt;/RESPONSE&gt;</code></pre>

      <h2>Artifacts Saved</h2>
      <p>
        Dataset: <code>data/academic_assistant.jsonl</code><br/>
        Before/after: <code>outputs/before_after_dp.txt</code><br/>
        Failure cases: <code>outputs/failure_cases.txt</code>
      </p>

      <h2>Reflection</h2>
      <p>
        (Write your own analysis: what improved, where hallucination/oversimplification appears, and why.)
      </p>
    </div>
  </body>
</html>
\"\"\"

        md_path = os.path.join(BASE_DIR, "Lab_Report_13.md")
        html_path = os.path.join(BASE_DIR, "Lab_Report_13.html")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        print("Saved:", md_path)
        print("Saved:", html_path)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
        "colab": {"name": "lab13_llm_finetuning_ta_style_colab.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print("Wrote:", NOTEBOOK_PATH)
