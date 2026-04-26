import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "lab13_llm_finetuning_academic_assistant_colab.ipynb"


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
        # Lab 13: Introduction to Fine-Tuning Large Language Models (LLMs)

        **Course:** COMP-341L — Artificial Neural Networks Lab  
        **Student:** Ali Hamza  
        **Roll Number:** B23F0063AI106  
        **Section:** B.S AI - Red  
        **Execution Environment:** Google Colab

        ## Learning Objectives
        - Understand fine-tuning at a conceptual + mathematical level
        - Connect fine-tuning to Transformer attention (Q, K, V)
        - Differentiate: full fine-tuning vs PEFT vs prompting
        - Implement a practical fine-tuning pipeline using HuggingFace
        - Apply LoRA (parameter-efficient fine-tuning)
        - Evaluate and debug a fine-tuned model

        ## Lab Task (Goal)
        Fine-tune a small LLM to behave like a **university-level teaching assistant**:
        - structured explanations
        - academic tone
        - step-by-step clarity

        ## Important note (manual work requirement)
        The lab requires **manual rewriting** of 20–30 samples into instruction format.
        This notebook provides a ready-to-edit dataset file in Google Drive. You must review, rewrite,
        and adjust it to reflect *your own understanding* before final submission.
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

        STUDENT_NAME = "Ali Hamza"
        STUDENT_ROLL = "B23F0063AI106"
        STUDENT_SECTION = "B.S AI - Red"
        STUDENT_FOLDER_NAME = "Ali Hamza's Lab"
        USE_GOOGLE_DRIVE = True

        if IN_COLAB:
            if not USE_GOOGLE_DRIVE:
                raise RuntimeError("Set USE_GOOGLE_DRIVE=True to save everything on Google Drive.")
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

        print("IN_COLAB:", IN_COLAB)
        print("BASE_DIR:", os.path.abspath(BASE_DIR))
        print("DATA_DIR:", os.path.abspath(DATA_DIR))
        print("OUT_DIR :", os.path.abspath(OUT_DIR))
        """
    ),
    md_cell(
        """
        ## Part 0: Install Dependencies
        We will use:
        - `transformers` for model + tokenizer
        - `peft` for LoRA fine-tuning
        - `accelerate` for training utilities
        """
    ),
    code_cell(
        """
        import sys
        import subprocess

        # Colab environment repair:
        # If you see errors like:
        # - "numpy.dtype size changed ..."
        # - "No module named 'numpy.strings'"
        # your runtime has an incompatible NumPy install.
        #
        # Fix: FULL clean reinstall of NumPy and HF libs (factory-reset style).
        # IMPORTANT: After this cell finishes, do Runtime → Restart runtime.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)

        # Remove any conflicting installs first
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "numpy", "pandas", "pyarrow", "transformers", "huggingface_hub", "peft", "accelerate"],
            check=False,
        )

        # Aggressive cleanup (Colab can retain stray numpy folders that cause ABI mismatch)
        import glob
        import shutil
        import site

        for sp in site.getsitepackages():
            for pat in ["numpy", "numpy-*.dist-info", "numpy-*.data"]:
                for p in glob.glob(os.path.join(sp, pat)):
                    shutil.rmtree(p, ignore_errors=True)

        # Install NumPy first (ensures ABI consistency for anything importing NumPy C-extensions)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade", "--force-reinstall", "numpy==2.0.2"],
            check=True,
        )

        # Pin compatible versions.
        # `peft==0.12.0` is built for Transformers v4.x. If Colab installs Transformers v5.x,
        # you can hit import/runtime errors. So we pin Transformers < 5.
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
        # Older/newer hub versions differ; we only require compatibility with transformers 4.45.2
        print("has is_offline_mode:", hasattr(huggingface_hub, "is_offline_mode"))
        print("numpy file:", np.__file__)
        assert transformers.__version__.startswith("4."), "Transformers must be v4.x for this notebook."

        # Quick import test for the exact failure you reported:
        from transformers import DataCollatorForLanguageModeling
        print("DataCollatorForLanguageModeling import: OK")

        print("NEXT STEP: Runtime → Restart runtime, then run from the top again.")
        """
    ),
    md_cell(
        """
        ## Part 1: Dataset Creation (Instruction Format)

        We create a JSONL file with records like:

        ```json
        {"instruction": "...", "output": "..."}
        ```

        **You must edit these examples** so the explanations are:
        - conceptually correct
        - in your own words
        - consistent teaching style
        """
    ),
    code_cell(
        r"""
        import json
        from pathlib import Path

        dataset_path = Path(DATA_DIR) / "academic_assistant_instructions.jsonl"

        # Teaching style definition (used for manual rewriting)
        TEACHING_STYLE = {
            "tone": "university-level, academic, precise",
            "structure": [
                "Definition (1–2 lines)",
                "Step-by-step explanation",
                "Analogy (1 short real-life analogy)",
                "Mini example (simple)",
                "Common mistake / misconception",
                "Key takeaways (2–3 bullets)",
            ],
        }

        # Starter dataset (EDIT THIS MANUALLY before final submission)
        # Keep 20–30 items.
        starter = [
            {
                "instruction": "Explain recursion using a real-life analogy (teaching assistant style).",
                "output": "Definition: Recursion is a method where a function solves a problem by calling itself on a smaller version of the same problem.\n\nStep-by-step: (1) Identify the repeated sub-problem. (2) Define a base case that stops the calls. (3) Each call reduces the problem size until the base case.\n\nAnalogy: Think of two mirrors facing each other; each reflection contains a smaller reflection of the same scene until it becomes too small to notice.\n\nMini example: Factorial: n! = n × (n−1)! with base case 0! = 1.\n\nCommon mistake: Forgetting a base case causes infinite recursion.\n\nKey takeaways:\n- Recursion = self-reference + smaller sub-problem\n- Base case prevents infinite calls\n- Often matches naturally recursive structures (trees).",
            },
            {
                "instruction": "Explain dynamic programming for beginners, but keep it academically correct.",
                "output": "Definition: Dynamic programming (DP) is an optimization technique that solves problems by combining solutions to overlapping subproblems.\n\nStep-by-step: (1) Confirm optimal substructure and overlapping subproblems. (2) Define a state (what you store). (3) Write a recurrence (transition). (4) Compute using memoization (top-down) or tabulation (bottom-up).\n\nAnalogy: Reusing saved notes instead of re-deriving the same formula every time.\n\nMini example: Fibonacci with memoization stores results of F(n−1), F(n−2) so they aren’t recomputed.\n\nCommon mistake: Choosing a state that misses necessary information, leading to wrong transitions.\n\nKey takeaways:\n- DP trades memory for speed\n- State + transition define the solution\n- Memoization and tabulation are two implementations.",
            },
            {
                "instruction": "Explain what overfitting is and how to reduce it (short but structured).",
                "output": "Definition: Overfitting occurs when a model learns noise and details of the training data, reducing performance on unseen data.\n\nStep-by-step: (1) Detect via train vs validation gap. (2) Reduce model capacity or increase regularization. (3) Improve data quality/quantity.\n\nAnalogy: Memorizing answers to past papers instead of learning concepts.\n\nMini example: A model achieves 99% training accuracy but 70% validation accuracy.\n\nCommon mistake: Training longer always helps—after a point it can worsen generalization.\n\nKey takeaways:\n- Overfitting = poor generalization\n- Use dropout, weight decay, early stopping\n- More diverse data often helps most.",
            },
            {
                "instruction": "Explain the difference between BFS and DFS in graphs, with a small example.",
                "output": "Definition: BFS explores nodes level-by-level, while DFS explores as deep as possible before backtracking.\n\nStep-by-step: BFS uses a queue; DFS uses a stack (or recursion). BFS finds shortest paths in unweighted graphs; DFS is useful for topological ideas and connectivity.\n\nAnalogy: BFS is like searching rooms floor-by-floor; DFS is like going down one hallway fully before trying another.\n\nMini example: From node A, BFS visits neighbors first; DFS visits one neighbor chain until it ends.\n\nCommon mistake: Assuming DFS gives shortest path—it usually doesn’t.\n\nKey takeaways:\n- BFS: queue, shortest path in unweighted graphs\n- DFS: stack/recursion, depth exploration\n- Both are O(V+E).",
            },
            {
                "instruction": "Explain what a hash function is and why collisions happen.",
                "output": "Definition: A hash function maps input data to a fixed-size value (hash), often used for indexing and integrity.\n\nStep-by-step: (1) Input is processed by a deterministic algorithm. (2) Output is a fixed-length digest. (3) Many different inputs must share the limited output space.\n\nAnalogy: Assigning students to lockers with limited locker numbers.\n\nMini example: If the hash outputs only 100 possible values, two different strings can map to the same value.\n\nCommon mistake: Confusing cryptographic hash with encryption; hashes are one-way.\n\nKey takeaways:\n- Collisions are unavoidable with finite outputs\n- Good hashes distribute outputs uniformly\n- Cryptographic hashes resist intentional collisions.",
            },
            # Add more items yourself (target 20–30). Keep them consistent with the style above.
        ]

        # If file doesn't exist, write starter dataset.
        if not dataset_path.exists():
            with open(dataset_path, "w", encoding="utf-8") as f:
                for row in starter:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("Wrote starter dataset:", dataset_path)
        else:
            print("Dataset already exists (edit it manually):", dataset_path)

        # Quick validation
        rows = [json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print("Dataset samples:", len(rows))
        print("First instruction:", rows[0]["instruction"])
        """
    ),
    md_cell(
        """
        ## Part 2: Fine-Tuning Concepts (Q, K, V)

        In self-attention:
        - **Q = XWq**, **K = XWk**, **V = XWv**
        - Fine-tuning updates weights (fully or partially), changing how attention scores are computed.

        **Key intuition:** You’re not “teaching language from scratch”; you’re **reshaping attention flow** and the next-token distribution
        to match your dataset’s style and structure.
        """
    ),
    md_cell(
        """
        ## Part 3: Model Selection

        We use **DistilGPT2** because:
        - small and fast (Colab-friendly)
        - still demonstrates transformer fine-tuning behavior

        **Limitations:**
        - small models can hallucinate
        - limited reasoning depth compared to larger LLMs
        """
    ),
    md_cell(
        """
        ## Part 4: Fine-Tuning Strategy — LoRA (PEFT)

        LoRA updates a low-rank correction:

        **W' = W + A B**

        This allows training a small number of parameters while keeping the base model mostly frozen.
        On GPT-style models, LoRA commonly targets attention projections.
        """
    ),
    code_cell(
        """
        import json
        import math
        import torch

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

        set_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", device)

        # Load dataset
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        assert len(data) >= 5, "Add more examples (target 20–30) before final submission."

        # Simple train/eval split without HuggingFace `datasets` (avoids pandas/pyarrow dependency)
        rng = random.Random(42)
        idx = list(range(len(data)))
        rng.shuffle(idx)
        split = int(0.8 * len(idx))
        train_idx, eval_idx = idx[:split], idx[split:]
        train_rows = [data[i] for i in train_idx]
        eval_rows = [data[i] for i in eval_idx]
        print("train:", len(train_rows), "eval:", len(eval_rows))

        MODEL_NAME = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        # GPT2 has no pad token by default
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        def format_example(instruction: str, output: str):
            return (
                "### Instruction:\\n"
                + instruction.strip()
                + "\\n\\n### Answer:\\n"
                + output.strip()
                + tokenizer.eos_token
            )

        def tokenize_rows(rows):
            texts = [format_example(r["instruction"], r["output"]) for r in rows]
            return tokenizer(texts, max_length=512, truncation=True, padding=False)

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

        # LoRA config for GPT2-style models
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["c_attn", "c_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        """
    ),
    md_cell(
        """
        ## Part 5: Train
        We train for a small number of epochs (dataset is small). For final submission, increase dataset quality and
        keep training stable with a small learning rate.
        """
    ),
    code_cell(
        """
        train_args = TrainingArguments(
            output_dir=os.path.join(OUT_DIR, "lora_distilgpt2"),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=10,
            learning_rate=2e-4,
            weight_decay=0.0,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            report_to=[],
            fp16=torch.cuda.is_available(),
            optim="adamw_torch",
            seed=42,
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            data_collator=collator,
        )

        train_result = trainer.train()
        metrics = trainer.evaluate()
        print("Eval metrics:", metrics)

        # Save adapter weights + tokenizer (PEFT)
        save_dir = os.path.join(OUT_DIR, "lora_distilgpt2", "final_adapter")
        os.makedirs(save_dir, exist_ok=True)
        trainer.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("Saved adapter to:", save_dir)
        """
    ),
    md_cell(
        """
        ## Part 6: Evaluation (Before vs After)
        Use the same prompt:
        **“Explain dynamic programming”**
        Compare base model vs fine-tuned model outputs.
        """
    ),
    code_cell(
        """
        from transformers import pipeline
        from peft import PeftModel

        def generate_text(model, prompt: str, max_new_tokens: int = 180):
            gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
            )
            out = gen(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )[0]["generated_text"]
            return out

        prompt = "### Instruction:\\nExplain dynamic programming.\\n\\n### Answer:\\n"

        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        if base.config.pad_token_id is None:
            base.config.pad_token_id = tokenizer.eos_token_id

        # load fine-tuned adapter onto base
        tuned = PeftModel.from_pretrained(base, save_dir).to(device)

        base_out = generate_text(base, prompt)
        tuned_out = generate_text(tuned, prompt)

        print("=== Base Model Output ===")
        print(base_out)
        print("\\n=== Fine-tuned Model Output ===")
        print(tuned_out)

        out_path = os.path.join(OUT_DIR, "before_after_dynamic_programming.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("PROMPT:\\n" + prompt + "\\n\\n")
            f.write("=== Base ===\\n" + base_out + "\\n\\n")
            f.write("=== Fine-tuned ===\\n" + tuned_out + "\\n")
        print("Saved:", out_path)
        """
    ),
    md_cell(
        """
        ## Part 7: Failure Analysis (3 cases)
        Generate outputs for 3 prompts where the model might fail (e.g., hallucination, oversimplification).
        Then write your analysis: **why it failed** (data vs model vs training).
        """
    ),
    code_cell(
        """
        failure_prompts = [
            "### Instruction:\\nExplain backpropagation with a small worked example.\\n\\n### Answer:\\n",
            "### Instruction:\\nExplain time complexity of quicksort and when it becomes worst-case.\\n\\n### Answer:\\n",
            "### Instruction:\\nExplain what a mutex is and why deadlocks happen.\\n\\n### Answer:\\n",
        ]

        rows = []
        for i, fp in enumerate(failure_prompts, start=1):
            b = generate_text(base, fp, max_new_tokens=200)
            t = generate_text(tuned, fp, max_new_tokens=200)
            rows.append((i, fp, b, t))

        fail_path = os.path.join(OUT_DIR, "failure_cases.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            for i, fp, b, t in rows:
                f.write(f"\\n--- Case {i} ---\\n")
                f.write("PROMPT:\\n" + fp + "\\n\\n")
                f.write("BASE:\\n" + b + "\\n\\n")
                f.write("TUNED:\\n" + t + "\\n")
        print("Saved:", fail_path)
        """
    ),
    md_cell(
        """
        ## Part 8: Manual Improvement (Fix ONE failure case)
        Choose 1 failure case and improve the dataset:
        - Add a better example
        - Rewrite instruction formatting
        - Increase clarity and correctness

        Then re-train quickly (few epochs) and re-evaluate that prompt.
        """
    ),
    md_cell(
        """
        ## Export Deliverables (Report)
        This cell exports a Markdown + HTML report into Google Drive.
        """
    ),
    code_cell(
        """
        report_md = f\"\"\"# Lab 13: Fine-Tuning LLMs (Academic Assistant)

**Course:** COMP-341L — Artificial Neural Networks Lab  
**Student:** {STUDENT_NAME}  
**Roll Number:** {STUDENT_ROLL}  
**Section:** {STUDENT_SECTION}  
**Date:** {datetime.now().strftime('%B %d, %Y')}

## Teaching Style Definition
{json.dumps(TEACHING_STYLE, indent=2)}

## Dataset
- Path: `data/academic_assistant_instructions.jsonl`
- Samples: `{len(data)}` (target 20–30)

## Model Selection
- Model: `{MODEL_NAME}`
- Reason: small, Colab-friendly, demonstrates transformer fine-tuning

## Fine-Tuning Strategy
- Method: LoRA (PEFT)
- Target modules: `c_attn`, `c_proj`

## Training Setup
- Epochs: `{int(train_args.num_train_epochs)}`
- Effective batch size: `{int(train_args.per_device_train_batch_size * train_args.gradient_accumulation_steps)}`
- Learning rate: `{train_args.learning_rate}`

## Before vs After
Saved outputs:
- `outputs/before_after_dynamic_programming.txt`

## Failure Cases + Analysis
Saved outputs:
- `outputs/failure_cases.txt`

## Reflection (Write in your own words)
Explain:
- What the model learned
- Where it failed (hallucination / oversimplification / format issues)
- Why (data vs model size vs training)
\"\"\"

        html = f\"\"\"<!doctype html>
<html lang='en'>
  <head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
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
      h1 {{
        text-align: center;
        font-size: 18pt;
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
        background: #000;
        color: #fff;
        font-family: "Courier New", Courier, monospace;
        font-size: 9pt;
        overflow-x: auto;
      }}
      code {{ font-family: "Courier New", Courier, monospace; font-size: 9pt; }}
      @media print {{
        @page {{ size: A4; margin: 20mm 18mm 22mm; }}
        .paper {{ padding: 0; margin: 0; width: auto; max-width: none; }}
      }}
    </style>
  </head>
  <body>
    <div class='paper'>
      <h1>Lab 13 Report: Fine-Tuning LLMs (Academic Assistant)</h1>
      <p class='meta'>{STUDENT_NAME} ({STUDENT_ROLL}) — {STUDENT_SECTION} • Submission Date: {datetime.now().strftime('%B %d, %Y')}</p>

      <h2>Abstract</h2>
      <p>
        This lab fine-tunes a small GPT-style language model to produce structured, academic explanations for computer science concepts.
        Parameter-Efficient Fine-Tuning (LoRA) is used to reduce memory and training cost while adapting model behavior.
      </p>

      <h2>Key Idea (Q, K, V)</h2>
      <p>
        In attention, Q=XWq, K=XWk, V=XWv. Fine-tuning updates weights that shape attention flow and next-token probabilities. LoRA adapts these weights
        through low-rank corrections, enabling efficient specialization.
      </p>

      <h2>Teaching Style</h2>
      <pre><code>{json.dumps(TEACHING_STYLE, indent=2)}</code></pre>

      <h2>Dataset</h2>
      <p>
        A curated instruction dataset is stored at <code>data/academic_assistant_instructions.jsonl</code>. Total samples: <code>{len(data)}</code>.
      </p>

      <h2>Method</h2>
      <p>
        Model: <code>{MODEL_NAME}</code>. Adaptation: <code>LoRA</code> on attention modules (<code>c_attn</code>, <code>c_proj</code>).
      </p>

      <h2>Important Code Snippet (Prompt Formatting)</h2>
      <pre><code>### Instruction:
{{instruction}}

### Answer:
{{output}}</code></pre>

      <h2>Training Setup</h2>
      <p>
        Epochs: <code>{int(train_args.num_train_epochs)}</code>, LR: <code>{train_args.learning_rate}</code>,
        Batch size (effective): <code>{int(train_args.per_device_train_batch_size * train_args.gradient_accumulation_steps)}</code>.
      </p>

      <h2>Evaluation</h2>
      <p>
        Before/after outputs are saved to <code>outputs/before_after_dynamic_programming.txt</code>. Failure cases saved to
        <code>outputs/failure_cases.txt</code>.
      </p>

      <h2>Reflection</h2>
      <p>
        (Write your analysis here in your own words: what improved, where failures remain, and why.)
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
        "colab": {"name": "lab13_llm_finetuning_academic_assistant_colab.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print("Wrote:", NOTEBOOK_PATH)
