"""
Execute Assignment_1.ipynb cell-by-cell, save the notebook with outputs,
and write results to results.json for the report.
Run from project root: ./venv/bin/python Assignments/run_and_capture.py
"""
import json
import sys
import os
from io import StringIO

# Use non-interactive backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(_here, "Assignment_1.ipynb")
RESULTS_PATH = os.path.join(_here, "Assignment_1_results.json")

def main():
    with open(NOTEBOOK_PATH, encoding="utf-8") as f:
        nb = json.load(f)

    globals_dict = {
        "np": np,
        "plt": plt,
        "matplotlib": matplotlib,
    }
    exec_count = 0

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            cell["outputs"] = []
            cell["execution_count"] = None
            continue
        exec_count += 1
        cell["execution_count"] = exec_count
        stdout_capture = StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture
        try:
            exec(compile(source, f"<cell-{exec_count}>", "exec"), globals_dict)
        except Exception as e:
            sys.stdout = old_stdout
            cell["outputs"] = [
                {
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [],
                }
            ]
            raise
        sys.stdout = old_stdout
        out_text = stdout_capture.getvalue()
        outputs = []
        if out_text.strip():
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": out_text.splitlines(keepends=True) if out_text else [],
            })
        cell["outputs"] = outputs

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    # Capture results for the report (from last execution state)
    hist_and = globals_dict.get("hist_and")
    hist_or = globals_dict.get("hist_or")
    mse_and = globals_dict.get("mse_and")
    mse_or = globals_dict.get("mse_or")
    hist_xor = globals_dict.get("hist_xor")
    mse_xor = globals_dict.get("mse_xor")
    ada_xor = globals_dict.get("ada_xor")
    perc_and = globals_dict.get("perc_and")
    perc_or = globals_dict.get("perc_or")
    ada_and = globals_dict.get("ada_and")
    ada_or = globals_dict.get("ada_or")

    results = {
        "perceptron_and_epochs": int(len(hist_and)) if hist_and is not None else None,
        "perceptron_or_epochs": int(len(hist_or)) if hist_or is not None else None,
        "perceptron_and_converged": bool(hist_and is not None and len(hist_and) > 0 and hist_and[-1] == 0),
        "perceptron_or_converged": bool(hist_or is not None and len(hist_or) > 0 and hist_or[-1] == 0),
        "adaline_and_epochs": int(len(mse_and)) if mse_and is not None else None,
        "adaline_and_final_mse": float(mse_and[-1]) if mse_and is not None else None,
        "adaline_or_epochs": int(len(mse_or)) if mse_or is not None else None,
        "adaline_or_final_mse": float(mse_or[-1]) if mse_or is not None else None,
        "perceptron_xor_last_misclass": int(hist_xor[-1]) if hist_xor is not None else None,
        "adaline_xor_final_mse": float(mse_xor[-1]) if mse_xor is not None else None,
        "adaline_xor_weights": [float(x) for x in ada_xor.w] if ada_xor is not None else None,
        "adaline_xor_decision_boundary_eq": (
            f"{ada_xor.w[0]:.4f} + {ada_xor.w[1]:.4f}*x1 + {ada_xor.w[2]:.4f}*x2 = 0"
            if ada_xor is not None else None
        ),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Notebook executed and saved. Results written to", RESULTS_PATH)
    for k, v in results.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
