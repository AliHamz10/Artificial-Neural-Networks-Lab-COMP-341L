# Run ANN_Assignment1.ipynb in order and save outputs + results for the report.
# From repo root: python "Assignments/Zarmeena Jawad's Assignments/execute_notebook.py"

import json
import os
import sys
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
NB_PATH = os.path.join(BASE, "ANN_Assignment1.ipynb")
OUT_PATH = os.path.join(BASE, "report_data.json")

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

g = {"np": np, "plt": plt}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell.get("source", []))
    if not src.strip():
        continue
    cap = StringIO()
    old_stdout = sys.stdout
    sys.stdout = cap
    try:
        exec(compile(src, "<cell>", "exec"), g)
    finally:
        sys.stdout = old_stdout
    out = cap.getvalue()
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": out.splitlines(keepends=True)}] if out.strip() else []
    cell["execution_count"] = cell.get("execution_count") or 1

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

# Persist key results for report
data = {}
if "err_and" in g:
    data["perceptron_and_epochs"] = int(len(g["err_and"]))
if "err_or" in g:
    data["perceptron_or_epochs"] = int(len(g["err_or"]))
if "mse_and" in g:
    data["adaline_and_epochs"] = int(len(g["mse_and"]))
    data["adaline_and_final_mse"] = float(g["mse_and"][-1])
if "mse_or" in g:
    data["adaline_or_epochs"] = int(len(g["mse_or"]))
    data["adaline_or_final_mse"] = float(g["mse_or"][-1])
if "err_xor" in g:
    data["perceptron_xor_last_err"] = int(g["err_xor"][-1])
if "mse_xor" in g:
    data["adaline_xor_final_mse"] = float(g["mse_xor"][-1])
if "lms_xor" in g:
    w = g["lms_xor"].weights
    data["adaline_xor_weights"] = w.tolist()
    data["adaline_xor_boundary_eq"] = "{:.4f} + {:.4f}*x1 + {:.4f}*x2 = 0".format(w[0], w[1], w[2])

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
print("Notebook run complete. Results in", OUT_PATH)
