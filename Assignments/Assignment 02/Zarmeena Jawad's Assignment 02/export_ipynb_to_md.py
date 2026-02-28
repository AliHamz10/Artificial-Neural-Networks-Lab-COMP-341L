#!/usr/bin/env python3
"""Export an executed Jupyter notebook to Markdown with extracted images.

Example:
python3 export_ipynb_to_md.py \
  --input Assignment_B23F0115AI125.ipynb \
  --output Assignment_B23F0115AI125_report.md \
  --assets-dir Assignment_B23F0115AI125_report_assets
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(value)
    return str(value)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_b64(data_obj, path: Path) -> None:
    path.write_bytes(base64.b64decode(to_text(data_obj)))


def append_text_block(out: list[str], text: str) -> None:
    if not text:
        return
    out.append("```text\n")
    out.append(text)
    if not text.endswith("\n"):
        out.append("\n")
    out.append("```\n\n")


def export_notebook(input_path: Path, output_path: Path, assets_dir: Path) -> dict[str, int]:
    nb = json.loads(input_path.read_text(encoding="utf-8"))
    ensure_dir(assets_dir)

    out: list[str] = []
    rel_assets = assets_dir.name

    out.append(f"# Notebook Report: {input_path.name}\n\n")
    out.append("Generated from executed notebook outputs.\n\n")
    out.append("---\n\n")

    cells_total = 0
    code_cells = 0
    markdown_cells = 0
    text_outputs = 0
    image_outputs = 0

    for idx, cell in enumerate(nb.get("cells", []), start=1):
        cells_total += 1
        ctype = cell.get("cell_type", "")
        out.append(f"<!-- Cell {idx} ({ctype}) -->\n\n")

        if ctype == "markdown":
            markdown_cells += 1
            content = to_text(cell.get("source", ""))
            out.append(content)
            if not content.endswith("\n"):
                out.append("\n")
            out.append("\n")
            continue

        if ctype != "code":
            raw = to_text(cell.get("source", ""))
            if raw:
                out.append("```text\n")
                out.append(raw)
                if not raw.endswith("\n"):
                    out.append("\n")
                out.append("```\n\n")
            continue

        code_cells += 1
        src = to_text(cell.get("source", ""))
        out.append("```python\n")
        out.append(src)
        if not src.endswith("\n"):
            out.append("\n")
        out.append("```\n\n")

        outputs = cell.get("outputs", []) or []
        if not outputs:
            continue

        out.append(f"**Outputs (Cell {idx})**\n\n")

        for j, entry in enumerate(outputs, start=1):
            otype = entry.get("output_type", "")

            if otype == "stream":
                txt = to_text(entry.get("text", ""))
                append_text_block(out, txt)
                if txt.strip():
                    text_outputs += 1
                continue

            if otype == "error":
                txt = to_text(entry.get("traceback", ""))
                append_text_block(out, txt)
                if txt.strip():
                    text_outputs += 1
                continue

            data = entry.get("data", {}) or {}

            plain = to_text(data.get("text/plain", ""))
            if plain.strip():
                append_text_block(out, plain)
                text_outputs += 1

            if "image/png" in data:
                fname = f"cell_{idx:03d}_output_{j:02d}.png"
                save_b64(data["image/png"], assets_dir / fname)
                out.append(f"![Cell {idx} Output {j}]({rel_assets}/{fname})\n\n")
                image_outputs += 1

            if "image/jpeg" in data:
                fname = f"cell_{idx:03d}_output_{j:02d}.jpg"
                save_b64(data["image/jpeg"], assets_dir / fname)
                out.append(f"![Cell {idx} Output {j}]({rel_assets}/{fname})\n\n")
                image_outputs += 1

        out.append("\n")

    output_path.write_text("".join(out), encoding="utf-8")

    return {
        "cells_total": cells_total,
        "markdown_cells": markdown_cells,
        "code_cells": code_cells,
        "text_outputs": text_outputs,
        "image_outputs": image_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export executed notebook to markdown with assets")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--assets-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    assets_dir = Path(args.assets_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Notebook not found: {input_path}")

    stats = export_notebook(input_path, output_path, assets_dir)

    print("Input notebook :", input_path)
    print("Output markdown:", output_path)
    print("Assets dir     :", assets_dir)
    print("---")
    print(
        f"Summary: cells={stats['cells_total']}, markdown={stats['markdown_cells']}, "
        f"code={stats['code_cells']}, text_outputs={stats['text_outputs']}, image_outputs={stats['image_outputs']}"
    )


if __name__ == "__main__":
    main()
