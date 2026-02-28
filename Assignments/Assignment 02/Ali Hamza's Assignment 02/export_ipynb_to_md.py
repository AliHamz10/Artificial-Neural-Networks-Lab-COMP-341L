#!/usr/bin/env python3
"""Export an executed Jupyter notebook to Markdown with extracted output images.

Usage:
  python3 export_ipynb_to_md.py \
    --input Assignment_B23F0063AI106.ipynb \
    --output Assignment_B23F0063AI106_report.md \
    --assets-dir Assignment_B23F0063AI106_report_assets
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Iterable


def _lines_to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(value)
    return str(value)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_base64_blob(data_obj, out_path: Path) -> None:
    raw = _lines_to_text(data_obj)
    out_path.write_bytes(base64.b64decode(raw))


def _write_text_block(lines: list[str], text: str) -> None:
    if not text:
        return
    lines.append("```text\n")
    lines.append(text)
    if not text.endswith("\n"):
        lines.append("\n")
    lines.append("```\n\n")


def _render_markdown_cell(lines: list[str], cell: dict) -> None:
    text = _lines_to_text(cell.get("source", ""))
    lines.append(text)
    if not text.endswith("\n"):
        lines.append("\n")
    lines.append("\n")


def _render_code_cell(
    lines: list[str],
    cell: dict,
    cell_idx: int,
    assets_dir: Path,
    assets_rel: str,
) -> tuple[int, int]:
    text_outputs = 0
    image_outputs = 0

    code = _lines_to_text(cell.get("source", ""))
    lines.append("```python\n")
    lines.append(code)
    if not code.endswith("\n"):
        lines.append("\n")
    lines.append("```\n\n")

    outputs = cell.get("outputs", []) or []
    if not outputs:
        return text_outputs, image_outputs

    lines.append(f"**Outputs (Cell {cell_idx})**\n\n")

    for out_idx, out in enumerate(outputs, start=1):
        out_type = out.get("output_type", "")

        if out_type == "stream":
            text = _lines_to_text(out.get("text", ""))
            _write_text_block(lines, text)
            if text.strip():
                text_outputs += 1
            continue

        if out_type == "error":
            traceback = _lines_to_text(out.get("traceback", ""))
            _write_text_block(lines, traceback)
            if traceback.strip():
                text_outputs += 1
            continue

        data = out.get("data", {}) or {}

        text_plain = _lines_to_text(data.get("text/plain", ""))
        if text_plain.strip():
            _write_text_block(lines, text_plain)
            text_outputs += 1

        if "image/png" in data:
            fname = f"cell_{cell_idx:03d}_output_{out_idx:02d}.png"
            out_path = assets_dir / fname
            _save_base64_blob(data["image/png"], out_path)
            lines.append(f"![Cell {cell_idx} Output {out_idx}]({assets_rel}/{fname})\n\n")
            image_outputs += 1

        if "image/jpeg" in data:
            fname = f"cell_{cell_idx:03d}_output_{out_idx:02d}.jpg"
            out_path = assets_dir / fname
            _save_base64_blob(data["image/jpeg"], out_path)
            lines.append(f"![Cell {cell_idx} Output {out_idx}]({assets_rel}/{fname})\n\n")
            image_outputs += 1

    lines.append("\n")
    return text_outputs, image_outputs


def export_notebook(input_path: Path, output_path: Path, assets_dir: Path) -> dict[str, int]:
    nb = json.loads(input_path.read_text(encoding="utf-8"))

    _ensure_dir(assets_dir)
    assets_rel = assets_dir.name

    lines: list[str] = []
    lines.append(f"# Notebook Report: {input_path.name}\n\n")
    lines.append("Generated from executed notebook outputs.\n\n")
    lines.append("---\n\n")

    n_cells = 0
    n_code = 0
    n_md = 0
    total_text_outputs = 0
    total_image_outputs = 0

    for idx, cell in enumerate(nb.get("cells", []), start=1):
        n_cells += 1
        cell_type = cell.get("cell_type", "")
        lines.append(f"<!-- Cell {idx} ({cell_type}) -->\n\n")

        if cell_type == "markdown":
            n_md += 1
            _render_markdown_cell(lines, cell)
        elif cell_type == "code":
            n_code += 1
            txt, img = _render_code_cell(lines, cell, idx, assets_dir, assets_rel)
            total_text_outputs += txt
            total_image_outputs += img
        else:
            raw = _lines_to_text(cell.get("source", ""))
            if raw:
                lines.append("```text\n")
                lines.append(raw)
                if not raw.endswith("\n"):
                    lines.append("\n")
                lines.append("```\n\n")

    output_path.write_text("".join(lines), encoding="utf-8")

    return {
        "cells_total": n_cells,
        "markdown_cells": n_md,
        "code_cells": n_code,
        "text_outputs": total_text_outputs,
        "image_outputs": total_image_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export executed notebook to Markdown with images.")
    parser.add_argument("--input", required=True, help="Path to input .ipynb notebook")
    parser.add_argument("--output", required=True, help="Path to output .md file")
    parser.add_argument("--assets-dir", required=True, help="Directory to store extracted images")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    assets_dir = Path(args.assets_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input notebook not found: {input_path}")

    stats = export_notebook(input_path, output_path, assets_dir)

    print(f"Input notebook : {input_path}")
    print(f"Output markdown: {output_path}")
    print(f"Assets dir     : {assets_dir}")
    print("---")
    print(
        "Summary: "
        f"cells={stats['cells_total']}, "
        f"markdown={stats['markdown_cells']}, "
        f"code={stats['code_cells']}, "
        f"text_outputs={stats['text_outputs']}, "
        f"image_outputs={stats['image_outputs']}"
    )


if __name__ == "__main__":
    main()
