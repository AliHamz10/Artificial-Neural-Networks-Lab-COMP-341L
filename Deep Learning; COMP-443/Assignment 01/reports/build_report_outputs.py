from __future__ import annotations

import html
import re
import textwrap
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


MD_NAME = "COMP443_Assignment01_Report.md"
HTML_NAME = "COMP443_Assignment01_Report.html"
PDF_NAME = "COMP443_Assignment01_Report.pdf"


def build_html(md_path: Path, out_html: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    running_title = "CLASSICAL VS DEEP MODELS ON FASHION-MNIST"

    def normalize_list_spacing(src: str) -> str:
        lines = src.splitlines()
        out: list[str] = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                out.append(line)
                continue
            if not in_code and re.match(r"^(\- |\* |\d+\.\s)", stripped):
                leading_spaces = len(line) - len(line.lstrip(" "))
                prev = out[-1] if out else ""
                prev_stripped = prev.strip()
                prev_is_listish = bool(
                    re.match(r"^(\- |\* |\d+\.\s|#|>|```|\|)", prev_stripped)
                )
                # Insert a blank line only for top-level list blocks. Nested list items
                # should remain attached to their parent bullet/numbering item.
                if leading_spaces == 0 and prev_stripped and not prev_is_listish:
                    out.append("")
            out.append(line)
        return "\n".join(out)

    title_placeholder = "APA_TITLE_BLOCK_TOKEN_9F6A"
    title_html = ""
    title_match = re.search(r"<div align=\"center\">\s*(.*?)\s*</div>", md_text, flags=re.DOTALL)
    if title_match:
        inner_md = title_match.group(1).strip()
        inner_html = markdown.markdown(inner_md, extensions=["tables", "fenced_code"])
        title_html = f'<section class="apa-title-page"><div class="apa-title-block">{inner_html}</div></section>'
        md_text = md_text[: title_match.start()] + title_placeholder + md_text[title_match.end() :]

    md_text = normalize_list_spacing(md_text)
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    if title_html:
        body = body.replace(f"<p>{title_placeholder}</p>", title_html).replace(title_placeholder, title_html)

    def wrap_image_paragraph(match: re.Match[str]) -> str:
        inner = match.group(1)
        alt_match = re.search(r'alt="([^"]*)"', inner)
        caption = html.unescape(alt_match.group(1)).strip() if alt_match else ""
        caption_html = (
            f'<figcaption><span class="fig-label"></span><span class="fig-title">{html.escape(caption)}</span></figcaption>'
            if caption
            else ""
        )
        return f'<figure class="apa-figure">{inner}{caption_html}</figure>'

    body = re.sub(r"<p>\s*(<img[^>]+>)\s*</p>", wrap_image_paragraph, body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(
        r"(<h2>Abstract</h2>.*?)(<hr\s*/?>)",
        r'<section class="apa-abstract-page">\1</section>\2',
        body,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )

    css = """
    :root {
      --ink: #111111;
      --muted: #444444;
      --paper: #ffffff;
      --rule: #c8c8c8;
      --codebg: #f7f7f7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #efefef;
      color: var(--ink);
      font-family: "Times New Roman", Times, serif;
      font-size: 12pt;
      line-height: 2;
      text-align: left;
      counter-reset: figure;
    }
    .apa-sheet {
      width: 8.5in;
      max-width: 100%;
      margin: 24px auto 40px;
      position: relative;
    }
    .page {
      background: var(--paper);
      border: 1px solid #d4d4d4;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
      padding: 1.2in 1in 0.95in;
      min-height: 11in;
    }
    .apa-running-header,
    .apa-running-footer {
      position: absolute;
      left: 1in;
      right: 1in;
      color: #444;
      font-size: 10pt;
      line-height: 1.2;
      background: transparent;
      pointer-events: none;
      z-index: 2;
    }
    .apa-running-header {
      top: 0.42in;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #ececec;
      padding-bottom: 0.06in;
      letter-spacing: 0.02em;
    }
    .apa-running-title {
      font-variant: small-caps;
      letter-spacing: 0.04em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 85%;
    }
    .apa-running-page {
      min-width: 2ch;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .apa-running-page::before {
      content: "";
    }
    .apa-running-footer {
      bottom: 0.38in;
      text-align: center;
      border-top: 1px solid #ececec;
      padding-top: 0.06in;
      color: #666;
    }
    .apa-title-page {
      min-height: 9.0in;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      text-align: center;
      padding-top: 2.1in;
      padding-bottom: 1.2in;
      page-break-after: always;
      break-after: page;
    }
    .apa-title-block {
      width: 100%;
      max-width: 6.2in;
    }
    .apa-title-block p {
      text-indent: 0;
      margin: 0;
    }
    .apa-title-block h1 {
      font-size: 12pt;
      text-align: center;
      margin: 0 0 0.4in 0;
      line-height: 2;
      font-weight: 700;
    }
    .apa-title-block h2,
    .apa-title-block h3 {
      font-size: 12pt;
      text-align: center;
      margin: 0;
      line-height: 2;
      font-weight: 700;
    }
    .apa-title-block h3 {
      font-weight: 400;
    }
    h1, h2, h3, h4 {
      color: var(--ink);
      line-height: 2;
      margin-top: 0.4rem;
      margin-bottom: 0;
      font-weight: 700;
    }
    h1 {
      font-size: 12pt;
      text-align: center;
    }
    h2 {
      font-size: 12pt;
      text-align: center;
      margin-top: 0.8rem;
      margin-bottom: 0;
    }
    h3 {
      font-size: 12pt;
      text-align: left;
      margin-top: 0.5rem;
      font-weight: 700;
    }
    h4 {
      font-size: 12pt;
      font-style: italic;
      text-align: left;
      font-weight: 700;
    }
    p {
      margin: 0;
      color: var(--ink);
      text-indent: 0.5in;
    }
    h1 + p, h2 + p, h3 + p, h4 + p,
    hr + p, table + p, figure + p, pre + p,
    blockquote p, li p, td p, th p,
    .apa-title-block p, .apa-abstract-page p:first-of-type {
      text-indent: 0;
    }
    ul, ol {
      margin: 0;
      padding-left: 0.5in;
    }
    li {
      margin: 0;
      color: var(--ink);
    }
    ul ul, ul ol, ol ul, ol ol {
      padding-left: 0.32in;
      margin-top: 0;
    }
    code {
      background: var(--codebg);
      padding: 0 0.15rem;
      font-size: 0.92em;
      font-family: "Courier New", Courier, monospace;
      border: 1px solid #e5e5e5;
    }
    pre {
      background: #fafafa;
      color: #111;
      padding: 10px 12px;
      margin: 0.25rem 0;
      overflow-x: auto;
      border: 1px solid #d6d6d6;
    }
    pre code { background: transparent; padding: 0; color: inherit; }
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 0.35rem 0 0.55rem;
      font-size: 11pt;
      line-height: 1.5;
      border-top: 1.4px solid #222;
      border-bottom: 1.4px solid #222;
    }
    th, td {
      border: none;
      padding: 6px 8px;
      vertical-align: top;
      text-align: left;
    }
    th {
      background: transparent;
      text-align: left;
      font-weight: 700;
      border-bottom: 1px solid #222;
    }
    tbody td {
      border-bottom: 1px solid #e1e1e1;
    }
    tbody tr:last-child td {
      border-bottom: none;
    }
    .apa-figure {
      display: block;
      margin: 0.35rem auto 0.5rem;
      text-align: center;
      page-break-inside: avoid;
      break-inside: avoid;
      counter-increment: figure;
    }
    .apa-figure img {
      display: block;
      max-width: 100%;
      margin: 0 auto 0.2rem;
      border: 1px solid #d4d4d4;
      padding: 2px;
      background: white;
    }
    .apa-figure figcaption {
      font-size: 11pt;
      line-height: 1.5;
      color: var(--muted);
      text-align: left;
      margin-top: 0.15rem;
    }
    .apa-figure .fig-label::before {
      content: "Figure " counter(figure) ". ";
      font-weight: 700;
      font-style: normal;
      color: #222;
    }
    .apa-figure .fig-title {
      font-style: italic;
      color: #333;
    }
    hr {
      border: 0;
      height: 0;
      margin: 0.35rem 0;
    }
    blockquote {
      margin: 0.25rem 0;
      padding: 0 0 0 0.4in;
      border-left: 2px solid #cfcfcf;
      background: transparent;
      color: var(--muted);
    }
    .apa-abstract-page {
      page-break-after: always;
      break-after: page;
      min-height: 8.8in;
    }
    .apa-abstract-page h2 {
      margin-top: 0;
    }
    .page > p:first-of-type {
      text-indent: 0;
    }
    @media print {
      @page {
        size: letter;
        margin: 1in;
      }
      body {
        background: white;
        margin: 0;
      }
      .apa-sheet {
        width: auto;
        max-width: none;
        margin: 0;
      }
      .apa-running-header,
      .apa-running-footer {
        position: fixed;
        left: 1in;
        right: 1in;
      }
      .apa-running-header {
        top: 0.4in;
        border-bottom: none;
        padding-bottom: 0;
      }
      .apa-running-page::before {
        content: counter(page);
      }
      .apa-running-footer {
        bottom: 0.35in;
        border-top: none;
        padding-top: 0;
      }
      .page {
        box-shadow: none;
        border: none;
        margin: 0;
        width: auto;
        padding: 0;
        min-height: auto;
      }
      .apa-title-page {
        min-height: auto;
        display: block;
        padding: 0;
        page-break-after: always;
        break-after: page;
      }
      .apa-figure img { max-height: 8.2in; object-fit: contain; }
    }
    """

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>COMP443 Assignment 01 Report</title>
  <style>{css}</style>
</head>
<body>
  <div class="apa-sheet">
    <header class="apa-running-header" aria-hidden="true">
      <div class="apa-running-title">{running_title}</div>
      <div class="apa-running-page"></div>
    </header>
    <footer class="apa-running-footer" aria-hidden="true">
      COMP-443 Deep Learning | Assignment 01 Report
    </footer>
    <main class="page" role="document">
      {body}
    </main>
  </div>
</body>
</html>
"""
    out_html.write_text(html_doc, encoding="utf-8")


class PdfComposer:
    def __init__(self, pdf: PdfPages) -> None:
        self.pdf = pdf
        self.fig = None
        self.ax = None
        self.y = 0.95
        self.page_no = 0
        self.new_page()

    def new_page(self) -> None:
        if self.fig is not None:
            self.pdf.savefig(self.fig, bbox_inches="tight")
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        self.ax = self.fig.add_axes([0.06, 0.04, 0.88, 0.92])
        self.ax.axis("off")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.y = 0.97
        self.page_no += 1
        self.ax.text(
            0.995, 0.01, str(self.page_no), ha="right", va="bottom", fontsize=8, color="#666"
        )

    def ensure_space(self, needed: float) -> None:
        if self.y - needed < 0.04:
            self.new_page()

    def add_text(self, text: str, *, size: int = 10, weight: str = "normal", color: str = "#111827", mono: bool = False) -> None:
        if not text:
            self.add_blank(0.010)
            return
        wrap_width = max(18, int(100 - (size - 10) * 3))
        lines = []
        for raw in text.splitlines():
            if not raw.strip():
                lines.append("")
                continue
            if mono:
                wrapped = textwrap.wrap(raw, width=95, replace_whitespace=False, drop_whitespace=False) or [raw]
            else:
                wrapped = textwrap.wrap(raw, width=wrap_width)
            lines.extend(wrapped)
        line_h = 0.015 + (size - 10) * 0.0008
        needed = max(line_h, line_h * max(1, len(lines)) + 0.003)
        self.ensure_space(needed)
        family = "DejaVu Sans Mono" if mono else "DejaVu Serif"
        for line in lines:
            self.ax.text(
                0.02,
                self.y,
                line,
                ha="left",
                va="top",
                fontsize=size,
                fontweight=weight,
                color=color,
                family=family,
            )
            self.y -= line_h
        self.y -= 0.002

    def add_blank(self, h: float = 0.012) -> None:
        self.ensure_space(h)
        self.y -= h

    def add_rule(self) -> None:
        self.ensure_space(0.02)
        self.ax.plot([0.02, 0.98], [self.y, self.y], color="#d1d5db", lw=0.8)
        self.y -= 0.018

    def add_image_page(self, img_path: Path, caption: str | None = None) -> None:
        self.new_page()
        if caption:
            self.ax.text(0.02, 0.96, caption, ha="left", va="top", fontsize=12, fontweight="bold", family="DejaVu Serif")
        try:
            img = mpimg.imread(img_path)
            img_ax = self.fig.add_axes([0.08, 0.10, 0.84, 0.80])
            img_ax.imshow(img)
            img_ax.axis("off")
            if caption:
                img_ax.set_title(caption, fontsize=11, pad=8)
        except Exception as e:  # noqa: BLE001
            self.ax.text(0.02, 0.90, f"[Could not render image: {img_path.name}]", color="#b91c1c", fontsize=11)
            self.ax.text(0.02, 0.86, str(e), color="#7f1d1d", fontsize=9)

    def finish(self) -> None:
        if self.fig is not None:
            self.pdf.savefig(self.fig, bbox_inches="tight")
            plt.close(self.fig)
            self.fig = None


def build_pdf(md_path: Path, out_pdf: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    base_dir = md_path.parent

    with PdfPages(out_pdf) as pdf:
        c = PdfComposer(pdf)
        in_code = False

        for raw in lines:
            line = raw.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith("<div") or stripped.startswith("</div>"):
                continue

            if stripped.startswith("```"):
                in_code = not in_code
                c.add_blank(0.006)
                continue

            if in_code:
                c.add_text(line, size=9, mono=True, color="#111827")
                continue

            if stripped == "---":
                c.add_rule()
                continue

            img_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if img_match:
                caption = img_match.group(1).strip() or None
                img_rel = img_match.group(2).strip()
                img_path = (base_dir / img_rel).resolve()
                c.add_image_page(img_path, caption)
                continue

            if not stripped:
                c.add_blank(0.010)
                continue

            if stripped.startswith("# "):
                c.add_text(stripped[2:].strip(), size=18, weight="bold", color="#0f172a")
                continue
            if stripped.startswith("## "):
                c.add_blank(0.004)
                c.add_text(stripped[3:].strip(), size=14, weight="bold", color="#0f3d62")
                continue
            if stripped.startswith("### "):
                c.add_text(stripped[4:].strip(), size=12, weight="bold", color="#134e4a")
                continue

            if re.match(r"^\d+\.\s", stripped):
                c.add_text(stripped, size=10)
                continue

            if stripped.startswith("- "):
                c.add_text("• " + stripped[2:], size=10)
                continue

            if stripped.startswith("|"):
                c.add_text(line, size=8, mono=True, color="#1f2937")
                continue

            # Remove inline HTML tags if present
            clean = re.sub(r"<[^>]+>", "", line)
            c.add_text(html.unescape(clean), size=10)

        c.finish()


def main() -> None:
    here = Path(__file__).resolve().parent
    md_path = here / MD_NAME
    out_html = here / HTML_NAME
    out_pdf = here / PDF_NAME

    build_html(md_path, out_html)
    build_pdf(md_path, out_pdf)

    print(f"HTML written to: {out_html}")
    print(f"PDF written to:  {out_pdf}")


if __name__ == "__main__":
    main()
