#!/usr/bin/env python3
"""Convert assignment markdown report into polished academic HTML."""

from __future__ import annotations

import datetime as dt
import html
import re
from pathlib import Path


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    s = re.sub(r"\s+", "-", s)
    return s or "section"


def md_inline(text: str) -> str:
    text = html.escape(text)

    # Inline code
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # Bold then italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)

    # Links [text](url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    return text


def parse_table(lines: list[str], i: int) -> tuple[str, int] | tuple[None, int]:
    if i + 1 >= len(lines):
        return None, i

    head = lines[i].rstrip("\n")
    sep = lines[i + 1].rstrip("\n")

    if "|" not in head or "|" not in sep:
        return None, i

    if not re.match(r"^\s*\|?\s*[:\- ]+\|", sep):
        return None, i

    def split_row(row: str) -> list[str]:
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [c.strip() for c in row.split("|")]

    headers = split_row(head)
    rows = []

    j = i + 2
    while j < len(lines):
        row = lines[j].rstrip("\n")
        if "|" not in row or not row.strip():
            break
        rows.append(split_row(row))
        j += 1

    out = ["<div class=\"table-wrap\">\n", "<table>\n", "<thead><tr>"]
    for h in headers:
        out.append(f"<th>{md_inline(h)}</th>")
    out.append("</tr></thead>\n<tbody>\n")

    for r in rows:
        out.append("<tr>")
        for c in r:
            out.append(f"<td>{md_inline(c)}</td>")
        out.append("</tr>\n")

    out.append("</tbody></table>\n</div>\n")
    return "".join(out), j


def parse_markdown_to_html(md_text: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = md_text.splitlines(keepends=True)
    i = 0
    out: list[str] = []
    toc: list[tuple[int, str, str]] = []

    in_code = False
    code_lang = ""
    code_buf: list[str] = []
    list_open = False
    para_buf: list[str] = []

    def flush_para() -> None:
        nonlocal para_buf
        if para_buf:
            text = " ".join(p.strip() for p in para_buf if p.strip())
            out.append(f"<p>{md_inline(text)}</p>\n")
            para_buf = []

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            out.append("</ul>\n")
            list_open = False

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip("\n")
        stripped = line.strip()

        # fenced code blocks
        if stripped.startswith("```"):
            flush_para()
            close_list()
            if not in_code:
                in_code = True
                code_lang = stripped[3:].strip()
                code_buf = []
            else:
                cls = f' class="language-{html.escape(code_lang)}"' if code_lang else ""
                code = html.escape("\n".join(code_buf))
                out.append(f"<pre><code{cls}>{code}</code></pre>\n")
                in_code = False
                code_lang = ""
                code_buf = []
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # blank
        if not stripped:
            flush_para()
            close_list()
            i += 1
            continue

        # horizontal rule
        if stripped == "---":
            flush_para()
            close_list()
            out.append("<hr/>\n")
            i += 1
            continue

        # display math block \[ ... \]
        if stripped == r"\[":
            flush_para()
            close_list()
            math_lines = [line]
            j = i + 1
            while j < len(lines):
                ml = lines[j].rstrip("\n")
                math_lines.append(ml)
                if ml.strip() == r"\]":
                    break
                j += 1
            out.append("<div class=\"display-math\">\n")
            out.append("\n".join(math_lines))
            out.append("\n</div>\n")
            i = j + 1
            continue

        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush_para()
            close_list()
            level = len(m.group(1))
            title = m.group(2).strip()
            hid = slugify(title)
            toc.append((level, title, hid))
            out.append(f"<h{level} id=\"{hid}\">{md_inline(title)}</h{level}>\n")
            i += 1
            continue

        # image on its own line
        im = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if im:
            flush_para()
            close_list()
            alt = html.escape(im.group(1))
            src = html.escape(im.group(2))
            out.append("<figure>\n")
            out.append(f"<img src=\"{src}\" alt=\"{alt}\"/>\n")
            if alt:
                out.append(f"<figcaption>{alt}</figcaption>\n")
            out.append("</figure>\n")
            i += 1
            continue

        # table
        table_html, new_i = parse_table(lines, i)
        if table_html:
            flush_para()
            close_list()
            out.append(table_html)
            i = new_i
            continue

        # unordered list item
        lm = re.match(r"^[-*]\s+(.*)$", stripped)
        if lm:
            flush_para()
            if not list_open:
                out.append("<ul>\n")
                list_open = True
            out.append(f"<li>{md_inline(lm.group(1).strip())}</li>\n")
            i += 1
            continue

        # plain paragraph line
        para_buf.append(line)
        i += 1

    flush_para()
    close_list()

    return "".join(out), toc


def build_toc(toc: list[tuple[int, str, str]]) -> str:
    # Include level 2-4 in TOC for readability.
    filtered = [(lvl, t, hid) for lvl, t, hid in toc if 2 <= lvl <= 4]
    if not filtered:
        return ""

    out = ["<nav class=\"toc\">\n", "<h2>Table Of Contents</h2>\n", "<ol>\n"]
    for lvl, title, hid in filtered:
        cls = f"toc-l{lvl}"
        out.append(f"<li class=\"{cls}\"><a href=\"#{hid}\">{html.escape(title)}</a></li>\n")
    out.append("</ol>\n</nav>\n")
    return "".join(out)


def extract_cover_meta(preface: str) -> dict[str, str]:
    lines = [l.strip() for l in preface.splitlines() if l.strip()]

    title = "Assignment Report"
    subtitle = ""
    meta: dict[str, str] = {}

    for line in lines:
        if line.startswith("# ") and title == "Assignment Report":
            title = line[2:].strip()
            continue
        if line.startswith("## ") and not subtitle:
            subtitle = line[3:].strip()
            continue

        m = re.match(r"^\*\*(.+?):\*\*\s*(.+)$", line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            meta[key] = val

    return {
        "title": title,
        "subtitle": subtitle,
        "student": meta.get("Student Name", "Ali Hamza"),
        "roll": meta.get("Roll Number", "B23F0063AI106"),
        "course": meta.get("Course", "Artificial Neural Network (COMP-341)"),
        "instructor": meta.get("Instructor", "Dr. Abid Ali"),
    }


def build_html_document(cover: dict[str, str], toc_html: str, body_html: str) -> str:
    today = dt.date.today().strftime("%d %B %Y")

    css = r'''
:root {
  --ink: #16243a;
  --muted: #4c5d75;
  --line: #d8dee9;
  --accent: #234a84;
  --bg: #f7f9fc;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0;
  font-family: "Georgia", "Times New Roman", serif;
  color: var(--ink);
  background: white;
  line-height: 1.55;
}
.page {
  width: min(980px, 92vw);
  margin: 28px auto 56px;
}
.cover {
  min-height: 90vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  border: 1px solid var(--line);
  padding: 56px 64px;
  background: linear-gradient(180deg, #ffffff 0%, var(--bg) 100%);
}
.cover .inst {
  letter-spacing: 0.06em;
  font-size: 0.85rem;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 18px;
}
.cover h1 {
  margin: 0 0 8px;
  font-size: 2.1rem;
  color: var(--accent);
  line-height: 1.2;
}
.cover h2 {
  margin: 0 0 30px;
  font-size: 1.15rem;
  font-weight: 500;
  color: var(--muted);
}
.meta-grid {
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 6px 12px;
  margin: 20px 0 0;
  font-size: 1rem;
}
.meta-grid .k { color: var(--muted); }
.meta-grid .v { font-weight: 600; }
.cover .date {
  margin-top: 34px;
  color: var(--muted);
  font-size: 0.95rem;
}
.section {
  margin-top: 32px;
}
hr {
  border: none;
  border-top: 1px solid var(--line);
  margin: 26px 0;
}
h1,h2,h3,h4,h5,h6 {
  color: var(--accent);
  line-height: 1.28;
  margin-top: 1.3em;
  margin-bottom: 0.55em;
}
h1 { font-size: 1.95rem; }
h2 { font-size: 1.45rem; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
h3 { font-size: 1.15rem; }
h4 { font-size: 1.03rem; }
p { margin: 0.5em 0 0.85em; }
ul { margin: 0.45em 0 1em 1.2em; }
li { margin: 0.2em 0; }
code {
  font-family: "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: 0.9em;
  background: #eef3fa;
  border: 1px solid #dce5f2;
  padding: 1px 4px;
  border-radius: 4px;
}
pre {
  background: #0f172a;
  color: #e6edf7;
  border-radius: 8px;
  padding: 14px;
  overflow-x: auto;
  border: 1px solid #202c44;
}
pre code {
  background: transparent;
  border: none;
  padding: 0;
  color: inherit;
}
figure {
  margin: 18px auto 22px;
  padding: 12px;
  border: 1px solid var(--line);
  background: #fff;
}
figure img {
  width: 100%;
  height: auto;
  display: block;
}
figcaption {
  margin-top: 8px;
  text-align: center;
  color: var(--muted);
  font-size: 0.92rem;
}
.table-wrap {
  overflow-x: auto;
  margin: 14px 0 18px;
  break-inside: avoid;
  page-break-inside: avoid;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
  table-layout: fixed;
}
th, td {
  border: 1px solid var(--line);
  padding: 8px 10px;
  vertical-align: top;
  word-wrap: break-word;
  overflow-wrap: anywhere;
}
th {
  background: #edf3fb;
  color: var(--accent);
  text-align: left;
}
.toc {
  border: 1px solid var(--line);
  background: #fafcff;
  padding: 16px 20px;
  margin: 26px 0 24px;
}
.toc h2 {
  border: none;
  margin-top: 0;
  margin-bottom: 10px;
  padding-bottom: 0;
  font-size: 1.2rem;
}
.toc ol {
  margin: 0;
  padding-left: 1.2em;
}
.toc li { margin: 0.3em 0; }
.toc .toc-l3 { margin-left: 1.1em; }
.toc .toc-l4 { margin-left: 2.1em; }
.toc a {
  color: #1d3f72;
  text-decoration: none;
}
.toc a:hover { text-decoration: underline; }
.display-math {
  margin: 12px 0;
  padding: 6px 0;
}
@media print {
  @page {
    size: A4;
    margin: 10mm 11mm;
  }
  body { background: white; }
  .page { width: 100%; margin: 0; }
  .section { margin-top: 0; }
  .cover {
    page-break-after: always;
    min-height: 100vh;
    border: none;
    padding: 36px 44px;
  }
  .toc { page-break-after: always; }
  h2, h3, h4 { page-break-after: avoid; }
  figure, table, pre, .table-wrap { page-break-inside: avoid; break-inside: avoid; }
  table { font-size: 0.9rem; }
  th, td { padding: 6px 8px; }
}
'''

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(cover['title'])} - {html.escape(cover['roll'])}</title>
  <style>{css}</style>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js\"></script>
</head>
<body>
  <div class=\"page\">
    <section class=\"cover\">
      <div class=\"inst\">PAF Institute Of Applied Sciences And Technology · Department Of Computer Science & Artificial Intelligence</div>
      <h1>{html.escape(cover['title'])}</h1>
      <h2>{html.escape(cover['subtitle'])}</h2>

      <div class=\"meta-grid\">
        <div class=\"k\">Student Name</div><div class=\"v\">{html.escape(cover['student'])}</div>
        <div class=\"k\">Roll Number</div><div class=\"v\">{html.escape(cover['roll'])}</div>
        <div class=\"k\">Course</div><div class=\"v\">{html.escape(cover['course'])}</div>
        <div class=\"k\">Instructor</div><div class=\"v\">{html.escape(cover['instructor'])}</div>
        <div class=\"k\">Report Type</div><div class=\"v\">Technical Assignment Report</div>
      </div>

      <div class=\"date\">Prepared on {today}</div>
    </section>

    <section class=\"section\">
      {toc_html}
      {body_html}
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    md_path = Path("Assignment_B23F0063AI106_report_proper.md")
    html_path = Path("Assignment_B23F0063AI106_report_proper.html")

    raw = md_path.read_text(encoding="utf-8")

    parts = raw.split("\n---\n", 1)
    preface = parts[0]
    content = parts[1] if len(parts) > 1 else raw

    cover = extract_cover_meta(preface)
    body_html, toc = parse_markdown_to_html(content)
    toc_html = build_toc(toc)

    final_html = build_html_document(cover, toc_html, body_html)
    html_path.write_text(final_html, encoding="utf-8")

    print(f"Generated: {html_path}")
    print(f"Sections in TOC: {len([x for x in toc if 2 <= x[0] <= 4])}")


if __name__ == "__main__":
    main()
