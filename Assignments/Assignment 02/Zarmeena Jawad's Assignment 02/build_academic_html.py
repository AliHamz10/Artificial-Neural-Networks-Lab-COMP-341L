#!/usr/bin/env python3
"""Render Zarmeena Assignment-02 report markdown to a distinct academic HTML layout."""

from __future__ import annotations

import datetime as dt
import html
import re
from pathlib import Path


def slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    text = re.sub(r"\s+", "-", text)
    return text or "section"


def fmt_inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def parse_table(lines: list[str], i: int):
    if i + 1 >= len(lines):
        return None, i

    h = lines[i].rstrip("\n")
    sep = lines[i + 1].rstrip("\n")

    if "|" not in h or "|" not in sep:
        return None, i
    if not re.match(r"^\s*\|?\s*[:\- ]+\|", sep):
        return None, i

    def split_row(row: str):
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [x.strip() for x in row.split("|")]

    heads = split_row(h)
    rows = []
    j = i + 2
    while j < len(lines):
        row = lines[j].rstrip("\n")
        if not row.strip() or "|" not in row:
            break
        rows.append(split_row(row))
        j += 1

    out = ["<div class=\"tbl-wrap\">\n<table>\n<thead><tr>"]
    for c in heads:
        out.append(f"<th>{fmt_inline(c)}</th>")
    out.append("</tr></thead>\n<tbody>\n")

    for r in rows:
        out.append("<tr>")
        for c in r:
            out.append(f"<td>{fmt_inline(c)}</td>")
        out.append("</tr>\n")

    out.append("</tbody></table>\n</div>\n")
    return "".join(out), j


def md_to_html(md: str):
    lines = md.splitlines(keepends=True)
    out = []
    toc = []

    i = 0
    in_code = False
    code_lang = ""
    code_buf = []
    in_list = False
    para_buf = []

    def flush_para():
        nonlocal para_buf
        if para_buf:
            text = " ".join(x.strip() for x in para_buf if x.strip())
            out.append(f"<p>{fmt_inline(text)}</p>\n")
            para_buf = []

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>\n")
            in_list = False

    while i < len(lines):
        line = lines[i].rstrip("\n")
        s = line.strip()

        if s.startswith("```"):
            flush_para()
            close_list()
            if not in_code:
                in_code = True
                code_lang = s[3:].strip()
                code_buf = []
            else:
                cls = f' class="language-{html.escape(code_lang)}"' if code_lang else ""
                out.append(f"<pre><code{cls}>{html.escape(chr(10).join(code_buf))}</code></pre>\n")
                in_code = False
                code_lang = ""
                code_buf = []
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if not s:
            flush_para()
            close_list()
            i += 1
            continue

        if s == "---":
            flush_para()
            close_list()
            out.append("<hr/>\n")
            i += 1
            continue

        if s == r"\[":
            flush_para()
            close_list()
            math_lines = [line]
            j = i + 1
            while j < len(lines):
                mline = lines[j].rstrip("\n")
                math_lines.append(mline)
                if mline.strip() == r"\]":
                    break
                j += 1
            out.append("<div class=\"display-math\">\n" + "\n".join(math_lines) + "\n</div>\n")
            i = j + 1
            continue

        hm = re.match(r"^(#{1,6})\s+(.*)$", line)
        if hm:
            flush_para()
            close_list()
            lvl = len(hm.group(1))
            title = hm.group(2).strip()
            hid = slug(title)
            toc.append((lvl, title, hid))
            out.append(f"<h{lvl} id=\"{hid}\">{fmt_inline(title)}</h{lvl}>\n")
            i += 1
            continue

        im = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
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

        table_html, ni = parse_table(lines, i)
        if table_html:
            flush_para()
            close_list()
            out.append(table_html)
            i = ni
            continue

        lm = re.match(r"^[-*]\s+(.*)$", s)
        if lm:
            flush_para()
            if not in_list:
                out.append("<ul>\n")
                in_list = True
            out.append(f"<li>{fmt_inline(lm.group(1).strip())}</li>\n")
            i += 1
            continue

        para_buf.append(line)
        i += 1

    flush_para()
    close_list()
    return "".join(out), toc


def build_toc(toc):
    # Keep major levels only for a cleaner visual.
    use = [(l, t, h) for l, t, h in toc if 2 <= l <= 3]
    if not use:
        return ""

    items = ["<section class=\"toc-card\">\n<h2>Contents</h2>\n<ul>\n"]
    for lvl, title, hid in use:
        cls = "minor" if lvl == 3 else "major"
        items.append(f"<li class=\"{cls}\"><a href=\"#{hid}\">{html.escape(title)}</a></li>\n")
    items.append("</ul>\n</section>\n")
    return "".join(items)


def cover_meta(preface: str):
    lines = [x.strip() for x in preface.splitlines() if x.strip()]
    title = "Assignment Report"
    subtitle = ""
    meta = {}

    for line in lines:
        if line.startswith("# ") and title == "Assignment Report":
            title = line[2:].strip()
            continue
        if line.startswith("## ") and not subtitle:
            subtitle = line[3:].strip()
            continue
        m = re.match(r"^\*\*(.+?):\*\*\s*(.+)$", line)
        if m:
            meta[m.group(1).strip()] = m.group(2).strip()

    return {
        'title': title,
        'subtitle': subtitle,
        'author': meta.get('Author', 'Zarmeena Jawad'),
        'roll': meta.get('Registration ID', 'B23F0115AI125'),
        'course': meta.get('Course', 'Artificial Neural Network (COMP-341)'),
        'supervisor': meta.get('Supervisor', 'Dr. Abid Ali'),
    }


def render_html(meta, toc_html, body_html):
    today = dt.date.today().strftime("%d %B %Y")

    css = r'''
:root {
  --bg: #f2f5f8;
  --panel: #ffffff;
  --ink: #1b2430;
  --muted: #556272;
  --accent: #0f766e;
  --accent-soft: #d8f0ed;
  --line: #d4dde6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: radial-gradient(circle at 20% 10%, #eef7f6 0%, var(--bg) 45%, #ecf1f6 100%);
  color: var(--ink);
  font-family: "Cambria", "Times New Roman", serif;
  line-height: 1.58;
}
.shell {
  width: min(1100px, 94vw);
  margin: 26px auto 52px;
}
.hero {
  background: linear-gradient(135deg, #0f766e 0%, #155e75 65%, #1d4e89 100%);
  color: #fff;
  border-radius: 16px;
  padding: 38px 44px;
  box-shadow: 0 10px 28px rgba(12, 36, 62, 0.18);
}
.hero .dept {
  text-transform: uppercase;
  letter-spacing: 0.07em;
  font-size: 0.78rem;
  opacity: 0.92;
  margin-bottom: 14px;
}
.hero h1 {
  margin: 0 0 6px;
  font-size: 2rem;
  line-height: 1.2;
}
.hero h2 {
  margin: 0 0 24px;
  font-size: 1.08rem;
  font-weight: 500;
  opacity: 0.95;
}
.meta {
  display: grid;
  grid-template-columns: repeat(2, minmax(240px, 1fr));
  gap: 10px 22px;
  font-size: 0.97rem;
}
.meta .k { opacity: 0.86; display: inline-block; min-width: 132px; }
.meta .v { font-weight: 600; }
.meta .date {
  grid-column: 1 / -1;
  margin-top: 4px;
  opacity: 0.9;
}

.report {
  margin-top: 20px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 24px 30px 30px;
}

.toc-card {
  background: #f9fffe;
  border: 1px solid #cceae6;
  border-left: 6px solid var(--accent);
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 20px;
}
.toc-card h2 {
  margin: 0 0 8px;
  font-family: "Segoe UI", Tahoma, sans-serif;
  font-size: 1.02rem;
  color: #115e59;
  border: none;
  padding: 0;
}
.toc-card ul {
  list-style: none;
  margin: 0;
  padding: 0;
  column-count: 2;
  column-gap: 28px;
}
.toc-card li { margin: 0.25em 0; break-inside: avoid; }
.toc-card li.minor { margin-left: 10px; }
.toc-card a {
  color: #0b4f4a;
  text-decoration: none;
}
.toc-card a:hover { text-decoration: underline; }

h1, h2, h3, h4, h5, h6 {
  color: #153b5c;
  line-height: 1.3;
  margin-top: 1.28em;
  margin-bottom: 0.52em;
  font-family: "Segoe UI", Tahoma, sans-serif;
}
h1 { font-size: 1.82rem; }
h2 {
  font-size: 1.34rem;
  border-bottom: 2px solid #e5eef7;
  padding-bottom: 6px;
}
h3 { font-size: 1.08rem; }
h4 { font-size: 1.0rem; }

p { margin: 0.52em 0 0.9em; }
ul { margin: 0.45em 0 1em 1.2em; }
li { margin: 0.22em 0; }

a { color: #0f766e; text-decoration: none; }
a:hover { text-decoration: underline; }

code {
  font-family: "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: 0.9em;
  background: #eef6ff;
  border: 1px solid #d7e7fb;
  border-radius: 4px;
  padding: 1px 4px;
}
pre {
  background: #111827;
  color: #f3f4f6;
  border: 1px solid #2f3d53;
  border-radius: 9px;
  padding: 13px;
  overflow-x: auto;
}
pre code {
  background: transparent;
  border: none;
  color: inherit;
  padding: 0;
}

figure {
  margin: 18px 0 22px;
  background: #fff;
  border: 1px solid #d7e0eb;
  border-radius: 10px;
  padding: 10px;
}
figure img {
  width: 100%;
  height: auto;
  border-radius: 6px;
  display: block;
}
figcaption {
  margin-top: 7px;
  color: var(--muted);
  text-align: center;
  font-size: 0.92rem;
}

.tbl-wrap {
  overflow-x: auto;
  margin: 14px 0 20px;
  break-inside: avoid;
  page-break-inside: avoid;
}
table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 0.94rem;
}
th, td {
  border: 1px solid #d5dee8;
  padding: 8px 9px;
  text-align: left;
  vertical-align: top;
  overflow-wrap: anywhere;
}
th {
  background: #edf6ff;
  color: #12406b;
}

.display-math { margin: 11px 0; }

@media print {
  @page { size: A4; margin: 10mm 11mm; }
  body { background: #fff; }
  .shell { width: 100%; margin: 0; }
  .hero {
    border-radius: 0;
    box-shadow: none;
    page-break-after: always;
    min-height: 98vh;
  }
  .report {
    border: none;
    border-radius: 0;
    padding: 0;
    margin-top: 0;
  }
  .toc-card { page-break-after: always; }
  .toc-card ul { column-count: 1; }
  h2, h3, h4 { page-break-after: avoid; }
  figure, pre, table, .tbl-wrap { break-inside: avoid; page-break-inside: avoid; }
  th, td { padding: 6px 8px; }
}
'''

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(meta['title'])} — {html.escape(meta['roll'])}</title>
  <style>{css}</style>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="dept">PAF-IAST · Department of Computer Science & Artificial Intelligence</div>
      <h1>{html.escape(meta['title'])}</h1>
      <h2>{html.escape(meta['subtitle'])}</h2>
      <div class="meta">
        <div><span class="k">Author:</span> <span class="v">{html.escape(meta['author'])}</span></div>
        <div><span class="k">Registration ID:</span> <span class="v">{html.escape(meta['roll'])}</span></div>
        <div><span class="k">Course:</span> <span class="v">{html.escape(meta['course'])}</span></div>
        <div><span class="k">Supervisor:</span> <span class="v">{html.escape(meta['supervisor'])}</span></div>
        <div class="date"><span class="k">Generated:</span> <span class="v">{dt.date.today().strftime('%d %B %Y')}</span></div>
      </div>
    </section>

    <main class="report">
      {toc_html}
      {body_html}
    </main>
  </div>
</body>
</html>
'''


def main():
    md_path = Path('Assignment_B23F0115AI125_report_proper.md')
    html_path = Path('Assignment_B23F0115AI125_report_proper.html')

    if not md_path.exists():
        raise FileNotFoundError(f'Markdown file missing: {md_path}')

    text = md_path.read_text(encoding='utf-8')
    parts = text.split('\n---\n', 1)
    preface = parts[0]
    body = parts[1] if len(parts) > 1 else text

    meta = cover_meta(preface)
    body_html, toc = md_to_html(body)
    toc_html = build_toc(toc)

    html_doc = render_html(meta, toc_html, body_html)
    html_path.write_text(html_doc, encoding='utf-8')

    print('Generated:', html_path)


if __name__ == '__main__':
    main()
