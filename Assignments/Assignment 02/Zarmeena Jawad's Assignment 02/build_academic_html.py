#!/usr/bin/env python3
"""Build polished academic HTML from report markdown (Zarmeena Assignment 02)."""

from __future__ import annotations

import datetime as dt
import html
import re
from pathlib import Path


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text).strip().lower()
    text = re.sub(r"\s+", "-", text)
    return text or "section"


def inline_md(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def parse_table(lines: list[str], i: int):
    if i + 1 >= len(lines):
        return None, i
    head = lines[i].rstrip("\n")
    sep = lines[i + 1].rstrip("\n")
    if "|" not in head or "|" not in sep:
        return None, i
    if not re.match(r"^\s*\|?\s*[:\- ]+\|", sep):
        return None, i

    def split_row(row: str):
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
        r = lines[j].rstrip("\n")
        if "|" not in r or not r.strip():
            break
        rows.append(split_row(r))
        j += 1

    out = ["<div class=\"table-wrap\">\n<table>\n<thead><tr>"]
    for h in headers:
        out.append(f"<th>{inline_md(h)}</th>")
    out.append("</tr></thead>\n<tbody>\n")

    for r in rows:
        out.append("<tr>")
        for c in r:
            out.append(f"<td>{inline_md(c)}</td>")
        out.append("</tr>\n")

    out.append("</tbody></table>\n</div>\n")
    return "".join(out), j


def parse_markdown(md_text: str):
    lines = md_text.splitlines(keepends=True)
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
            p = " ".join(x.strip() for x in para_buf if x.strip())
            out.append(f"<p>{inline_md(p)}</p>\n")
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
            math = [line]
            j = i + 1
            while j < len(lines):
                ml = lines[j].rstrip("\n")
                math.append(ml)
                if ml.strip() == r"\]":
                    break
                j += 1
            out.append("<div class=\"display-math\">\n" + "\n".join(math) + "\n</div>\n")
            i = j + 1
            continue

        hm = re.match(r"^(#{1,6})\s+(.*)$", line)
        if hm:
            flush_para()
            close_list()
            lvl = len(hm.group(1))
            title = hm.group(2).strip()
            hid = slugify(title)
            toc.append((lvl, title, hid))
            out.append(f"<h{lvl} id=\"{hid}\">{inline_md(title)}</h{lvl}>\n")
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

        table_html, new_i = parse_table(lines, i)
        if table_html:
            flush_para()
            close_list()
            out.append(table_html)
            i = new_i
            continue

        lm = re.match(r"^[-*]\s+(.*)$", s)
        if lm:
            flush_para()
            if not in_list:
                out.append("<ul>\n")
                in_list = True
            out.append(f"<li>{inline_md(lm.group(1).strip())}</li>\n")
            i += 1
            continue

        para_buf.append(line)
        i += 1

    flush_para()
    close_list()
    return "".join(out), toc


def make_toc(toc):
    items = [(l, t, h) for l, t, h in toc if 2 <= l <= 4]
    if not items:
        return ""
    out = ["<nav class=\"toc\">\n<h2>Table Of Contents</h2>\n<ol>\n"]
    for l, t, h in items:
        out.append(f"<li class=\"toc-l{l}\"><a href=\"#{h}\">{html.escape(t)}</a></li>\n")
    out.append("</ol>\n</nav>\n")
    return "".join(out)


def extract_cover(preface: str):
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
        'student': meta.get('Student Name', 'Zarmeena Jawad'),
        'roll': meta.get('Roll Number', 'B23F0115AI125'),
        'course': meta.get('Course', 'Artificial Neural Network (COMP-341)'),
        'instructor': meta.get('Instructor', 'Dr. Abid Ali'),
    }


def build_html(cover, toc_html, body_html):
    today = dt.date.today().strftime('%d %B %Y')

    css = r'''
:root {
  --ink: #16243a;
  --muted: #4d5f79;
  --line: #d8dee9;
  --accent: #20457c;
  --bg: #f7f9fc;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0;
  font-family: "Georgia", "Times New Roman", serif;
  color: var(--ink);
  background: #fff;
  line-height: 1.55;
}
.page { width: min(980px, 92vw); margin: 28px auto 56px; }
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
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 18px;
}
.cover h1 {
  margin: 0 0 8px;
  font-size: 2.05rem;
  color: var(--accent);
  line-height: 1.2;
}
.cover h2 {
  margin: 0 0 30px;
  font-size: 1.14rem;
  color: var(--muted);
  font-weight: 500;
}
.meta-grid {
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 6px 12px;
  margin-top: 20px;
}
.meta-grid .k { color: var(--muted); }
.meta-grid .v { font-weight: 600; }
.cover .date { margin-top: 34px; color: var(--muted); font-size: 0.95rem; }

.section { margin-top: 32px; }
hr { border: none; border-top: 1px solid var(--line); margin: 26px 0; }

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

a { color: #214f8f; text-decoration: none; }
a:hover { text-decoration: underline; }

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
pre code { background: transparent; border: none; padding: 0; color: inherit; }

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
.toc ol { margin: 0; padding-left: 1.2em; }
.toc li { margin: 0.3em 0; }
.toc .toc-l3 { margin-left: 1.1em; }
.toc .toc-l4 { margin-left: 2.1em; }

.display-math { margin: 12px 0; padding: 6px 0; }

@media print {
  @page { size: A4; margin: 10mm 11mm; }
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

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(cover['title'])} - {html.escape(cover['roll'])}</title>
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
  <div class="page">
    <section class="cover">
      <div class="inst">PAF Institute Of Applied Sciences And Technology · Department Of Computer Science & Artificial Intelligence</div>
      <h1>{html.escape(cover['title'])}</h1>
      <h2>{html.escape(cover['subtitle'])}</h2>

      <div class="meta-grid">
        <div class="k">Student Name</div><div class="v">{html.escape(cover['student'])}</div>
        <div class="k">Roll Number</div><div class="v">{html.escape(cover['roll'])}</div>
        <div class="k">Course</div><div class="v">{html.escape(cover['course'])}</div>
        <div class="k">Instructor</div><div class="v">{html.escape(cover['instructor'])}</div>
        <div class="k">Report Type</div><div class="v">Technical Assignment Report</div>
      </div>

      <div class="date">Prepared on {today}</div>
    </section>

    <section class="section">
      {toc_html}
      {body_html}
    </section>
  </div>
</body>
</html>
'''


def main():
    md_path = Path('Assignment_B23F0115AI125_report_proper.md')
    html_path = Path('Assignment_B23F0115AI125_report_proper.html')

    if not md_path.exists():
        raise FileNotFoundError(f'Report markdown not found: {md_path}')

    text = md_path.read_text(encoding='utf-8')
    parts = text.split('\n---\n', 1)
    preface = parts[0]
    body = parts[1] if len(parts) > 1 else text

    cover = extract_cover(preface)
    body_html, toc = parse_markdown(body)
    toc_html = make_toc(toc)

    html_doc = build_html(cover, toc_html, body_html)
    html_path.write_text(html_doc, encoding='utf-8')

    print('Generated:', html_path)


if __name__ == '__main__':
    main()
