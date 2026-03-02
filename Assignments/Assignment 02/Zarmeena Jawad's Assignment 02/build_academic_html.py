#!/usr/bin/env python3
"""Build APA-style HTML from report markdown (no table of contents)."""

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
        return [x.strip() for x in row.split("|")]

    headers = split_row(head)
    rows = []

    j = i + 2
    while j < len(lines):
        r = lines[j].rstrip("\n")
        if not r.strip() or "|" not in r:
            break
        rows.append(split_row(r))
        j += 1

    out = ["<div class=\"tbl-wrap\">\n<table>\n<thead><tr>"]
    for h in headers:
        out.append(f"<th>{fmt_inline(h)}</th>")
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

        # Display math block
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
    return "".join(out)


def extract_meta(preface: str):
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


def render(meta: dict[str, str], body_html: str) -> str:
    due_or_date = dt.date.today().strftime("%d %B %Y")

    css = r'''
:root {
  --ink: #111;
  --muted: #333;
  --line: #999;
}
* { box-sizing: border-box; }

body {
  margin: 0;
  background: #fff;
  color: var(--ink);
  font-family: "Times New Roman", Times, serif;
  font-size: 12pt;
  line-height: 2;
}

.container {
  width: min(900px, 95vw);
  margin: 0 auto;
  padding: 0;
}

.title-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  page-break-after: always;
}

.title-inner {
  width: 100%;
  max-width: 720px;
}

.title-main {
  font-size: 18pt;
  font-weight: 700;
  margin: 0 0 8px;
}

.title-sub {
  font-size: 14pt;
  margin: 0 0 30px;
}

.meta-block {
  margin-top: 28px;
}

.meta-line {
  margin: 4px 0;
}

.report {
  padding: 0;
}

h1, h2, h3, h4, h5, h6 {
  color: #000;
  line-height: 2;
  margin-top: 1.25em;
  margin-bottom: 0.35em;
  font-weight: 700;
}

h1 { font-size: 14pt; text-align: center; }
h2 { font-size: 13pt; }
h3, h4, h5, h6 { font-size: 12pt; }

p {
  margin: 0 0 0.85em;
  text-indent: 0.5in;
}

ul {
  margin: 0.4em 0 1em 1.5em;
}

li {
  margin: 0.2em 0;
}

a {
  color: #000;
  text-decoration: underline;
}

code {
  font-family: "Courier New", Courier, monospace;
  font-size: 10.5pt;
  background: #f5f5f5;
  border: 1px solid #ddd;
  padding: 1px 4px;
}

pre {
  font-family: "Courier New", Courier, monospace;
  font-size: 10.5pt;
  line-height: 1.45;
  background: #fafafa;
  border: 1px solid #ccc;
  padding: 10px;
  overflow-x: auto;
}

pre code {
  background: transparent;
  border: none;
  padding: 0;
}

hr {
  border: none;
  border-top: 1px solid var(--line);
  margin: 16px 0;
}

figure {
  margin: 12px 0 16px;
  border: 1px solid #ccc;
  padding: 8px;
  page-break-inside: avoid;
}

figure img {
  width: 100%;
  height: auto;
  display: block;
}

figcaption {
  text-align: center;
  font-size: 11pt;
  color: var(--muted);
  margin-top: 6px;
}

.tbl-wrap {
  overflow-x: auto;
  margin: 10px 0 14px;
  page-break-inside: avoid;
}

table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 11pt;
}

th, td {
  border: 1px solid #999;
  padding: 6px 8px;
  text-align: left;
  vertical-align: top;
  overflow-wrap: anywhere;
}

th {
  background: #f3f3f3;
}

.display-math {
  margin: 10px 0;
}

@media print {
  @page {
    size: A4;
    margin: 1in;
  }

  body {
    font-size: 12pt;
    line-height: 2;
  }

  .container {
    width: 100%;
    margin: 0;
  }

  .title-page {
    min-height: 100vh;
    page-break-after: always;
  }

  h2, h3, h4 {
    page-break-after: avoid;
  }

  figure, pre, table, .tbl-wrap {
    page-break-inside: avoid;
    break-inside: avoid;
  }
}
'''

    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(meta['title'])} - {html.escape(meta['roll'])}</title>
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
  <div class="container">
    <section class="title-page">
      <div class="title-inner">
        <div class="title-main">{html.escape(meta['title'])}</div>
        <div class="title-sub">{html.escape(meta['subtitle'])}</div>

        <div class="meta-block">
          <div class="meta-line">{html.escape(meta['author'])}</div>
          <div class="meta-line">Department of Computer Science & Artificial Intelligence</div>
          <div class="meta-line">PAF Institute of Applied Sciences and Technology</div>
          <div class="meta-line">{html.escape(meta['course'])}</div>
          <div class="meta-line">{html.escape(meta['supervisor'])}</div>
          <div class="meta-line">Registration ID: {html.escape(meta['roll'])}</div>
          <div class="meta-line">Date: {due_or_date}</div>
        </div>
      </div>
    </section>

    <main class="report">
      {body_html}
    </main>
  </div>
</body>
</html>
'''


def main() -> None:
    md_path = Path('Assignment_B23F0115AI125_report_proper.md')
    html_path = Path('Assignment_B23F0115AI125_report_proper.html')

    if not md_path.exists():
        raise FileNotFoundError(f'Markdown file missing: {md_path}')

    raw = md_path.read_text(encoding='utf-8')
    parts = raw.split('\n---\n', 1)
    preface = parts[0]
    body = parts[1] if len(parts) > 1 else raw

    meta = extract_meta(preface)
    body_html = md_to_html(body)

    html_doc = render(meta, body_html)
    html_path.write_text(html_doc, encoding='utf-8')

    print('Generated:', html_path)


if __name__ == '__main__':
    main()
