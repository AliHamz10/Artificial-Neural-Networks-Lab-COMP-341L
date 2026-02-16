"""
Convert Assignment_1_Report.md to PDF with images embedded.
Run from repo root: python "Assignments/Ali Hamza's Assignments/md_to_pdf.py"

Requires: pip install markdown
Optional: pip install weasyprint  (for direct PDF; otherwise open the generated HTML and Print to PDF)
"""
import base64
import os
import re

# Script directory = Ali Hamza's assignment folder; figures only from this folder
BASE = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE, "figures")
MD_PATH = os.path.join(BASE, "Assignment_1_Report.md")
HTML_PATH = os.path.join(BASE, "Assignment_1_Report.html")
PDF_PATH = os.path.join(BASE, "Assignment_1_Report.pdf")

# Title page details for submission
TITLE_PAGE = {
    "name": "Ali Hamza",
    "registration_number": "B23F0063AI106",
    "course": "Artificial Neural Networks (ANN)",
    "professor": "Dr. Abid Ali",
    "assignment_title": "Assignment 1",
    "subtitle": "Perceptron vs Adaline and the XOR Problem",
    "due_date": "21-02-2026",
}

def main():
    try:
        import markdown
    except ImportError:
        print("Install markdown: pip install markdown")
        return

    with open(MD_PATH, encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(md_text, extensions=["extra", "nl2br"])

    # Embed images from THIS assignment's figures/ only (Ali Hamza's figures)
    def embed_images(html):
        def repl(m):
            full = m.group(0)
            src_m = re.search(r'src=["\']([^"\']+)["\']', full)
            alt_m = re.search(r'alt=["\']([^"\']*)["\']', full)
            if not src_m:
                return full
            src = src_m.group(1)
            alt = alt_m.group(1) if alt_m else ""
            if not src.startswith("figures/"):
                return full
            filename = os.path.basename(src)
            path = os.path.join(FIGURES_DIR, filename)
            if not os.path.isfile(path):
                print("Warning: figure not found (Ali's figures/):", path)
                return full
            with open(path, "rb") as img:
                data = base64.b64encode(img.read()).decode("ascii")
            ext = os.path.splitext(filename)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            return f'<img src="data:{mime};base64,{data}" alt="{alt}" class="report-img" />'
        return re.sub(r'<img[^>]+>', repl, html)

    html_body = embed_images(html_body)

    # Professional title page (first page when printing)
    title_page_html = f"""
    <div class="title-page">
      <div class="title-page-inner">
        <p class="title-page-course">{TITLE_PAGE["course"]}</p>
        <h1 class="title-page-main">{TITLE_PAGE["assignment_title"]}</h1>
        <p class="title-page-subtitle">{TITLE_PAGE["subtitle"]}</p>
        <div class="title-page-rule"></div>
        <table class="title-page-meta" role="presentation">
          <tr><td class="label">Name</td><td>{TITLE_PAGE["name"]}</td></tr>
          <tr><td class="label">Registration Number</td><td>{TITLE_PAGE["registration_number"]}</td></tr>
          <tr><td class="label">Course</td><td>{TITLE_PAGE["course"]}</td></tr>
          <tr><td class="label">Professor</td><td>{TITLE_PAGE["professor"]}</td></tr>
          <tr><td class="label">Due Date</td><td>{TITLE_PAGE["due_date"]}</td></tr>
        </table>
      </div>
    </div>
"""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{TITLE_PAGE["assignment_title"]} Report — {TITLE_PAGE["course"]}</title>
  <style>
    body {{
      font-family: "Georgia", "Times New Roman", serif;
      font-size: 11pt;
      line-height: 1.5;
      max-width: 800px;
      margin: 0 auto;
      padding: 0 1.5em;
      color: #1a1a1a;
    }}
    .title-page {{
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      page-break-after: always;
      padding: 2em;
    }}
    .title-page-inner {{
      max-width: 520px;
    }}
    .title-page-course {{
      font-size: 12pt;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: #444;
      margin: 0 0 1.5em 0;
    }}
    .title-page-main {{
      font-size: 22pt;
      font-weight: 700;
      margin: 0 0 0.25em 0;
      color: #111;
    }}
    .title-page-subtitle {{
      font-size: 13pt;
      color: #333;
      margin: 0 0 2em 0;
    }}
    .title-page-rule {{
      width: 120px;
      height: 2px;
      background: #222;
      margin: 0 auto 2em auto;
    }}
    .title-page-meta {{
      margin: 0 auto;
      text-align: left;
      font-size: 11pt;
    }}
    .title-page-meta td {{
      padding: 0.35em 0.5em 0.35em 0;
      vertical-align: top;
    }}
    .title-page-meta .label {{
      color: #555;
      font-weight: 600;
      white-space: nowrap;
      width: 1%;
    }}
    .report-body {{
      padding: 2em 0 3em 0;
    }}
    .report-body h1 {{ font-size: 18pt; margin-top: 0; border-bottom: 1px solid #ccc; padding-bottom: 0.35em; font-family: Arial, sans-serif; }}
    .report-body h2 {{ font-size: 14pt; margin-top: 1.5em; font-family: Arial, sans-serif; }}
    .report-body h3 {{ font-size: 12pt; margin-top: 1.2em; font-family: Arial, sans-serif; }}
    .report-body p {{ margin: 0.6em 0; }}
    .report-body ul {{ margin: 0.4em 0; padding-left: 1.5em; }}
    .report-body img.report-img {{ max-width: 100%; height: auto; display: block; margin: 0.8em 0; }}
    .report-body strong {{ font-weight: bold; }}
    .report-body hr {{ border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }}
    @media print {{
      body {{ max-width: none; }}
      .title-page {{ min-height: 100vh; box-sizing: border-box; }}
      .report-body img.report-img {{ max-width: 95%; }}
    }}
  </style>
</head>
<body>
{title_page_html}
<div class="report-body">
{html_body}
</div>
</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print("Generated:", HTML_PATH)

    # Try WeasyPrint for direct PDF
    try:
        from weasyprint import HTML
        HTML(string=html_doc, base_url=BASE).write_pdf(PDF_PATH)
        print("Generated:", PDF_PATH)
    except ImportError:
        print("For direct PDF install: pip install weasyprint")
        print("Otherwise: open the HTML file in Chrome or Safari, then File → Print → Save as PDF.")
    except Exception as e:
        print("PDF generation failed:", e)
        print("Open the HTML file in a browser and use Print → Save as PDF.")

if __name__ == "__main__":
    main()
