"""Generate publication-quality PDF report from Markdown documentation.

Converts Markdown reports (e.g. docs/PDE_SUPERVISION_EVALUATION_REPORT.md or
docs/PROB_LATENT_EVALUATION_REPORT.md) into self-contained HTML with embedded assets,
crisp MathJax-rendered formulas, and modern typography, then renders to PDF via headless Google Chrome.
"""

import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def image_to_base64_data_uri(img_path: Path) -> str:
    """Read an image file and convert to base64 data URI."""
    if not img_path.exists():
        print(f"Warning: image {img_path} not found.")
        return ""
    suffix = img_path.suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_styled_html(md_content: str, doc_dir: Path) -> str:
    """Convert Markdown content to styled HTML with embedded assets and MathJax support."""
    # 0. Extract document title if available
    title = "世界模型科研评估与治理报告"
    first_heading = re.search(r"^#\s+(.+)$", md_content, flags=re.MULTILINE)
    if first_heading:
        title = first_heading.group(1).strip()

    # 1. Protect math formulas from markdown parser (avoiding _ converting to <em>)
    math_blocks = []

    def save_block(m):
        idx = len(math_blocks)
        math_blocks.append(m.group(0))
        return f"MATHBLOCKTOKEN{idx}END"

    def save_inline(m):
        idx = len(math_blocks)
        math_blocks.append(m.group(0))
        return f"MATHINLINETOKEN{idx}END"

    processed_md = re.sub(r"\$\$(.+?)\$\$", save_block, md_content, flags=re.DOTALL)
    processed_md = re.sub(
        r"(?<!\$)\$(?!\$)([^\n$]+?)(?<!\$)\$(?!\$)", save_inline, processed_md
    )

    # 2. Resolve image links: ![alt](path) -> embed as data URI
    def replace_image(match):
        alt = match.group(1)
        src = match.group(2).strip()

        candidates = []
        if src.startswith("/"):
            candidates.append(Path(src))
        else:
            candidates.append((doc_dir / src).resolve())
            candidates.append((PROJECT_ROOT / src.lstrip("/")).resolve())
            candidates.append((PROJECT_ROOT / "outputs" / "figures" / "probabilistic" / Path(src).name).resolve())
            candidates.append((PROJECT_ROOT / "outputs" / "figures" / "pde_controlled" / Path(src).name).resolve())
            for p in (PROJECT_ROOT / "outputs" / "figures").glob(f"**/{Path(src).name}"):
                candidates.append(p.resolve())

        found_path = None
        for c in candidates:
            if c.exists() and c.is_file():
                found_path = c
                break

        if found_path:
            data_uri = image_to_base64_data_uri(found_path)
            return (
                f'<div class="figure-container"><img src="{data_uri}" alt="{alt}" '
                f'class="report-figure" /><div class="figure-caption">{alt}</div></div>'
            )
        else:
            print(f"[WARNING] Image not found for src='{src}', searched {len(candidates)} candidates.")
            return f'<p class="missing-image">[Image: {alt} ({src})]</p>'

    processed_md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, processed_md)

    # 3. Convert GitHub-style alerts: > [!WARNING] etc.
    alert_pattern = re.compile(r">\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*\n((?:>.*\n?)+)")

    def replace_alert(m):
        alert_type = m.group(1).lower()
        content = m.group(2)
        cleaned_content = re.sub(r"^>\s?", "", content, flags=re.MULTILINE)
        return (
            f'<div class="alert alert-{alert_type}"><div class="alert-title">{m.group(1)}</div>\n\n'
            f"{cleaned_content}\n</div>"
        )

    processed_md = alert_pattern.sub(replace_alert, processed_md)

    # 4. Render markdown to HTML
    body_html = markdown.markdown(
        processed_md,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
            "toc",
        ],
    )

    # 5. Restore math formulas
    for idx, block in enumerate(math_blocks):
        body_html = body_html.replace(f"MATHBLOCKTOKEN{idx}END", block)
        body_html = body_html.replace(f"MATHINLINETOKEN{idx}END", block)

    # 6. Wrap in professional report CSS layout
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  svg: {{
    fontCache: 'global'
  }}
}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
    @page {{
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
        @bottom-right {{
            content: counter(page);
            font-size: 8.5pt;
            color: #64748b;
        }}
    }}
    * {{
        box-sizing: border-box;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "WenQuanYi Zen Hei", Roboto, sans-serif;
        color: #1e293b;
        line-height: 1.55;
        font-size: 9.8pt;
        background: #ffffff;
        margin: 0;
        padding: 0;
    }}
    h1 {{
        font-size: 18pt;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 2.5px solid #2563eb;
        padding-bottom: 8px;
        margin-top: 0;
        margin-bottom: 12px;
        line-height: 1.3;
    }}
    h2 {{
        font-size: 13pt;
        font-weight: 600;
        color: #0f172a;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 9px;
        page-break-after: avoid;
    }}
    h3 {{
        font-size: 11pt;
        font-weight: 600;
        color: #1e293b;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }}
    h4 {{
        font-size: 10pt;
        font-weight: 600;
        color: #334155;
        margin-top: 12px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 7px;
        text-align: justify;
    }}
    blockquote {{
        margin: 8px 0;
        padding: 8px 12px;
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        color: #334155;
        font-size: 9.2pt;
    }}
    blockquote p:last-child {{
        margin-bottom: 0;
    }}
    .alert {{
        margin: 10px 0;
        padding: 10px 12px;
        border-radius: 6px;
        font-size: 9.2pt;
        page-break-inside: avoid;
    }}
    .alert-warning {{
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        color: #92400e;
    }}
    .alert-note {{
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        color: #1e40af;
    }}
    .alert-important {{
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        color: #991b1b;
    }}
    .alert-title {{
        font-weight: 700;
        margin-bottom: 4px;
        text-transform: uppercase;
        font-size: 8.5pt;
        letter-spacing: 0.5px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 8.8pt;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #cbd5e1;
        padding: 5.5px 7px;
        text-align: left;
    }}
    th {{
        background-color: #f1f5f9;
        font-weight: 600;
        color: #0f172a;
    }}
    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}
    code {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 1px 3.5px;
        border-radius: 3px;
        font-size: 8.5pt;
    }}
    pre {{
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        color: #0f172a;
        padding: 10px 12px;
        border-radius: 6px;
        overflow-x: auto;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        font-size: 7.8pt;
        line-height: 1.25;
        margin: 10px 0;
        page-break-inside: avoid;
        white-space: pre;
    }}
    pre code {{
        background-color: transparent;
        color: inherit;
        padding: 0;
    }}
    .figure-container {{
        text-align: center;
        margin: 14px 0;
        page-break-inside: avoid;
    }}
    .report-figure {{
        max-width: 98%;
        height: auto;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .figure-caption {{
        margin-top: 5px;
        font-size: 8.5pt;
        color: #64748b;
        font-style: italic;
    }}
    hr {{
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 14px 0;
    }}
    ul, ol {{
        margin-top: 0;
        margin-bottom: 7px;
        padding-left: 18px;
    }}
    li {{
        margin-bottom: 3.5px;
    }}
    strong {{
        color: #0f172a;
    }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""
    return html


def convert_markdown_to_pdf(md_path: Path, output_pdf_path: Path):
    """Convert input markdown file to styled PDF via Google Chrome."""
    if not md_path.exists():
        raise FileNotFoundError(f"Input markdown not found at {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    print(f"Generating styled HTML from {md_path}...")
    html_content = build_styled_html(md_text, md_path.parent)

    tmp_html_path = md_path.parent / (md_path.stem + "_tmp_render.html")
    with open(tmp_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Printing PDF to {output_pdf_path} via Headless Google Chrome...")
    cmd = [
        "/usr/bin/google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={output_pdf_path.resolve()}",
        str(tmp_html_path.resolve()),
    ]

    ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45)
    if ret.returncode != 0:
        print(f"Chrome error stderr:\n{ret.stderr}")
        raise RuntimeError(f"Google Chrome PDF conversion failed with exit code {ret.returncode}")

    if tmp_html_path.exists():
        tmp_html_path.unlink()

    if output_pdf_path.exists():
        size_kb = output_pdf_path.stat().st_size / 1024
        print(f"[SUCCESS] PDF successfully created: {output_pdf_path} ({size_kb:.1f} KB)")
    else:
        raise RuntimeError(f"Target PDF {output_pdf_path} was not generated.")


def main():
    parser = argparse.ArgumentParser(description="Export markdown documentation to PDF.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="docs/PDE_SUPERVISION_EVALUATION_REPORT.md",
        help="Path to input markdown file (relative to project root or absolute).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="Path to output PDF file (defaults to replacing .md with .pdf).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all standard evaluation reports in docs/.",
    )
    args = parser.parse_args()

    targets = []
    if args.all:
        targets = [
            (
                PROJECT_ROOT / "docs" / "PDE_SUPERVISION_EVALUATION_REPORT.md",
                PROJECT_ROOT / "docs" / "PDE_SUPERVISION_EVALUATION_REPORT.pdf",
            ),
            (
                PROJECT_ROOT / "docs" / "PROB_LATENT_EVALUATION_REPORT.md",
                PROJECT_ROOT / "docs" / "PROB_LATENT_EVALUATION_REPORT.pdf",
            ),
        ]
    else:
        in_p = Path(args.input)
        if not in_p.is_absolute():
            in_p = (PROJECT_ROOT / in_p).resolve()

        if args.output:
            out_p = Path(args.output)
            if not out_p.is_absolute():
                out_p = (PROJECT_ROOT / out_p).resolve()
        else:
            out_p = in_p.with_suffix(".pdf")
        targets = [(in_p, out_p)]

    # Current conversation artifact directory
    artifact_dirs = [
        Path("/root/.gemini/antigravity-ide/brain/25604b97-21bd-42b9-814b-524c55a6667a"),
        Path("/root/.gemini/antigravity-ide/brain/df6d6fdd-c45c-4927-91ce-a466b0537f1e"),
    ]

    for md_file, pdf_file in targets:
        convert_markdown_to_pdf(md_file, pdf_file)

        for adir in artifact_dirs:
            if adir.exists():
                target_artifact_pdf = adir / pdf_file.name
                shutil.copy2(pdf_file, target_artifact_pdf)
                print(f"[COPIED] Also copied to artifact directory: {target_artifact_pdf}")


if __name__ == "__main__":
    main()
