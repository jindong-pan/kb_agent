#!/usr/bin/env python3
"""
pdf2md.py — Extract PDF files to Markdown in a configurable output path.

Usage:
    python pdf2md.py <pdf_file_or_dir> [<pdf_file_or_dir> ...]

Configuration:
    Set OUTPUT_DIR below to control where Markdown files are written.
    Set INPUT_DIR to scan a directory automatically (used when no args given).
"""

import sys
import re
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_DIR  = Path("docs")          # scanned when no CLI args are given
OUTPUT_DIR = Path("knowledge")     # where .md files are written
# ──────────────────────────────────────────────────────────────────────────────


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract text using PyMuPDF (fitz), preserving page structure."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"<!-- page {i} -->\n{text.rstrip()}")
    doc.close()
    return "\n\n".join(pages)


def extract_text_pdfminer(pdf_path: Path) -> str:
    """Fallback extractor using pdfminer.six."""
    from pdfminer.high_level import extract_text
    return extract_text(str(pdf_path))


def extract_text(pdf_path: Path) -> str:
    try:
        return extract_text_pymupdf(pdf_path)
    except ImportError:
        pass
    try:
        return extract_text_pdfminer(pdf_path)
    except ImportError:
        pass
    raise RuntimeError(
        "No PDF library found. Install one:\n"
        "  pip install pymupdf        # recommended\n"
        "  pip install pdfminer.six   # fallback"
    )


def text_to_markdown(text: str, title: str) -> str:
    """Apply lightweight heuristics to produce readable Markdown."""
    lines = text.splitlines()
    md_lines = [f"# {title}\n"]

    for line in lines:
        stripped = line.strip()

        # preserve page markers as-is
        if stripped.startswith("<!-- page"):
            md_lines.append(f"\n{stripped}\n")
            continue

        if not stripped:
            md_lines.append("")
            continue

        # simple heading heuristic: short ALL-CAPS or Title Case lines
        if (
            len(stripped) < 80
            and stripped == stripped.upper()
            and re.search(r"[A-Z]", stripped)
            and not stripped.startswith("-")
        ):
            md_lines.append(f"\n## {stripped.title()}\n")
            continue

        md_lines.append(stripped)

    # collapse runs of blank lines to at most two
    result, prev_blank = [], False
    for line in md_lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        result.append(line)
        prev_blank = is_blank

    return "\n".join(result)


def pdf_to_md(pdf_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (pdf_path.stem + ".md")

    print(f"  extracting : {pdf_path}")
    raw = extract_text(pdf_path)
    md  = text_to_markdown(raw, title=pdf_path.stem.replace("_", " ").replace("-", " ").title())

    out_path.write_text(md, encoding="utf-8")
    print(f"  written    : {out_path}  ({len(md):,} chars)")
    return out_path


def collect_pdfs(targets: list[str]) -> list[Path]:
    pdfs = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            pdfs.extend(sorted(p.rglob("*.pdf")))
        elif p.suffix.lower() == ".pdf" and p.is_file():
            pdfs.append(p)
        else:
            print(f"[warn] skipping {t!r} (not a PDF or directory)")
    return pdfs


def main():
    targets = sys.argv[1:]
    if not targets:
        if INPUT_DIR.is_dir():
            targets = [str(INPUT_DIR)]
        else:
            print(f"Usage: python pdf2md.py <pdf_or_dir> ...\n"
                  f"       (or set INPUT_DIR; currently {INPUT_DIR!r} not found)")
            sys.exit(1)

    pdfs = collect_pdfs(targets)
    if not pdfs:
        print("No PDF files found.")
        sys.exit(0)

    print(f"Converting {len(pdfs)} PDF(s) → {OUTPUT_DIR}/\n")
    for pdf in pdfs:
        pdf_to_md(pdf, OUTPUT_DIR)

    print(f"\nDone. {len(pdfs)} file(s) written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
