#!/usr/bin/env python3
"""
pic2txt.py — OCR images (PNG, GIF, TIFF) and PDFs into Markdown files.

For each input file a <stem>.md is written to OUTPUT_DIR.

Supported formats:
    Images  — PNG, GIF, TIFF  (any Pillow-readable file)
    PDFs    — text layer extracted directly; image-only pages are OCR'd

OCR backends (tried in order):
    1. pytesseract  — pip install pytesseract  +  apt install tesseract-ocr
    2. easyocr      — pip install easyocr  (downloads model on first run ~1 GB)

Usage:
    python pic2txt.py <file_or_dir> [<file_or_dir> ...]
    python pic2txt.py          # scans INPUT_DIR
"""

import io
import re
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_DIR  = Path("__docs_raw")        # scanned when no CLI args are given
OUTPUT_DIR = Path("__knowledge_raw")   # where .md files are written
OCR_LANG   = "eng+chi_sim+chi_tra"  # tesseract language codes; add/remove as needed
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_SUFFIXES = {".png", ".gif", ".tiff", ".tif"}
PDF_SUFFIX     = ".pdf"
ALL_SUFFIXES   = IMAGE_SUFFIXES | {PDF_SUFFIX}

# ── OCR backends ──────────────────────────────────────────────────────────────

def _ocr_pytesseract(pil_image) -> str:
    import pytesseract
    return pytesseract.image_to_string(pil_image, lang=OCR_LANG)


_easyocr_reader = None  # cached reader (model loading is expensive)

def _ocr_easyocr(pil_image) -> str:
    global _easyocr_reader
    import easyocr
    import numpy as np
    if _easyocr_reader is None:
        langs = [l.strip() for l in OCR_LANG.replace("+", ",").split(",")]
        # map tesseract codes → easyocr codes where they differ
        _LANG_MAP = {"eng": "en", "chi_sim": "ch_sim", "chi_tra": "ch_tra"}
        langs = [_LANG_MAP.get(l, l) for l in langs]
        _easyocr_reader = easyocr.Reader(langs, gpu=False)
    arr = np.array(pil_image.convert("RGB"))
    results = _easyocr_reader.readtext(arr, detail=0, paragraph=True)
    return "\n".join(results)


def ocr_image(pil_image) -> str:
    """Run OCR on a PIL Image, trying backends in order."""
    try:
        return _ocr_pytesseract(pil_image)
    except ImportError:
        pass
    try:
        return _ocr_easyocr(pil_image)
    except ImportError:
        pass
    raise RuntimeError(
        "No OCR backend found. Install one:\n"
        "  pytesseract:  pip install pytesseract  &&  sudo apt install tesseract-ocr\n"
        "  easyocr:      pip install easyocr"
    )


# ── Extractors ────────────────────────────────────────────────────────────────

def extract_image_file(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for an image file (single page)."""
    from PIL import Image
    with Image.open(path) as img:
        frames = []
        try:
            while True:
                frames.append(img.copy().convert("RGB"))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        if not frames:
            frames = [img.convert("RGB")]

    pages = []
    for i, frame in enumerate(frames, start=1):
        text = ocr_image(frame).strip()
        pages.append((i, text))
    return pages


def extract_pdf(path: Path) -> list[tuple[int, str]]:
    """
    Return [(page_number, text), ...] for a PDF.
    Uses PyMuPDF text layer when available; falls back to OCR for image-only pages.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    pages = []

    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        if len(text) >= 8:            # text layer is usable
            pages.append((i, text))
            continue

        # Render page to image and OCR
        mat  = fitz.Matrix(2.0, 2.0)  # 2× zoom → ~144 dpi
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")

        from PIL import Image
        pil_img = Image.open(io.BytesIO(img_bytes))
        ocr_text = ocr_image(pil_img).strip()
        pages.append((i, ocr_text))

    doc.close()
    return pages


def extract(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == PDF_SUFFIX:
        return extract_pdf(path)
    if suffix in IMAGE_SUFFIXES:
        return extract_image_file(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


# ── Markdown builder ──────────────────────────────────────────────────────────

def build_markdown(title: str, pages: list[tuple[int, str]]) -> str:
    def clean(text: str) -> str:
        # collapse excess blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    parts = [f"# {title}\n"]
    for page_num, text in pages:
        if len(pages) > 1:
            parts.append(f"\n<!-- page {page_num} -->\n")
        cleaned = clean(text)
        if cleaned:
            parts.append(cleaned)
    return "\n".join(parts) + "\n"


# ── File I/O ──────────────────────────────────────────────────────────────────

def convert(src: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / (src.stem + ".md")
    title = src.stem.replace("_", " ").replace("-", " ").title()

    print(f"  converting : {src}")
    pages = extract(src)
    md    = build_markdown(title, pages)

    out.write_text(md, encoding="utf-8")
    total_chars = sum(len(t) for _, t in pages)
    print(f"  written    : {out}  ({total_chars:,} chars across {len(pages)} page(s))")
    return out


def collect(targets: list[str]) -> list[Path]:
    files = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            for suffix in ALL_SUFFIXES:
                files.extend(sorted(p.rglob(f"*{suffix}")))
        elif p.suffix.lower() in ALL_SUFFIXES and p.is_file():
            files.append(p)
        else:
            print(f"[warn] skipping {t!r} — not a supported file or directory")
    # stable order, no duplicates
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def main():
    targets = sys.argv[1:]
    if not targets:
        if INPUT_DIR.is_dir():
            targets = [str(INPUT_DIR)]
        else:
            print(
                f"Usage: python pic2txt.py <file_or_dir> ...\n"
                f"Supported: {', '.join(sorted(ALL_SUFFIXES))}\n"
                f"(or set INPUT_DIR; currently {INPUT_DIR!r} not found)"
            )
            sys.exit(1)

    files = collect(targets)
    if not files:
        print("No supported files found.")
        sys.exit(0)

    print(f"Converting {len(files)} file(s) → {OUTPUT_DIR}/\n")
    errors = []
    for f in files:
        try:
            convert(f, OUTPUT_DIR)
        except Exception as exc:
            print(f"  [error] {f}: {exc}")
            errors.append(f)

    ok = len(files) - len(errors)
    print(f"\nDone. {ok}/{len(files)} file(s) written to {OUTPUT_DIR}/")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
