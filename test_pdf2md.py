#!/usr/bin/env python3
"""
Tests for pdf2md.py

Run with:
    python -m pytest test_pdf2md.py -v
    # or directly:
    python test_pdf2md.py
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))
import pdf2md


# ── helpers ───────────────────────────────────────────────────────────────────

def make_minimal_pdf(path: Path) -> None:
    """Write a minimal valid PDF with one text page (no external library needed)."""
    content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f\r
0000000009 00000 n\r
0000000058 00000 n\r
0000000115 00000 n\r
0000000266 00000 n\r
0000000360 00000 n\r
trailer<</Size 6/Root 1 0 R>>
startxref
441
%%EOF"""
    path.write_bytes(content)


# ── text_to_markdown ──────────────────────────────────────────────────────────

class TestTextToMarkdown(unittest.TestCase):

    def test_title_as_h1(self):
        md = pdf2md.text_to_markdown("", title="My Doc")
        self.assertIn("# My Doc", md)

    def test_allcaps_line_becomes_h2(self):
        md = pdf2md.text_to_markdown("INTRODUCTION", title="doc")
        self.assertIn("## Introduction", md)

    def test_long_allcaps_line_not_heading(self):
        long_line = "A" * 85
        md = pdf2md.text_to_markdown(long_line, title="doc")
        self.assertNotIn("##", md)

    def test_page_marker_preserved(self):
        md = pdf2md.text_to_markdown("<!-- page 1 -->\nsome text", title="doc")
        self.assertIn("<!-- page 1 -->", md)

    def test_blank_lines_collapsed(self):
        text = "para one\n\n\n\npara two"
        md = pdf2md.text_to_markdown(text, title="doc")
        # should not have three consecutive blank lines
        self.assertNotIn("\n\n\n", md)

    def test_dash_prefixed_allcaps_not_heading(self):
        md = pdf2md.text_to_markdown("- NOTE", title="doc")
        self.assertNotIn("##", md)

    def test_normal_text_preserved(self):
        md = pdf2md.text_to_markdown("This is a sentence.", title="doc")
        self.assertIn("This is a sentence.", md)

    def test_empty_input(self):
        md = pdf2md.text_to_markdown("", title="Empty")
        self.assertIn("# Empty", md)


# ── collect_pdfs ──────────────────────────────────────────────────────────────

class TestCollectPdfs(unittest.TestCase):

    def test_single_pdf_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sample.pdf"
            p.touch()
            result = pdf2md.collect_pdfs([str(p)])
            self.assertEqual(result, [p])

    def test_directory_finds_pdfs(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "a.pdf").touch()
            (d / "b.pdf").touch()
            (d / "ignore.txt").touch()
            result = pdf2md.collect_pdfs([td])
            names = {r.name for r in result}
            self.assertEqual(names, {"a.pdf", "b.pdf"})

    def test_directory_recurses(self):
        with tempfile.TemporaryDirectory() as td:
            sub = Path(td) / "sub"
            sub.mkdir()
            (sub / "deep.pdf").touch()
            result = pdf2md.collect_pdfs([td])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "deep.pdf")

    def test_non_pdf_file_skipped(self, ):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "notes.txt"
            p.touch()
            result = pdf2md.collect_pdfs([str(p)])
            self.assertEqual(result, [])

    def test_missing_path_skipped(self):
        result = pdf2md.collect_pdfs(["/nonexistent/path/file.pdf"])
        self.assertEqual(result, [])


# ── pdf_to_md (mocked extraction) ────────────────────────────────────────────

class TestPdfToMd(unittest.TestCase):

    def test_output_file_created(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "report.pdf"
            pdf.touch()
            out_dir = Path(td) / "out"

            with patch("pdf2md.extract_text", return_value="SUMMARY\nsome body text"):
                result = pdf2md.pdf_to_md(pdf, out_dir)

            self.assertTrue(result.exists())
            self.assertEqual(result.name, "report.md")

    def test_output_contains_title(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "my_report.pdf"
            pdf.touch()
            out_dir = Path(td) / "out"

            with patch("pdf2md.extract_text", return_value="body"):
                pdf2md.pdf_to_md(pdf, out_dir)

            md_text = (out_dir / "my_report.md").read_text()
            self.assertIn("# My Report", md_text)

    def test_output_dir_created_if_missing(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "x.pdf"
            pdf.touch()
            out_dir = Path(td) / "nested" / "output"

            with patch("pdf2md.extract_text", return_value="text"):
                pdf2md.pdf_to_md(pdf, out_dir)

            self.assertTrue(out_dir.is_dir())

    def test_stem_underscores_become_spaces_in_title(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "annual_review_2024.pdf"
            pdf.touch()
            out_dir = Path(td) / "out"

            with patch("pdf2md.extract_text", return_value="content"):
                pdf2md.pdf_to_md(pdf, out_dir)

            md_text = (out_dir / "annual_review_2024.md").read_text()
            self.assertIn("# Annual Review 2024", md_text)


# ── extract_text fallback logic ───────────────────────────────────────────────

class TestExtractTextFallback(unittest.TestCase):

    def test_uses_pymupdf_when_available(self):
        with patch("pdf2md.extract_text_pymupdf", return_value="via pymupdf") as mock:
            result = pdf2md.extract_text(Path("any.pdf"))
        mock.assert_called_once()
        self.assertEqual(result, "via pymupdf")

    def test_falls_back_to_pdfminer_on_import_error(self):
        with patch("pdf2md.extract_text_pymupdf", side_effect=ImportError):
            with patch("pdf2md.extract_text_pdfminer", return_value="via pdfminer") as mock:
                result = pdf2md.extract_text(Path("any.pdf"))
        mock.assert_called_once()
        self.assertEqual(result, "via pdfminer")

    def test_raises_runtime_error_when_no_library(self):
        with patch("pdf2md.extract_text_pymupdf", side_effect=ImportError):
            with patch("pdf2md.extract_text_pdfminer", side_effect=ImportError):
                with self.assertRaises(RuntimeError):
                    pdf2md.extract_text(Path("any.pdf"))


# ── integration: real PDF (requires pymupdf or pdfminer) ─────────────────────

class TestRealPdf(unittest.TestCase):

    def test_convert_minimal_pdf(self):
        """End-to-end test using a real (minimal) PDF file."""
        try:
            import fitz  # noqa: F401  PyMuPDF
        except ImportError:
            try:
                from pdfminer.high_level import extract_text  # noqa: F401
            except ImportError:
                self.skipTest("No PDF library installed (pip install pymupdf)")

        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "hello.pdf"
            make_minimal_pdf(pdf)
            out_dir = Path(td) / "md_out"

            result = pdf2md.pdf_to_md(pdf, out_dir)

            self.assertTrue(result.exists())
            md_text = result.read_text()
            self.assertIn("# Hello", md_text)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
