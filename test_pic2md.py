#!/usr/bin/env python3
"""
Tests for pic2txt.py

Run:
    python test_pic2txt.py
    python -m pytest test_pic2txt.py -v   # if pytest is installed
"""

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent))
import pic2txt


# ── helpers ───────────────────────────────────────────────────────────────────

def make_white_png(path: Path, width: int = 64, height: int = 32) -> None:
    from PIL import Image
    Image.new("RGB", (width, height), color="white").save(path)


def make_pdf(path: Path, pages: list[str]) -> None:
    """Create a multi-page PDF with text layers using PyMuPDF."""
    import fitz
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=612, height=792)
        if text:
            page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()


# ── build_markdown ────────────────────────────────────────────────────────────

class TestBuildMarkdown(unittest.TestCase):

    def test_title_as_h1(self):
        md = pic2txt.build_markdown("My Doc", [(1, "some text")])
        self.assertTrue(md.startswith("# My Doc"))

    def test_single_page_no_page_marker(self):
        md = pic2txt.build_markdown("Doc", [(1, "hello")])
        self.assertNotIn("<!-- page", md)

    def test_multi_page_adds_markers(self):
        md = pic2txt.build_markdown("Doc", [(1, "p1"), (2, "p2")])
        self.assertIn("<!-- page 1 -->", md)
        self.assertIn("<!-- page 2 -->", md)

    def test_empty_page_text_skipped(self):
        md = pic2txt.build_markdown("Doc", [(1, ""), (2, "real text")])
        self.assertIn("real text", md)

    def test_whitespace_only_page_skipped(self):
        md = pic2txt.build_markdown("Doc", [(1, "   \n  "), (2, "content")])
        self.assertIn("content", md)

    def test_excess_blank_lines_collapsed(self):
        md = pic2txt.build_markdown("Doc", [(1, "a\n\n\n\nb")])
        self.assertNotIn("\n\n\n", md)

    def test_ends_with_newline(self):
        md = pic2txt.build_markdown("Doc", [(1, "text")])
        self.assertTrue(md.endswith("\n"))

    def test_all_empty_pages_still_has_title(self):
        md = pic2txt.build_markdown("Empty Doc", [(1, ""), (2, "")])
        self.assertIn("# Empty Doc", md)

    def test_page_text_preserved(self):
        md = pic2txt.build_markdown("Doc", [(1, "发票合计 Invoice Total")])
        self.assertIn("发票合计 Invoice Total", md)

    def test_page_markers_use_correct_numbers(self):
        md = pic2txt.build_markdown("Doc", [(3, "c"), (7, "g")])
        self.assertIn("<!-- page 3 -->", md)
        self.assertIn("<!-- page 7 -->", md)


# ── collect ───────────────────────────────────────────────────────────────────

class TestCollect(unittest.TestCase):

    def _tmp_files(self, names):
        td = tempfile.mkdtemp()
        paths = []
        for n in names:
            p = Path(td) / n
            p.touch()
            paths.append(p)
        return td, paths

    def test_single_png(self):
        td, [p] = self._tmp_files(["img.png"])
        self.assertEqual(pic2txt.collect([str(p)]), [p])

    def test_single_gif(self):
        td, [p] = self._tmp_files(["anim.gif"])
        self.assertEqual(pic2txt.collect([str(p)]), [p])

    def test_single_tiff(self):
        td, [p] = self._tmp_files(["scan.tiff"])
        self.assertEqual(pic2txt.collect([str(p)]), [p])

    def test_single_tif_short_extension(self):
        td, [p] = self._tmp_files(["scan.tif"])
        self.assertEqual(pic2txt.collect([str(p)]), [p])

    def test_single_pdf(self):
        td, [p] = self._tmp_files(["doc.pdf"])
        self.assertEqual(pic2txt.collect([str(p)]), [p])

    def test_unsupported_file_skipped(self):
        td, [p] = self._tmp_files(["notes.docx"])
        self.assertEqual(pic2txt.collect([str(p)]), [])

    def test_directory_finds_all_supported(self):
        td, _ = self._tmp_files(["a.png", "b.gif", "c.tiff", "d.pdf", "e.txt"])
        names = {f.name for f in pic2txt.collect([td])}
        self.assertEqual(names, {"a.png", "b.gif", "c.tiff", "d.pdf"})

    def test_directory_recurses(self):
        td = tempfile.mkdtemp()
        sub = Path(td) / "sub"
        sub.mkdir()
        (sub / "deep.png").touch()
        result = pic2txt.collect([td])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "deep.png")

    def test_no_duplicates_same_path_twice(self):
        td, [p] = self._tmp_files(["x.png"])
        self.assertEqual(len(pic2txt.collect([str(p), str(p)])), 1)

    def test_missing_file_skipped(self):
        self.assertEqual(pic2txt.collect(["/nonexistent/file.png"]), [])

    def test_uppercase_extension_accepted(self):
        td = tempfile.mkdtemp()
        p = Path(td) / "img.PNG"
        p.touch()
        self.assertEqual(pic2txt.collect([str(p)]), [p])

    def test_multiple_dirs_merged(self):
        td1 = tempfile.mkdtemp()
        td2 = tempfile.mkdtemp()
        (Path(td1) / "a.png").touch()
        (Path(td2) / "b.png").touch()
        result = pic2txt.collect([td1, td2])
        names = {f.name for f in result}
        self.assertEqual(names, {"a.png", "b.png"})


# ── ocr_image: backend selection & fallback ───────────────────────────────────

class TestOcrImageFallback(unittest.TestCase):

    def setUp(self):
        pic2txt._easyocr_reader = None

    def test_uses_pytesseract_first(self):
        fake_img = MagicMock()
        with patch("pic2txt._ocr_pytesseract", return_value="tess") as mock:
            result = pic2txt.ocr_image(fake_img)
        mock.assert_called_once_with(fake_img)
        self.assertEqual(result, "tess")

    def test_falls_back_to_easyocr_on_import_error(self):
        fake_img = MagicMock()
        with patch("pic2txt._ocr_pytesseract", side_effect=ImportError):
            with patch("pic2txt._ocr_easyocr", return_value="easy") as mock:
                result = pic2txt.ocr_image(fake_img)
        mock.assert_called_once_with(fake_img)
        self.assertEqual(result, "easy")

    def test_raises_runtime_error_when_no_backend(self):
        fake_img = MagicMock()
        with patch("pic2txt._ocr_pytesseract", side_effect=ImportError):
            with patch("pic2txt._ocr_easyocr", side_effect=ImportError):
                with self.assertRaises(RuntimeError) as ctx:
                    pic2txt.ocr_image(fake_img)
        self.assertIn("pip install", str(ctx.exception))

    def test_error_message_mentions_both_backends(self):
        fake_img = MagicMock()
        with patch("pic2txt._ocr_pytesseract", side_effect=ImportError):
            with patch("pic2txt._ocr_easyocr", side_effect=ImportError):
                with self.assertRaises(RuntimeError) as ctx:
                    pic2txt.ocr_image(fake_img)
        msg = str(ctx.exception)
        self.assertIn("pytesseract", msg)
        self.assertIn("easyocr", msg)


# ── _ocr_pytesseract: OCR_LANG forwarded correctly ────────────────────────────

class TestPytesseractLangArg(unittest.TestCase):

    def _call_with_lang(self, ocr_lang: str) -> dict:
        """Return the kwargs pytesseract.image_to_string was called with."""
        captured = {}
        fake_img = MagicMock()
        original = pic2txt.OCR_LANG
        pic2txt.OCR_LANG = ocr_lang
        try:
            mock_tess = MagicMock()
            mock_tess.image_to_string.side_effect = (
                lambda img, **kw: captured.update(kw) or ""
            )
            with patch.dict("sys.modules", {"pytesseract": mock_tess}):
                pic2txt._ocr_pytesseract(fake_img)
        finally:
            pic2txt.OCR_LANG = original
        return captured

    def test_passes_eng_lang(self):
        kw = self._call_with_lang("eng")
        self.assertEqual(kw.get("lang"), "eng")

    def test_passes_trilingual_lang_string(self):
        kw = self._call_with_lang("eng+chi_sim+chi_tra")
        self.assertEqual(kw.get("lang"), "eng+chi_sim+chi_tra")

    def test_default_ocr_lang_contains_all_three(self):
        codes = set(pic2txt.OCR_LANG.split("+"))
        self.assertIn("eng",     codes)
        self.assertIn("chi_sim", codes)
        self.assertIn("chi_tra", codes)


# ── easyocr: language code mapping ───────────────────────────────────────────

class TestEasyocrLangMapping(unittest.TestCase):

    def setUp(self):
        pic2txt._easyocr_reader = None

    def _captured_langs(self, ocr_lang: str) -> list:
        captured = {}

        class FakeReader:
            def __init__(self, langs, **kw):
                captured["langs"] = langs
            def readtext(self, arr, **kw):
                return []

        original = pic2txt.OCR_LANG
        pic2txt.OCR_LANG = ocr_lang
        try:
            import numpy as np          # noqa: F401
            from PIL import Image
            fake_img = Image.new("RGB", (8, 8))
            with patch("easyocr.Reader", FakeReader):
                pic2txt._ocr_easyocr(fake_img)
        except ImportError:
            self.skipTest("easyocr not installed")
        finally:
            pic2txt.OCR_LANG = original
            pic2txt._easyocr_reader = None
        return captured.get("langs", [])

    def test_eng_maps_to_en(self):
        langs = self._captured_langs("eng")
        self.assertIn("en", langs)
        self.assertNotIn("eng", langs)

    def test_chi_sim_maps_to_ch_sim(self):
        self.assertIn("ch_sim", self._captured_langs("chi_sim"))

    def test_chi_tra_maps_to_ch_tra(self):
        self.assertIn("ch_tra", self._captured_langs("chi_tra"))

    def test_combined_trilingual_all_mapped(self):
        langs = self._captured_langs("eng+chi_sim+chi_tra")
        self.assertEqual(sorted(langs), ["ch_sim", "ch_tra", "en"])

    def test_no_unmapped_tesseract_codes(self):
        langs = self._captured_langs("eng+chi_sim+chi_tra")
        for raw in ("eng", "chi_sim", "chi_tra"):
            self.assertNotIn(raw, langs)

    def test_reader_cached_after_first_call(self):
        captured_calls = []

        class CountingReader:
            def __init__(self, langs, **kw):
                captured_calls.append(langs)
            def readtext(self, arr, **kw):
                return []

        try:
            import numpy as np   # noqa: F401
            from PIL import Image
            fake_img = Image.new("RGB", (8, 8))
            with patch("easyocr.Reader", CountingReader):
                pic2txt._ocr_easyocr(fake_img)
                pic2txt._ocr_easyocr(fake_img)
        except ImportError:
            self.skipTest("easyocr not installed")
        finally:
            pic2txt._easyocr_reader = None

        self.assertEqual(len(captured_calls), 1, "Reader should be instantiated only once")


# ── extract: routing by file suffix ───────────────────────────────────────────

class TestExtractRouting(unittest.TestCase):

    def test_pdf_routes_to_extract_pdf(self):
        p = Path("dummy.pdf")
        with patch("pic2txt.extract_pdf", return_value=[(1, "ok")]) as mock:
            result = pic2txt.extract(p)
        mock.assert_called_once_with(p)
        self.assertEqual(result, [(1, "ok")])

    def test_png_routes_to_extract_image_file(self):
        p = Path("dummy.png")
        with patch("pic2txt.extract_image_file", return_value=[(1, "img")]) as mock:
            result = pic2txt.extract(p)
        mock.assert_called_once_with(p)

    def test_gif_routes_to_extract_image_file(self):
        p = Path("anim.gif")
        with patch("pic2txt.extract_image_file", return_value=[(1, "g")]) as mock:
            pic2txt.extract(p)
        mock.assert_called_once()

    def test_tiff_routes_to_extract_image_file(self):
        p = Path("scan.tiff")
        with patch("pic2txt.extract_image_file", return_value=[(1, "t")]) as mock:
            pic2txt.extract(p)
        mock.assert_called_once()

    def test_tif_routes_to_extract_image_file(self):
        p = Path("scan.tif")
        with patch("pic2txt.extract_image_file", return_value=[(1, "t")]) as mock:
            pic2txt.extract(p)
        mock.assert_called_once()

    def test_unsupported_raises_value_error(self):
        with self.assertRaises(ValueError):
            pic2txt.extract(Path("doc.docx"))


# ── extract_pdf: text layer vs OCR, multi-page ────────────────────────────────

class TestExtractPdf(unittest.TestCase):

    def test_text_layer_page_not_ocrd(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "text.pdf"
            make_pdf(pdf, ["Long enough text layer here"])
            with patch("pic2txt.ocr_image") as mock_ocr:
                pages = pic2txt.extract_pdf(pdf)
            mock_ocr.assert_not_called()
        self.assertEqual(len(pages), 1)
        self.assertIn("Long enough text layer here", pages[0][1])

    def test_blank_page_falls_back_to_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "blank.pdf"
            make_pdf(pdf, [""])        # blank page → text layer empty
            with patch("pic2txt.ocr_image", return_value="ocr result"):
                pages = pic2txt.extract_pdf(pdf)
        self.assertEqual(pages[0][1], "ocr result")

    def test_multi_page_returns_correct_count(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "multi.pdf"
            make_pdf(pdf, ["Page one content here", "Page two content here", "Page three text"])
            pages = pic2txt.extract_pdf(pdf)
        self.assertEqual(len(pages), 3)

    def test_multi_page_numbers_are_sequential(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "multi.pdf"
            make_pdf(pdf, ["First page text here", "Second page text here"])
            pages = pic2txt.extract_pdf(pdf)
        self.assertEqual([n for n, _ in pages], [1, 2])

    def test_mixed_text_and_blank_pages(self):
        """Text pages use text layer; blank pages use OCR."""
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "mixed.pdf"
            make_pdf(pdf, ["Real text content here", ""])   # p1 text, p2 blank
            ocr_calls = []
            with patch("pic2txt.ocr_image", side_effect=lambda img: (ocr_calls.append(1) or "ocr")):
                pages = pic2txt.extract_pdf(pdf)
        self.assertEqual(len(pages), 2)
        self.assertEqual(len(ocr_calls), 1)   # only the blank page triggered OCR


# ── extract_image_file: frames ────────────────────────────────────────────────

class TestExtractImageFile(unittest.TestCase):

    def test_single_frame_png_returns_one_page(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "single.png"
            make_white_png(img)
            with patch("pic2txt.ocr_image", return_value="hello"):
                pages = pic2txt.extract_image_file(img)
        self.assertEqual(pages, [(1, "hello")])

    def test_multi_frame_gif_returns_one_page_per_frame(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            gif = Path(td) / "anim.gif"
            frames = [Image.new("RGB", (32, 32), c) for c in ["red", "blue", "green"]]
            frames[0].save(gif, save_all=True, append_images=frames[1:], loop=0)
            with patch("pic2txt.ocr_image", return_value="frame"):
                pages = pic2txt.extract_image_file(gif)
        self.assertEqual(len(pages), 3)
        self.assertEqual([n for n, _ in pages], [1, 2, 3])

    def test_ocr_result_stripped(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "img.png"
            make_white_png(img)
            with patch("pic2txt.ocr_image", return_value="  hello world  \n"):
                pages = pic2txt.extract_image_file(img)
        self.assertEqual(pages[0][1], "hello world")

    def test_tiff_image_processed(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            tiff = Path(td) / "scan.tiff"
            Image.new("RGB", (64, 32), color="white").save(tiff)
            with patch("pic2txt.ocr_image", return_value="tiff text"):
                pages = pic2txt.extract_image_file(tiff)
        self.assertEqual(pages[0][1], "tiff text")


# ── convert ───────────────────────────────────────────────────────────────────

class TestConvert(unittest.TestCase):

    def test_output_file_created(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "scan.png"
            make_white_png(img)
            with patch("pic2txt.ocr_image", return_value="text"):
                result = pic2txt.convert(img, Path(td) / "out")
            self.assertTrue(result.exists())
            self.assertEqual(result.name, "scan.md")

    def test_title_cased_from_stem(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "my_scan_2024.png"
            make_white_png(img)
            out_dir = Path(td) / "out"
            with patch("pic2txt.ocr_image", return_value="text"):
                pic2txt.convert(img, out_dir)
            md = (out_dir / "my_scan_2024.md").read_text()
            self.assertIn("# My Scan 2024", md)

    def test_hyphen_in_stem_becomes_space(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "annual-report.png"
            make_white_png(img)
            out_dir = Path(td) / "out"
            with patch("pic2txt.ocr_image", return_value="text"):
                pic2txt.convert(img, out_dir)
            md = (out_dir / "annual-report.md").read_text()
            self.assertIn("# Annual Report", md)

    def test_output_dir_created_if_missing(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "x.png"
            make_white_png(img)
            out_dir = Path(td) / "deep" / "nested"
            with patch("pic2txt.ocr_image", return_value="text"):
                pic2txt.convert(img, out_dir)
            self.assertTrue(out_dir.is_dir())

    def test_pdf_text_layer_not_ocrd(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "report.pdf"
            make_pdf(pdf, ["Invoice Total Amount 12345"])
            out_dir = Path(td) / "out"
            with patch("pic2txt.ocr_image") as mock_ocr:
                pic2txt.convert(pdf, out_dir)
            mock_ocr.assert_not_called()
            md = (out_dir / "report.md").read_text()
            self.assertIn("Invoice Total Amount 12345", md)

    def test_convert_returns_path_object(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "x.png"
            make_white_png(img)
            with patch("pic2txt.ocr_image", return_value=""):
                result = pic2txt.convert(img, Path(td) / "out")
            self.assertIsInstance(result, Path)

    def test_extract_error_propagates(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "bad.png"
            make_white_png(img)
            with patch("pic2txt.extract", side_effect=RuntimeError("no backend")):
                with self.assertRaises(RuntimeError):
                    pic2txt.convert(img, Path(td) / "out")


# ── main(): CLI paths ─────────────────────────────────────────────────────────

class TestMain(unittest.TestCase):

    def _run_main(self, argv):
        """Run main() with given sys.argv; return SystemExit code or None."""
        with patch("sys.argv", ["pic2txt.py"] + argv):
            try:
                pic2txt.main()
                return None
            except SystemExit as e:
                return e.code

    def test_no_args_input_dir_missing_exits_1(self):
        with patch("pic2txt.INPUT_DIR", Path("/nonexistent/absent")):
            code = self._run_main([])
        self.assertEqual(code, 1)

    def test_no_files_found_exits_0(self):
        with tempfile.TemporaryDirectory() as td:
            code = self._run_main([td])   # empty dir
        self.assertEqual(code, 0)

    def test_convert_error_exits_1(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "bad.png"
            make_white_png(img)
            out_dir = Path(td) / "out"
            with patch("pic2txt.OUTPUT_DIR", out_dir):
                with patch("pic2txt.convert", side_effect=RuntimeError("fail")):
                    code = self._run_main([str(img)])
        self.assertEqual(code, 1)

    def test_successful_conversion_exits_none(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "ok.png"
            make_white_png(img)
            out_dir = Path(td) / "out"
            with patch("pic2txt.OUTPUT_DIR", out_dir):
                with patch("pic2txt.convert", return_value=out_dir / "ok.md"):
                    code = self._run_main([str(img)])
        self.assertIsNone(code)

    def test_no_args_uses_input_dir(self):
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "img.png"
            make_white_png(img)
            out_dir = Path(td) / "out"
            with patch("pic2txt.INPUT_DIR", Path(td)):
                with patch("pic2txt.OUTPUT_DIR", out_dir):
                    with patch("pic2txt.convert", return_value=out_dir / "img.md") as mock_conv:
                        self._run_main([])
            mock_conv.assert_called_once()


# ── integration: real image OCR (requires pytesseract or easyocr) ─────────────

class TestRealOcr(unittest.TestCase):

    def _skip_if_no_backend(self):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return
        except Exception:
            pass
        try:
            import easyocr  # noqa
            return
        except ImportError:
            pass
        self.skipTest("No OCR backend available")

    def test_white_image_returns_string(self):
        """OCR on a blank image must not crash and must return str."""
        self._skip_if_no_backend()
        from PIL import Image
        img = Image.new("RGB", (200, 50), color="white")
        result = pic2txt.ocr_image(img)
        self.assertIsInstance(result, str)

    def test_end_to_end_text_pdf(self):
        """Text-layer PDF → markdown without any OCR backend."""
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "hello.pdf"
            make_pdf(pdf, ["Hello World document text"])
            out_dir = Path(td) / "md"
            result = pic2txt.convert(pdf, out_dir)
            self.assertTrue(result.exists())
            self.assertIn("Hello World document text", result.read_text())

    def test_end_to_end_multi_page_text_pdf(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "multi.pdf"
            make_pdf(pdf, ["Chapter one content", "Chapter two content", "Chapter three"])
            out_dir = Path(td) / "md"
            result = pic2txt.convert(pdf, out_dir)
            md = result.read_text()
            self.assertIn("<!-- page 1 -->", md)
            self.assertIn("<!-- page 2 -->", md)
            self.assertIn("<!-- page 3 -->", md)
            self.assertIn("Chapter one content", md)
            self.assertIn("Chapter two content", md)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
