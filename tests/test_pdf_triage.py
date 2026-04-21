import os
import sys
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("DISCORD_TOKEN", "test-token")

if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")
    httpx_stub.AsyncClient = object
    httpx_stub.Limits = object
    httpx_stub.Response = object
    httpx_stub.TimeoutException = Exception
    sys.modules["httpx"] = httpx_stub

if "bs4" not in sys.modules:
    bs4_stub = types.ModuleType("bs4")
    bs4_stub.BeautifulSoup = object
    sys.modules["bs4"] = bs4_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv_stub

from tools import pdf_parser, scraper


class PdfTriageTests(unittest.TestCase):
    def test_low_memory_profile_never_escalates_marker(self):
        stats = {
            "pages_seen": 2,
            "pages_with_text": 0,
            "text_chars": 0,
            "avg_chars_per_page": 0.0,
            "text_coverage": 0.0,
            "total_images": 2,
            "image_only_pages": 2,
        }

        with patch.object(pdf_parser, "ENABLE_MARKER", True):
            should_escalate, reason = pdf_parser._should_escalate_to_marker(
                stats,
                filepath=__file__,
                source_bytes=512 * 1024,
                resource_profile="low-memory",
            )

        self.assertFalse(should_escalate)
        self.assertEqual(reason, "low-memory-disables-marker")

    def test_balanced_profile_escalates_for_small_scanned_pdf(self):
        stats = {
            "pages_seen": 3,
            "pages_with_text": 0,
            "text_chars": 0,
            "avg_chars_per_page": 0.0,
            "text_coverage": 0.0,
            "total_images": 4,
            "image_only_pages": 3,
        }

        with patch.object(pdf_parser, "ENABLE_MARKER", True):
            should_escalate, reason = pdf_parser._should_escalate_to_marker(
                stats,
                filepath=__file__,
                source_bytes=2 * 1024 * 1024,
                resource_profile="balanced",
            )

        self.assertTrue(should_escalate)
        self.assertEqual(reason, "image-only-pages-detected")

    def test_balanced_profile_keeps_lightweight_when_text_is_good(self):
        stats = {
            "pages_seen": 4,
            "pages_with_text": 4,
            "text_chars": 4000,
            "avg_chars_per_page": 1000.0,
            "text_coverage": 1.0,
            "total_images": 0,
            "image_only_pages": 0,
        }

        with patch.object(pdf_parser, "ENABLE_MARKER", True):
            should_escalate, reason = pdf_parser._should_escalate_to_marker(
                stats,
                filepath=__file__,
                source_bytes=3 * 1024 * 1024,
                resource_profile="balanced",
            )

        self.assertFalse(should_escalate)
        self.assertEqual(reason, "lightweight-text-sufficient")

    def test_extract_markdown_uses_lightweight_result_when_triage_rejects_marker(self):
        light_stats = {
            "pages_seen": 4,
            "pages_with_text": 4,
            "text_chars": 4000,
            "avg_chars_per_page": 1000.0,
            "text_coverage": 1.0,
            "total_images": 0,
            "image_only_pages": 0,
        }

        with patch.object(pdf_parser, "ENABLE_MARKER", True), \
             patch.object(pdf_parser, "_extract_with_pymupdf", return_value=("light text", light_stats)), \
             patch.object(pdf_parser, "_extract_with_marker") as marker_extract:
            result = pdf_parser.extract_markdown_from_pdf(
                "/fake/sample.pdf",
                source_bytes=1024,
                resource_profile="balanced",
            )

        self.assertEqual(result, "light text")
        marker_extract.assert_not_called()

    def test_scraper_skips_header_announced_oversized_pdf(self):
        self.assertEqual(scraper._parse_content_length({"Content-Length": "123"}), 123)
        self.assertIsNone(scraper._parse_content_length({"Content-Length": "invalid"}))
        self.assertTrue(scraper._should_skip_pdf_download(scraper.MAX_PDF_SIZE + 1))
        self.assertFalse(scraper._should_skip_pdf_download(scraper.MAX_PDF_SIZE))


if __name__ == "__main__":
    unittest.main()
