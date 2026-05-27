"""
Unit tests for hrm_ocr.pipeline.pdf_router
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hrm_ocr.pipeline.pdf_router import DocumentRoute, route_document


class TestPDFRouter:
    @patch("hrm_ocr.pipeline.pdf_router.Path.read_bytes")
    @patch("hrm_ocr.pipeline.pdf_router.Path.exists")
    def test_jpeg_routes_to_image(self, mock_exists, mock_read_bytes):
        mock_exists.return_value = True
        mock_read_bytes.return_value = b"fake_jpeg_bytes"
        
        route = route_document("dummy.jpg", mime_type="image/jpeg")
        assert route.type == "image"
        assert route.extraction_method == "ocr"
        assert route.parsability_score == 0.0
        assert route.routing_reason == "image_input"
        
    @patch("hrm_ocr.pipeline.pdf_router._extract_pdf_text_first_3_pages")
    def test_digital_pdf_routes_to_text(self, mock_extract):
        # 3 checks must pass:
        # 1. >= 100 chars
        # 2. >= 0.85 printable ratio
        # 3. >= 3 lines with 4+ words
        valid_text = (
            "This is a line with four words.\n"
            "Here is another line with many words in it.\n"
            "And a third line that has enough words too.\n"
            + "A" * 100  # Ensure > 100 chars
        )
        mock_extract.return_value = (valid_text, 1)
        
        route = route_document(b"fake_pdf_bytes", mime_type="application/pdf")
        assert route.type == "text"
        assert route.extraction_method == "text_layer"
        assert route.parsability_score == 1.0
        assert route.routing_reason == "all_checks_passed"
        
    @patch("hrm_ocr.pipeline.pdf_router._extract_pdf_text_first_3_pages")
    @patch("hrm_ocr.pipeline.pdf_router.convert_from_bytes")
    def test_scanned_pdf_routes_to_image(self, mock_convert, mock_extract):
        # Empty text fails all checks
        mock_extract.return_value = ("", 1)
        
        # Mock pdf2image
        mock_img = MagicMock()
        mock_img.convert.return_value = MagicMock()
        # Mock numpy array conversion
        with patch("hrm_ocr.pipeline.pdf_router.np.array", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
            route = route_document(b"fake_pdf_bytes", mime_type="application/pdf")
        
        assert route.type == "image"
        assert route.extraction_method == "ocr"
        assert route.parsability_score == 0.0
        assert "min_chars" in route.routing_reason
        assert "min_text_lines" in route.routing_reason
        
    @patch("hrm_ocr.pipeline.pdf_router._extract_pdf_text_first_3_pages")
    @patch("hrm_ocr.pipeline.pdf_router.convert_from_bytes")
    def test_pdf_with_junk_text_routes_to_image(self, mock_convert, mock_extract):
        # Less than 100 chars
        mock_extract.return_value = ("Junk text\nOnly two lines", 1)
        
        with patch("hrm_ocr.pipeline.pdf_router.np.array", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
            route = route_document(b"fake_pdf_bytes", mime_type="application/pdf")
            
        assert route.type == "image"
        # It passes printable ratio (1.0), but fails min_chars and min_text_lines
        # Score = 1/3 = 0.333
        assert abs(route.parsability_score - 0.333) < 0.01
        assert "min_chars" in route.routing_reason
