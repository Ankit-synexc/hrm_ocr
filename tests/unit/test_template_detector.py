"""
tests/unit/test_template_detector.py
====================================
Unit tests for the rule-based template detector.
"""
from __future__ import annotations

import cv2
import numpy as np

from hrm_ocr.models.template_detector import detect_template


def _create_mock_image(color_bgr: tuple[int, int, int], width: int, height: int) -> np.ndarray:
    """Create a solid color image for testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color_bgr
    return img


def _add_mock_qr(img: np.ndarray) -> np.ndarray:
    """Add a noisy black and white pattern in the bottom right to simulate a QR code."""
    h, w = img.shape[:2]
    # QR code is roughly 200x200 in the bottom right
    qr_size = 200
    qr = np.random.randint(0, 2, (qr_size, qr_size, 3), dtype=np.uint8) * 255
    img[h-qr_size:h, w-qr_size:w] = qr
    return img


class TestTemplateDetector:
    def test_aadhaar_v4_portrait_aspect_ratio(self):
        # portrait card ~ 0.63
        img = _create_mock_image((255, 255, 255), 638, 1012)
        result = detect_template(img)
        assert result.doc_type == "aadhaar"
        assert result.template_version == "aadhaar_v4"
        assert result.detection_method == "aspect_ratio_portrait"
        
    def test_aadhaar_v1_blue_no_qr(self):
        # Blue background (BGR)
        img = _create_mock_image((255, 200, 150), 1012, 638)
        result = detect_template(img)
        assert result.doc_type == "aadhaar"
        assert result.template_version == "aadhaar_v1"
        assert "hsv_blue_no_qr" in result.detection_method
        assert len(result.field_coordinate_map) > 0
        
    def test_pan_v1_cream(self):
        # Cream background (BGR roughly #FFF5E1 -> 225, 245, 255)
        img = _create_mock_image((225, 245, 255), 1012, 638)
        result = detect_template(img)
        assert result.doc_type == "pan"
        assert result.template_version == "pan_v1"
        assert "hsv_cream" in result.detection_method
        assert len(result.field_coordinate_map) > 0

    def test_aadhaar_v3_white(self):
        # White background
        img = _create_mock_image((255, 255, 255), 1012, 638)
        # We simulate a QR code detection fallback by passing the regex text override
        # because the random noise QR doesn't always trigger cv2.QRCodeDetector.
        result = detect_template(img, extracted_text="1234 5678 9012")
        assert result.doc_type == "aadhaar"
        assert result.template_version == "aadhaar_v3"

    def test_pan_v3_white_with_text(self):
        # White background
        img = _create_mock_image((255, 255, 255), 1012, 638)
        # Passes the PAN regex
        result = detect_template(img, extracted_text="ABCDE1234F")
        assert result.doc_type == "pan"
        # PAN v2 without QR
        assert result.template_version == "pan_v2" 

    def test_unknown_garbage(self):
        # Random noise, weird aspect ratio
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = detect_template(img)
        assert result.doc_type == "unknown"
        assert result.confidence == 0.0
