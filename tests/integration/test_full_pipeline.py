"""
Integration tests for the full extraction pipeline (without HTTP layer).
Mocks only PaddleOCR — all other modules run for real.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


def _make_card_image(w: int = 1012, h: int = 638) -> np.ndarray:
    import cv2
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    return img


class TestFullPipelineNoOCR:
    """Run pipeline components end-to-end without PaddleOCR."""

    def test_ingest_preprocess_detect(self):
        from hrm_ocr.pipeline.ingest import ingest
        from hrm_ocr.pipeline.preprocess import preprocess
        from hrm_ocr.models.template_detector import TemplateDetector

        # Create a minimal test image
        import cv2
        img_np = np.full((638, 1012, 3), 200, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img_np)
        img_bytes = buf.tobytes()

        img = ingest(img_bytes)
        assert img.shape[2] == 3

        processed = preprocess(img)
        assert processed.shape == (638, 1012, 3)

        detector = TemplateDetector()
        template, score = detector.detect(processed)
        assert template in [
            "aadhaar_v1", "aadhaar_v2", "aadhaar_v3", "aadhaar_v4",
            "pan_v1", "pan_v2", "pan_v3",
        ]
        assert 0.0 <= score <= 1.0

    def test_extract_and_correct_with_mock_ocr(self):
        from hrm_ocr.pipeline.text_extractor import extract_fields
        from hrm_ocr.correction.patterns import apply_rules

        img = np.full((638, 1012, 3), 200, dtype=np.uint8)
        field_coords = {
            "aadhaar_number": [100, 310, 660, 380],
            "name": [100, 120, 700, 175],
            "dob": [100, 185, 450, 235],
        }
        mock_engine = MagicMock()
        mock_engine.run_on_crop.side_effect = [
            ("2345 5O78 9O12", 0.88),
            ("RAVI  KUMAR", 0.95),
            ("15/O8/1985", 0.91),
        ]

        raw_fields = extract_fields(img, field_coords, mock_engine)
        corrected = {k: apply_rules(k, v[0]) for k, v in raw_fields.items()}

        assert "O" not in corrected.get("dob", "")
        # Name should have collapsed internal space
        assert "  " not in corrected.get("name", "")
