"""
tests/unit/test_ocr_engine.py
=============================
Unit tests for the PaddleOCR engine wrapper.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from hrm_ocr.models.ocr_engine import OCREngine, get_engine


class TestOCREngine:
    @patch("hrm_ocr.models.ocr_engine.PaddleOCR")
    def test_engine_initialization(self, mock_paddle):
        """Engine should initialize without error."""
        engine = OCREngine(model_dir=MagicMock(), lang="en")
        assert engine.lang == "en"
        mock_paddle.assert_called_once()
        
    @patch("hrm_ocr.models.ocr_engine.PaddleOCR")
    def test_recognize_full_card(self, mock_paddle):
        # Mock PaddleOCR return structure:
        # [ [ [bbox], ('text', conf) ] ]
        mock_instance = MagicMock()
        mock_instance.ocr.return_value = [
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]], ("HELLO", 0.95)
            ]
        ]
        mock_paddle.return_value = mock_instance
        
        engine = OCREngine(model_dir=MagicMock(), lang="en")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        regions = engine.recognize_full_card(img)
        assert len(regions) == 1
        assert regions[0].text == "HELLO"
        assert abs(regions[0].confidence - 0.95) < 0.01

    @patch("hrm_ocr.models.ocr_engine.PaddleOCR")
    def test_recognize_field_crop(self, mock_paddle):
        mock_instance = MagicMock()
        # Mock two lines of text in a crop
        mock_instance.ocr.return_value = [
            [
                [[0, 0], [10, 0], [10, 10], [0, 10]], ("JOHN", 0.90)
            ],
            [
                [[0, 15], [10, 15], [10, 25], [0, 25]], ("DOE", 0.80)
            ]
        ]
        mock_paddle.return_value = mock_instance
        
        engine = OCREngine(model_dir=MagicMock(), lang="en")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = engine.recognize_field_crop(img)
        
        # text should be concatenated
        assert "JOHN" in result.text
        assert "DOE" in result.text
        
        # confidence should be length-weighted mean:
        # "JOHN" len=4, conf=0.9
        # "DOE" len=3, conf=0.8
        # sum = 4*0.9 + 3*0.8 = 3.6 + 2.4 = 6.0
        # total_weight = 7
        # final = 6.0 / 7 = 0.857
        assert abs(result.confidence - 0.857) < 0.01

    @patch("hrm_ocr.models.ocr_engine.PaddleOCR")
    def test_get_engine_singleton(self, mock_paddle):
        """Ensure the engine registry works as a singleton per language."""
        engine1 = get_engine("en")
        engine2 = get_engine("en")
        assert engine1 is engine2
        
        engine3 = get_engine("hi")
        assert engine1 is not engine3
