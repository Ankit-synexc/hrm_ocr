"""
tests/unit/test_router.py
=========================
Unit tests for the confidence escalation router.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from hrm_ocr.correction.patterns import CorrectionResult
from hrm_ocr.validation.router import route_field
from hrm_ocr.validation.validators import ValidationResult


class TestRouter:
    def test_text_layer_accept(self):
        val = ValidationResult(True)
        corr = CorrectionResult("original", "corrected", True, "desc")
        
        decision = route_field(
            "name", None, "raw", 0.0, corr, val, None, "text_layer"
        )
        assert decision.status == "accept"
        assert decision.confidence == 1.0

    def test_text_layer_flag(self, tmp_path: Path):
        val = ValidationResult(False, "error")
        corr = CorrectionResult("original", "corrected", False, "")
        
        # Override log directory behavior by patching Path resolution?
        # Instead, we just let it write to the repo logs/ dir which is fine, 
        # or we test the return object directly.
        decision = route_field(
            "name", None, "raw", 0.0, corr, val, None, "text_layer"
        )
        assert decision.status == "flag"
        assert decision.flag_reason == "validation_failed_text_layer"

    def test_ocr_high_confidence_validation_fail_escalates(self):
        val = ValidationResult(False, "error")
        corr = CorrectionResult("raw", "raw", False, "")
        
        decision = route_field(
            "uid", np.zeros((32, 32, 3)), "raw", 0.95, corr, val, MagicMock(), "ocr"
        )
        
        assert decision.status == "escalate"
        assert decision.flag_reason == "high_conf_validation_fail"

    def test_ocr_medium_confidence_retry_success(self):
        val_fail = ValidationResult(False, "error")
        corr_fail = CorrectionResult("bad", "bad", False, "")
        
        # Mock engine to return a successful retry
        mock_engine = MagicMock()
        mock_res = MagicMock()
        mock_res.text = "ABCDE1234F"
        mock_res.confidence = 0.95
        mock_engine.recognize_field_crop.return_value = mock_res
        
        decision = route_field(
            "pan", np.zeros((100, 100, 3)), "bad", 0.85, corr_fail, val_fail, mock_engine, "ocr"
        )
        
        assert decision.status == "accept"
        assert decision.value == "ABCDE1234F"
        assert decision.attempts == 2

    def test_ocr_low_confidence_sharpening_fail_flags(self):
        val_fail = ValidationResult(False, "error")
        corr_fail = CorrectionResult("bad", "bad", False, "")
        
        # Mock engine to return another failing retry
        mock_engine = MagicMock()
        mock_res = MagicMock()
        mock_res.text = "bad"
        mock_res.confidence = 0.60
        mock_engine.recognize_field_crop.return_value = mock_res
        
        decision = route_field(
            "pan", np.zeros((100, 100, 3)), "bad", 0.50, corr_fail, val_fail, mock_engine, "ocr"
        )
        
        assert decision.status == "flag"
        assert decision.flag_reason == "low_conf_validation_fail"
        assert decision.attempts == 2
