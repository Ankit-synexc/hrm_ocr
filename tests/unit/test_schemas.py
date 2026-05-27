"""
tests/unit/test_schemas.py
==========================
Unit tests for the Pydantic API response schemas.
"""
from __future__ import annotations

from hrm_ocr.api.schemas.response import (
    AadhaarFields,
    CVFields,
    PANFields,
    _mask_uid,
    build_response,
)
from hrm_ocr.validation.router import RoutingDecision
from hrm_ocr.validation.validators import ValidationResult


class TestResponseSchemas:
    def test_uid_masking(self):
        assert _mask_uid("123456789012") == "XXXX XXXX 9012"
        assert _mask_uid("1234 5678 9012") == "XXXX XXXX 9012"
        # If it's malformed, leave it or handle it cleanly
        assert _mask_uid("123") == "123"

    def test_build_response_aadhaar(self):
        routing = {
            "uid": RoutingDecision(
                "uid", "123456789012", 0.95, "accept", None, "12345678901Z", "123456789012", True, 1, "ocr"
            ),
            "name": RoutingDecision(
                "name", "John Doe", 0.99, "accept", None, "John Doe", "John Doe", False, 1, "ocr"
            )
        }
        validation = {
            "uid": ValidationResult(True),
            "name": ValidationResult(True)
        }
        
        response = build_response(
            routing=routing,
            validation=validation,
            corrections={},
            extraction_method="ocr",
            request_id="req_123",
            processing_time_ms=150.0,
            doc_type="aadhaar",
            template_version="aadhaar_v3"
        )
        
        assert response.doc_type == "aadhaar"
        assert response.meta.correction_count == 1
        assert response.validation.checksum_passed is True
        
        # Test masking
        assert isinstance(response.fields, AadhaarFields)
        assert response.fields.uid is not None
        assert response.fields.uid.value == "XXXX XXXX 9012"
        
        # Test flag creation
        assert len(response.flags) == 0

    def test_build_response_pan_with_flags(self):
        routing = {
            "pan_number": RoutingDecision(
                "pan_number", "ABCDE1234F", 0.80, "flag", "val_failed", "ABCDE1234F", "ABCDE1234F", False, 2, "ocr"
            )
        }
        validation = {
            "pan_number": ValidationResult(False, "Regex mismatch")
        }
        
        response = build_response(
            routing=routing,
            validation=validation,
            corrections={},
            extraction_method="ocr",
            request_id="req_456",
            processing_time_ms=50.0,
            doc_type="pan"
        )
        
        assert response.doc_type == "pan"
        assert isinstance(response.fields, PANFields)
        assert response.validation.checksum_passed is False
        
        assert len(response.flags) == 1
        assert response.flags[0].field_name == "pan_number"
        assert response.flags[0].flag_reason == "val_failed"

    def test_build_response_cv(self):
        routing = {
            "email": RoutingDecision(
                "email", "test@test.com", 1.0, "accept", None, "test@test.com", "test@test.com", False, 1, "text_layer"
            )
        }
        validation = {
            "email": ValidationResult(True)
        }
        
        response = build_response(
            routing=routing,
            validation=validation,
            corrections={},
            extraction_method="text_layer",
            request_id="req_789",
            processing_time_ms=10.0,
            doc_type="cv"
        )
        
        assert response.doc_type == "cv"
        assert isinstance(response.fields, CVFields)
        assert response.meta.extraction_method == "text_layer"
        assert response.meta.processing_time_ms == 10.0
