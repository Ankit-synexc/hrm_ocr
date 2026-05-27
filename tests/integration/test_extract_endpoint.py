"""
tests/integration/test_extract_endpoint.py
==========================================
Integration tests for the POST /extract FastAPI endpoint.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from hrm_ocr.api.main import app


class TestExtractEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_type"] == "paddleocr"

    @patch("hrm_ocr.api.routes.extract._run_pipeline")
    def test_extract_endpoint_success(self, mock_run_pipeline):
        # Mock the pipeline executor to avoid actually running PaddleOCR in basic HTTP tests
        mock_run_pipeline.return_value = {
            "request_id": "req_abc",
            "doc_type": "pan",
            "extraction_method": "ocr",
            "elapsed_ms": 120.5,
            "fields": {
                "pan_number": {
                    "value": "ABCDE1234F",
                    "confidence": 0.98,
                    "status": "accept",
                    "flag_reason": None
                }
            }
        }
        
        # Create a fake image payload
        fake_img_bytes = b"fake_jpeg_data"
        file_like = io.BytesIO(fake_img_bytes)
        
        response = self.client.post(
            "/api/v1/extract",
            files={"file": ("test.jpg", file_like, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["doc_type"] == "pan"
        assert data["elapsed_ms"] < 150.0  # Asserts response payload values
        assert data["extraction_method"] == "ocr"
        assert "pan_number" in data["fields"]
        
        # Check that X-Request-ID was injected
        assert "X-Request-ID" in response.headers

    def test_extract_endpoint_file_too_large(self):
        # Create a dummy payload > 10MB
        large_bytes = b"0" * (11 * 1024 * 1024)
        file_like = io.BytesIO(large_bytes)
        
        response = self.client.post(
            "/api/v1/extract",
            files={"file": ("test.jpg", file_like, "image/jpeg")}
        )
        
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()
