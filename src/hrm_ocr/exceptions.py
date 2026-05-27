"""
hrm_ocr.exceptions
==================
Typed exception hierarchy for the HRM OCR pipeline.

All public exceptions inherit from ``HRMOCRError`` so callers can catch
the entire family with a single ``except HRMOCRError``, or target
specific failure modes with the concrete subclasses.

Design rule: every exception carries a human-readable ``message`` and
an optional ``detail`` dict for structured logging / API error bodies.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class HRMOCRError(Exception):
    """Root exception for all HRM OCR pipeline errors."""

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = detail or {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.message!r})"


# ---------------------------------------------------------------------------
# Ingestion errors
# ---------------------------------------------------------------------------

class IngestError(HRMOCRError):
    """Raised when a document cannot be ingested."""


class FileSizeError(IngestError):
    """Raised when the uploaded file exceeds the configured size limit.

    Parameters
    ----------
    size_bytes : int
        Actual file size in bytes.
    limit_bytes : int
        Configured maximum size in bytes.
    """

    def __init__(self, size_bytes: int, limit_bytes: int) -> None:
        size_mb = size_bytes / (1024 * 1024)
        limit_mb = limit_bytes / (1024 * 1024)
        super().__init__(
            f"File size {size_mb:.2f} MB exceeds limit of {limit_mb:.0f} MB.",
            detail={"size_bytes": size_bytes, "limit_bytes": limit_bytes},
        )
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes


class UnsupportedFormatError(IngestError):
    """Raised when the MIME type / file magic is not supported.

    Parameters
    ----------
    mime_type : str
        Detected or declared MIME type.
    supported : list[str]
        List of MIME types the pipeline accepts.
    """

    def __init__(self, mime_type: str, supported: list[str] | None = None) -> None:
        supported = supported or ["image/jpeg", "image/png", "application/pdf"]
        super().__init__(
            f"Unsupported format '{mime_type}'. Accepted: {', '.join(supported)}.",
            detail={"mime_type": mime_type, "supported": supported},
        )
        self.mime_type = mime_type


class CorruptFileError(IngestError):
    """Raised when bytes are present but cannot be decoded as a valid image/PDF.

    Parameters
    ----------
    filename : str
        Filename or description of the source.
    reason : str
        Underlying error message from the decoder.
    """

    def __init__(self, filename: str, reason: str) -> None:
        super().__init__(
            f"Cannot decode '{filename}': {reason}",
            detail={"filename": filename, "reason": reason},
        )
        self.filename = filename
        self.reason = reason


# ---------------------------------------------------------------------------
# Preprocessing errors
# ---------------------------------------------------------------------------

class PreprocessError(HRMOCRError):
    """Raised when a preprocessing step fails unrecoverably."""


# ---------------------------------------------------------------------------
# OCR errors
# ---------------------------------------------------------------------------

class OCRError(HRMOCRError):
    """Raised when the OCR engine fails on a valid image."""


# ---------------------------------------------------------------------------
# Template / detection errors
# ---------------------------------------------------------------------------

class TemplateDetectionError(HRMOCRError):
    """Raised when no template can be matched with sufficient confidence."""


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class ValidationError(HRMOCRError):
    """Raised when extracted field values fail business-rule validation."""
