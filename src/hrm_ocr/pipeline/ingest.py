"""
hrm_ocr.pipeline.ingest
========================
Document ingestion layer: ``load_document(source)`` is the single entry
point for all upstream callers.

Contract
--------
``load_document(source) -> tuple[bytes, str]``

Returns
    (raw_bytes, mime_type)

    ``raw_bytes`` — unmodified file content, ready for downstream decoding.
    ``mime_type`` — canonical MIME type string, one of:
        ``"image/jpeg"``  ``"image/png"``  ``"application/pdf"``

Raises
------
:class:`hrm_ocr.exceptions.FileSizeError`
    File exceeds the configured ``api.max_request_size_mb`` limit (default 10 MB).
:class:`hrm_ocr.exceptions.UnsupportedFormatError`
    MIME type detected from magic bytes is not in the accepted set.
:class:`hrm_ocr.exceptions.CorruptFileError`
    Bytes present but cannot be decoded as a valid image or PDF.
:class:`hrm_ocr.exceptions.IngestError`
    Any other ingestion failure (e.g. unreadable path).

Design notes
------------
* MIME detection uses magic bytes (first 16 bytes), never the filename
  extension — extensions lie, magic bytes don't.
* Size validation happens before any decoding to fail fast on oversized
  uploads without wasting CPU.
* The function is *synchronous*: async I/O is the API layer's concern.
  Callers that need async should wrap with ``asyncio.to_thread``.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

from hrm_ocr.exceptions import (
    CorruptFileError,
    FileSizeError,
    IngestError,
    UnsupportedFormatError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default maximum file size (10 MB). Overridden by ``api.max_request_size_mb``
#: in configs/default.yaml.
DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB

#: MIME types accepted by the pipeline.
SUPPORTED_MIME_TYPES: tuple[str, ...] = (
    "image/jpeg",
    "image/png",
    "application/pdf",
)

#: Magic byte signatures → MIME type.
#: Checked in declaration order; first match wins.
_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "image/jpeg"),                    # JFIF / EXIF JPEG
    (b"\x89PNG\r\n\x1a\n", "image/png"),               # PNG
    (b"%PDF", "application/pdf"),                       # PDF
    # Extended JPEG variants
    (b"\xff\xd8", "image/jpeg"),                        # Bare SOI marker
]

# Type alias accepted by ``load_document``
DocumentSource = Union[bytes, str, Path]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(
    source: DocumentSource,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[bytes, str]:
    """Load a document from any supported source and return (bytes, mime_type).

    Parameters
    ----------
    source:
        * :class:`bytes` — raw file content already in memory.
        * :class:`str`   — filesystem path (absolute or relative).
        * :class:`pathlib.Path` — filesystem path object.
    max_bytes:
        Maximum allowed file size in bytes.  Defaults to 10 MB.
        Pass ``float("inf")`` to disable the check (tests only).

    Returns
    -------
    tuple[bytes, str]
        ``(raw_bytes, mime_type)`` where *mime_type* is one of the strings
        in :data:`SUPPORTED_MIME_TYPES`.

    Raises
    ------
    FileSizeError
        File is larger than *max_bytes*.
    UnsupportedFormatError
        Magic bytes do not match any accepted format.
    CorruptFileError
        Bytes are accepted but cannot be decoded as a valid image/PDF.
    IngestError
        Path does not exist or cannot be read.
    """
    raw = _read_source(source)
    _validate_size(raw, max_bytes)
    mime = _detect_mime(raw)
    _validate_integrity(raw, mime)
    logger.info(
        "load_document: accepted %d bytes as %s",
        len(raw), mime,
    )
    return raw, mime


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_source(source: DocumentSource) -> bytes:
    """Coerce *source* to raw bytes."""
    if isinstance(source, bytes):
        return source

    path = Path(source)
    if not path.exists():
        raise IngestError(
            f"File not found: {path}",
            detail={"path": str(path)},
        )
    if not path.is_file():
        raise IngestError(
            f"Path is not a regular file: {path}",
            detail={"path": str(path)},
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise IngestError(
            f"Cannot read file {path}: {exc}",
            detail={"path": str(path), "os_error": str(exc)},
        ) from exc


def _validate_size(raw: bytes, max_bytes: int) -> None:
    """Raise :class:`FileSizeError` if *raw* exceeds *max_bytes*."""
    if len(raw) > max_bytes:
        raise FileSizeError(size_bytes=len(raw), limit_bytes=max_bytes)


def _detect_mime(raw: bytes) -> str:
    """Return MIME type inferred from magic bytes.

    Raises
    ------
    UnsupportedFormatError
        If no known magic signature matches.
    """
    header = raw[:16]
    for magic, mime in _MAGIC:
        if header.startswith(magic):
            logger.debug("Magic detected: %s → %s", header[:8].hex(), mime)
            return mime
    raise UnsupportedFormatError(
        mime_type=_describe_magic(header),
        supported=list(SUPPORTED_MIME_TYPES),
    )


def _describe_magic(header: bytes) -> str:
    """Return a human-readable description of unknown magic bytes."""
    try:
        text = header[:8].decode("ascii", errors="replace")
        return f"unknown (magic: {header[:8].hex()!r} / {text!r})"
    except Exception:
        return f"unknown (magic: {header[:8].hex()!r})"


def _validate_integrity(raw: bytes, mime: str) -> None:
    """Attempt a lightweight decode to confirm the bytes are not corrupt.

    For images: opens with Pillow (does not decode pixel data — very fast).
    For PDFs:  checks the ``%%EOF`` trailer and ``%PDF`` header.

    Raises
    ------
    CorruptFileError
        If the content cannot be decoded.
    """
    source_label = f"<{mime}>"
    if mime.startswith("image/"):
        _check_image_integrity(raw, source_label)
    elif mime == "application/pdf":
        _check_pdf_integrity(raw, source_label)


def _check_image_integrity(raw: bytes, label: str) -> None:
    """Verify image bytes are decodable by Pillow (header check only, fast)."""
    try:
        from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

        with Image.open(io.BytesIO(raw)) as img:
            img.verify()  # Checks headers/structure; does NOT decode pixels
    except Exception as exc:
        raise CorruptFileError(
            filename=label,
            reason=str(exc),
        ) from exc


def _check_pdf_integrity(raw: bytes, label: str) -> None:
    """Minimal PDF structural check: must start with %PDF and contain %%EOF."""
    if not raw.startswith(b"%PDF"):
        raise CorruptFileError(label, "Missing %PDF header")
    # %%EOF must appear in the last 1024 bytes
    tail = raw[-1024:]
    if b"%%EOF" not in tail and b"%%Eof" not in tail:
        raise CorruptFileError(
            label,
            "PDF appears truncated: no %%EOF marker in final 1024 bytes",
        )
