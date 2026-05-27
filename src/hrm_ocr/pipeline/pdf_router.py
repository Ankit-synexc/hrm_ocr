"""
hrm_ocr.pipeline.pdf_router
============================
Routes a PDF to either:
  A) Digital text extraction (pdfplumber) — fast, lossless
  B) Image render → OCR fallback (pdf2image + PaddleOCR)

Decision logic
--------------
A PDF is treated as "digital" only when ALL three heuristics pass:
  1. Extracted char count >= pdf_min_chars
  2. Printable-ASCII ratio >= pdf_min_printable_ratio
  3. Non-empty lines with 4+ words count >= 3

Otherwise it is a scanned/image-only PDF and goes through OCR.
"""
from __future__ import annotations

import io
import logging
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pdfplumber
from pdf2image import convert_from_bytes

from hrm_ocr.exceptions import IngestError

logger = logging.getLogger(__name__)


@dataclass
class DocumentRoute:
    type: Literal["text", "image"]
    raw_text: str | None
    pages: list[np.ndarray] | None
    page_count: int
    extraction_method: Literal["text_layer", "ocr"]
    parsability_score: float
    routing_reason: str


def route_document(source: str | bytes | Path, mime_type: str = "application/pdf") -> DocumentRoute:
    """Determine the optimal processing route for a document.
    
    Parameters
    ----------
    source : str | bytes | Path
        The raw bytes or file path of the document.
    mime_type : str
        The MIME type of the document (default 'application/pdf').
        
    Returns
    -------
    DocumentRoute
        Routing decision with either extracted text or rendered image pages.
    """
    if isinstance(source, Path):
        raw = source.read_bytes()
    elif isinstance(source, str):
        p = Path(source)
        if p.exists():
            raw = p.read_bytes()
        else:
            raise IngestError(f"File not found: {source}")
    else:
        raw = source

    if not mime_type.endswith("pdf"):
        return DocumentRoute(
            type="image",
            raw_text=None,
            pages=None,  # Handled upstream by image loader
            page_count=1,
            extraction_method="ocr",
            parsability_score=0.0,
            routing_reason="image_input"
        )

    # PDF routing logic
    pdf_min_chars = 100
    pdf_min_printable_ratio = 0.85
    
    try:
        raw_text, page_count = _extract_pdf_text_first_3_pages(raw)
    except Exception as exc:
        logger.warning("PDF text extraction failed: %s. Falling back to OCR.", exc)
        return _route_to_image(raw, f"failed_checks: [{exc}]")

    checks_passed = 0
    failed_list = []
    
    # Check A
    if len(raw_text.strip()) >= pdf_min_chars:
        checks_passed += 1
    else:
        failed_list.append("min_chars")
        
    # Check B
    printable = set(string.printable)
    ratio = sum(c in printable for c in raw_text) / max(len(raw_text), 1)
    if ratio >= pdf_min_printable_ratio:
        checks_passed += 1
    else:
        failed_list.append("printable_ratio")
        
    # Check C
    lines = raw_text.splitlines()
    lines_4_words = sum(1 for line in lines if len(line.strip().split()) >= 4)
    if lines_4_words >= 3:
        checks_passed += 1
    else:
        failed_list.append("min_text_lines")
        
    parsability_score = checks_passed / 3.0
    
    if checks_passed == 3:
        logger.info("PDF -> text_layer path")
        return DocumentRoute(
            type="text",
            raw_text=raw_text,
            pages=None,
            page_count=page_count,
            extraction_method="text_layer",
            parsability_score=parsability_score,
            routing_reason="all_checks_passed"
        )
    else:
        logger.info("PDF -> OCR path (failed %s)", failed_list)
        return _route_to_image(raw, f"failed_checks: {failed_list}", parsability_score)


def _extract_pdf_text_first_3_pages(pdf_bytes: bytes) -> tuple[str, int]:
    text_blocks = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        for i, page in enumerate(pdf.pages[:3]):
            extracted = page.extract_text()
            if extracted:
                text_blocks.append(extracted)
    return "\n".join(text_blocks), page_count


def _route_to_image(pdf_bytes: bytes, reason: str, score: float = 0.0) -> DocumentRoute:
    import cv2
    try:
        # Render the first page only for now
        pil_images = convert_from_bytes(pdf_bytes, dpi=300, first_page=1, last_page=1)
        pages = []
        for pil_img in pil_images:
            arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return DocumentRoute(
            type="image",
            raw_text=None,
            pages=pages,
            page_count=len(pages),  # Just indicating we have 1 rendered page
            extraction_method="ocr",
            parsability_score=score,
            routing_reason=reason
        )
    except Exception as exc:
        raise IngestError(f"PDF rendering failed: {exc}") from exc
