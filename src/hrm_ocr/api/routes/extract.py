"""
hrm_ocr.api.routes.extract
==========================
POST /extract endpoint executing the full OCR and validation pipeline.
"""
from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)
import numpy as np
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

# The following modules are part of the hrm_ocr pipeline architecture.
# We import them to orchestrate the complete end-to-end extraction.
from hrm_ocr.correction.patterns import correct_all_fields, correct_field
from hrm_ocr.feedback.logger import FeedbackLogger
from hrm_ocr.models.glyph_cache import CachedOCREngine
from hrm_ocr.models.ocr_engine import get_engine
from hrm_ocr.models.template_detector import detect_template

from hrm_ocr.pipeline.preprocess import clean_image
from hrm_ocr.validation.router import route_field
from hrm_ocr.validation.validators import validate_fields
from hrm_ocr.api.schemas.response import build_response

# Import pdf_router conditionally, but do NOT mock it if it fails.
try:
    from hrm_ocr.pipeline.pdf_router import route_document
    from hrm_ocr.pipeline.text_extractor import extract_fields_from_text
except ImportError:
    route_document = None
    extract_fields_from_text = None

router = APIRouter()

# Instantiate the FeedbackLogger globally for the router
feedback_logger = FeedbackLogger()

# correct_all_fields is imported from hrm_ocr.correction.patterns


def _run_pipeline(raw_bytes: bytes, mime_type: str, request_id: str) -> dict[str, Any]:
    """Synchronous pipeline execution meant to be run in an executor."""
    import cv2
    start_time = time.perf_counter()
    
    doc_type = "unknown"
    template_version = None
    
    fields_raw: dict[str, str] = {}
    corrections: dict[str, Any] = {}
    raw_ocr_results: dict[str, Any] = {}
    extraction_method = "ocr"
    
    # Pre-fetch the cached engine wrapper
    engine = get_engine("en")
    cached_engine = CachedOCREngine(engine)
    
    crop_cache: dict[str, Any] = {} # Store crops for the router
    
    # Bypass PDF router entirely if the file is an image
    if mime_type.startswith("image/"):
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        page_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if page_img is None:
            raise ValueError("Failed to decode uploaded image bytes.")
            
        img = clean_image(page_img)
        
        # 1. Run Full Page ML Extraction FIRST
        logger.info("Running full page ML text extraction for accurate classification...")
        full_regions = cached_engine.engine.recognize_full_card(page_img)
        full_text = " ".join([r.text for r in full_regions]) if full_regions else ""
        
        # 2. Detect template using the extracted text to fix any OpenCV errors on uncropped photos
        template = detect_template(img, extracted_text=full_text)
        doc_type = template.doc_type
        template_version = template.template_version
        
        # 3. Run Spatial Parsing using the pre-computed regions
        raw_ocr_results = cached_engine.engine.recognize_spatial_from_regions(full_regions, doc_type)
        
        # 4. If spatial misses fields, fall back to coordinate map
        if not raw_ocr_results or "aadhaar_number" not in raw_ocr_results:
            logger.info("Spatial extraction missed core fields, falling back to coordinate map...")
            raw_ocr_results = cached_engine.engine.recognize_all_fields(img, template.field_coordinate_map)
        
        # Build a crop cache for the router retries (using the downscaled img coords)
        for field, coords in template.field_coordinate_map.items():
            if not field.startswith("__") and len(coords) == 4:
                x1, y1, x2, y2 = coords
                crop_cache[field] = img[y1:y2, x1:x2]
                
    else:
        # Route through PDF parser
        if route_document is None:
            raise RuntimeError("PDF processing libraries (pdfplumber, pdf2image) are not installed.")
            
        route = route_document(raw_bytes)
        if route.type == 'text':
            raw_text = route.raw_text or ""
            
            # Auto-classify digital PDFs
            raw_lower = raw_text.lower()
            import re
            if "experience" in raw_lower and ("skills" in raw_lower or "education" in raw_lower):
                doc_type = "cv"
            elif re.search(r"[a-z]{5}[0-9]{4}[a-z]", raw_lower) or "income tax department" in raw_lower:
                doc_type = "pan"
            elif "aadhaar" in raw_lower or "unique identification" in raw_lower:
                doc_type = "aadhaar"
            else:
                doc_type = "unknown"
                
            result = extract_fields_from_text(raw_text, doc_type)
            fields_raw = result.fields
            extraction_method = 'text_layer'
            doc_type = getattr(result, 'doc_type', doc_type)
        else:
            if not route.pages:
                raise ValueError("No pages found in document route.")
            page_img = route.pages[0]
            img = clean_image(page_img)
            
            # Run full page OCR first, then spatial + template detection
            full_regions = cached_engine.engine.recognize_full_card(page_img)
            full_text = " ".join([r.text for r in full_regions]) if full_regions else ""
            
            template = detect_template(img, extracted_text=full_text)
            doc_type = template.doc_type
            template_version = template.template_version
            
            raw_ocr_results = cached_engine.engine.recognize_spatial_from_regions(full_regions, doc_type)
            if not raw_ocr_results or "aadhaar_number" not in raw_ocr_results:
                raw_ocr_results = cached_engine.engine.recognize_all_fields(img, template.field_coordinate_map)
            for field, coords in template.field_coordinate_map.items():
                if not field.startswith("__") and len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    crop_cache[field] = img[y1:y2, x1:x2]

    # ── Common OCR post-processing (runs for BOTH image and PDF-image paths) ──
    if extraction_method == 'ocr' and raw_ocr_results:
        raw_text_dict = {k: v.text for k, v in raw_ocr_results.items()}
        corrections = correct_all_fields(doc_type, raw_text_dict)
        fields_raw = {k: v.corrected for k, v in corrections.items()}
        cached_engine.reset_session()

    # Common path for both text and OCR routes
    validation = validate_fields(doc_type, fields_raw)
    routing = {}
    
    for field, value in fields_raw.items():
        if extraction_method == 'ocr':
            raw_res = raw_ocr_results.get(field)
            if not raw_res:
                continue
            current_crop = crop_cache.get(field)
            if raw_res.raw_regions:
                xs = [pt[0] for r in raw_res.raw_regions for pt in r.bbox]
                ys = [pt[1] for r in raw_res.raw_regions for pt in r.bbox]
                if xs and ys:
                    pad = 5
                    x1, x2 = max(0, min(xs) - pad), min(img.shape[1], max(xs) + pad)
                    y1, y2 = max(0, min(ys) - pad), min(img.shape[0], max(ys) + pad)
                    current_crop = img[y1:y2, x1:x2]

            decision = route_field(
                field_name=field,
                crop=current_crop,
                raw_text=raw_res.text,
                raw_confidence=raw_res.confidence,
                correction_result=corrections[field],
                validation_result=validation[field],
                cached_engine=cached_engine,
                extraction_method='ocr',
                doc_type=doc_type
            )
        else:
            # Fake correction result for text layer
            from hrm_ocr.correction.patterns import CorrectionResult
            mock_corr = CorrectionResult(
                original=value,
                corrected=value,
                was_changed=False,
                change_description=""
            )
            decision = route_field(
                field_name=field,
                crop=None,
                raw_text=value,
                raw_confidence=1.0,
                correction_result=mock_corr,
                validation_result=validation[field],
                cached_engine=None,
                extraction_method='text_layer',
                doc_type=doc_type
            )
            
        routing[field] = decision
        
        # Log issues via active learning framework
        if decision.status in ('flag', 'escalate'):
            if extraction_method == 'ocr':
                feedback_logger.log_ocr_issue(
                    request_id=request_id,
                    doc_type=doc_type,
                    field_name=field,
                    crop=crop_cache.get(field, np.zeros((10, 10, 3), dtype=np.uint8)),
                    raw_ocr_value=decision.raw_ocr_value,
                    corrected_value=decision.corrected_value,
                    was_corrected=decision.was_corrected,
                    confidence=decision.confidence,
                    flag_reason=decision.flag_reason or "unknown"
                )
            else:
                feedback_logger.log_validation_error(
                    request_id=request_id,
                    doc_type=doc_type,
                    field_name=field,
                    extracted_value=decision.corrected_value,
                    extraction_method='text_layer',
                    validation_reason=decision.flag_reason or "unknown"
                )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return build_response(
        routing=routing,
        validation=validation,
        corrections=corrections,
        extraction_method=extraction_method,
        request_id=request_id,
        processing_time_ms=elapsed_ms,
        doc_type=doc_type,
        template_version=template_version
    ).model_dump()


@router.post("/extract")
async def extract_endpoint(
    request: Request,
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Main extraction endpoint. Accepts JPEG, PNG, PDF (max 10MB).
    Processes image natively using PaddleOCR or directly parses text layers from digital PDFs.
    """
    # 1. Basic validation
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
    allowed_types = ['image/jpeg', 'image/png', 'application/pdf']
    if file.content_type not in allowed_types:
        # Fallback to suffix checking if content_type is missing/generic
        if not any(file.filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.pdf']):
            raise HTTPException(status_code=415, detail="Unsupported media type")
            
    # Read payload
    raw_bytes = await file.read()
    request_id = getattr(request.state, "request_id", "unknown_req")
    
    # 2. Run CPU-bound pipeline in an executor to avoid blocking the async event loop
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, 
            _run_pipeline, 
            raw_bytes, 
            file.content_type, 
            request_id
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")
