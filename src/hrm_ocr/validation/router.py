"""
hrm_ocr.validation.router
=========================
Confidence-based escalation router.
Evaluates post-correction and post-validation states to determine if a field
should be accepted, escalated for human review, or flagged as an anomaly.
Supports active retry loops (cropping, sharpening) for low-confidence OCR results.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any

import cv2
import numpy as np

from hrm_ocr.correction.patterns import CorrectionResult, correct_field
from hrm_ocr.models.glyph_cache import CachedOCREngine
from hrm_ocr.validation.validators import ValidationResult, validate_fields

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    field_name: str
    value: str
    confidence: float
    status: Literal['accept', 'escalate', 'flag']
    flag_reason: str | None
    raw_ocr_value: str
    corrected_value: str
    was_corrected: bool
    attempts: int
    extraction_method: Literal['text_layer', 'ocr']


def _log_decision(decision: RoutingDecision) -> None:
    """Log escalations and flags to JSONL for monitoring."""
    if decision.status == 'accept':
        return
        
    repo_root = Path(__file__).resolve().parents[3]
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "routing_decisions.jsonl"
    
    entry = {
        "field_name": decision.field_name,
        "value": decision.value,
        "confidence": decision.confidence,
        "status": decision.status,
        "flag_reason": decision.flag_reason,
        "raw_ocr_value": decision.raw_ocr_value,
        "was_corrected": decision.was_corrected,
        "attempts": decision.attempts,
        "extraction_method": decision.extraction_method,
    }
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def route_field(
    field_name: str,
    crop: np.ndarray | None,
    raw_text: str,
    raw_confidence: float,
    correction_result: CorrectionResult,
    validation_result: ValidationResult,
    cached_engine: CachedOCREngine | None,
    extraction_method: Literal['text_layer', 'ocr'],
    doc_type: str = "unknown"
) -> RoutingDecision:
    """Route a single field through the confidence and validation matrix."""
    
    conf_high = 0.92
    conf_low = 0.70
    attempts = 1
    
    # ---------------------------------------------------------
    # Text-Layer Path
    # ---------------------------------------------------------
    if extraction_method == 'text_layer':
        status: Literal['accept', 'escalate', 'flag'] = 'accept' if validation_result.is_valid else 'flag'
        flag_reason = None if validation_result.is_valid else 'validation_failed_text_layer'
        
        decision = RoutingDecision(
            field_name=field_name,
            value=correction_result.corrected,
            confidence=1.0,  # Native PDF text is absolute
            status=status,
            flag_reason=flag_reason,
            raw_ocr_value=raw_text,
            corrected_value=correction_result.corrected,
            was_corrected=correction_result.was_changed,
            attempts=1,
            extraction_method='text_layer'
        )
        _log_decision(decision)
        return decision

    # ---------------------------------------------------------
    # OCR Path
    # ---------------------------------------------------------
    
    # Initial Accept conditions
    if correction_result.was_changed and validation_result.is_valid:
        return RoutingDecision(
            field_name, correction_result.corrected, raw_confidence, 'accept', None,
            raw_text, correction_result.corrected, True, attempts, 'ocr'
        )
        
    if raw_confidence >= conf_high and validation_result.is_valid:
        return RoutingDecision(
            field_name, correction_result.corrected, raw_confidence, 'accept', None,
            raw_text, correction_result.corrected, correction_result.was_changed, attempts, 'ocr'
        )
        
    if raw_confidence >= conf_high and not validation_result.is_valid:
        decision = RoutingDecision(
            field_name, correction_result.corrected, raw_confidence, 'escalate', 'high_conf_validation_fail',
            raw_text, correction_result.corrected, correction_result.was_changed, attempts, 'ocr'
        )
        _log_decision(decision)
        return decision

    # We need the crop and engine to perform retries
    if crop is None or cached_engine is None:
        # Fallback if dependencies missing for retries
        decision = RoutingDecision(
            field_name, correction_result.corrected, raw_confidence, 'flag', 'missing_retry_deps',
            raw_text, correction_result.corrected, correction_result.was_changed, attempts, 'ocr'
        )
        _log_decision(decision)
        return decision

    h, w = crop.shape[:2]
    
    # MEDIUM Confidence: Retry with cropping
    if conf_low <= raw_confidence < conf_high:
        if validation_result.is_valid:
            return RoutingDecision(
                field_name, correction_result.corrected, raw_confidence, 'accept', None,
                raw_text, correction_result.corrected, correction_result.was_changed, attempts, 'ocr'
            )
            
        # Try cropping by 5px
        if h > 15 and w > 15:
            attempts += 1
            inner_crop = crop[5:-5, 5:-5]
            retry_res = cached_engine.recognize_field_crop(inner_crop)
            retry_corr = correct_field(field_name, retry_res.text)
            retry_val = validate_fields(doc_type, {field_name: retry_corr.corrected})[field_name]
            
            if retry_val.is_valid:
                return RoutingDecision(
                    field_name, retry_corr.corrected, retry_res.confidence, 'accept', None,
                    retry_res.text, retry_corr.corrected, retry_corr.was_changed, attempts, 'ocr'
                )
                
        # Escalate if retry failed
        decision = RoutingDecision(
            field_name, correction_result.corrected, raw_confidence, 'escalate', 'medium_conf_validation_fail',
            raw_text, correction_result.corrected, correction_result.was_changed, attempts, 'ocr'
        )
        _log_decision(decision)
        return decision
        
    # LOW Confidence: Retry with sharpening
    if raw_confidence < conf_low:
        attempts += 1
        
        # Simple cv2 sharpening kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        sharp_crop = cv2.filter2D(crop, -1, kernel)
        
        retry_res = cached_engine.recognize_field_crop(sharp_crop)
        retry_corr = correct_field(field_name, retry_res.text)
        retry_val = validate_fields(doc_type, {field_name: retry_corr.corrected})[field_name]
        
        if retry_val.is_valid:
            return RoutingDecision(
                field_name, retry_corr.corrected, retry_res.confidence, 'accept', None,
                retry_res.text, retry_corr.corrected, retry_corr.was_changed, attempts, 'ocr'
            )
            
        # Still failing -> flag
        decision = RoutingDecision(
            field_name, retry_corr.corrected, retry_res.confidence, 'flag', 'low_conf_validation_fail',
            retry_res.text, retry_corr.corrected, retry_corr.was_changed, attempts, 'ocr'
        )
        _log_decision(decision)
        return decision

    # Absolute fallback (should never reach here)
    return RoutingDecision(
        field_name, correction_result.corrected, raw_confidence, 'flag', 'unreachable',
        raw_text, correction_result.corrected, correction_result.was_changed, attempts, 'ocr'
    )
