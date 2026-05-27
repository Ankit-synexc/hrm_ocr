"""
hrm_ocr.api.schemas.response
============================
Pydantic v2 schemas for the /extract response payload.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Union

from pydantic import BaseModel, Field

from hrm_ocr.validation.router import RoutingDecision
from hrm_ocr.validation.validators import ValidationResult


class FieldValue(BaseModel):
    value: str
    confidence: float
    status: Literal['accept', 'escalate', 'flag']
    flag_reason: str | None = None
    raw_ocr_value: str | None = None
    was_corrected: bool = False
    original_ocr_value: str | None = None  # Alias for compatibility if needed


class AadhaarFields(BaseModel):
    name: FieldValue | None = None
    name_regional: FieldValue | None = None
    dob: FieldValue | None = None
    gender: FieldValue | None = None
    uid: FieldValue | None = None
    address: FieldValue | None = None
    pincode: FieldValue | None = None
    qr_data: FieldValue | None = None


class PANFields(BaseModel):
    name: FieldValue | None = None
    father_name: FieldValue | None = None
    dob: FieldValue | None = None
    pan_number: FieldValue | None = None
    entity_type: FieldValue | None = None


class CVFields(BaseModel):
    name: FieldValue | None = None
    email: FieldValue | None = None
    phone: FieldValue | None = None
    linkedin: FieldValue | None = None
    current_title: FieldValue | None = None
    total_experience_years: FieldValue | None = None
    skills: FieldValue | None = None
    education: FieldValue | None = None


class ValidationSummary(BaseModel):
    all_fields_valid: bool
    checksum_passed: bool | None = None
    per_field: dict[str, dict[str, Any]]  # Maps field_name to ValidationResult dump


class ResponseMeta(BaseModel):
    extraction_method: Literal['text_layer', 'ocr']
    parsability_score: float
    cache_hit_rate: float | None = None
    correction_count: int
    escalation_count: int
    processing_time_ms: float
    ocr_engine: str


class FlagRecord(BaseModel):
    field_name: str
    flag_reason: str | None
    raw_ocr_value: str
    corrected_value: str
    confidence: float


class ExtractResponse(BaseModel):
    request_id: str
    doc_type: Literal['aadhaar', 'pan', 'cv', 'unknown']
    template_version: str | None = None
    overall_confidence: float
    fields: Union[AadhaarFields, PANFields, CVFields, dict[str, FieldValue]]
    validation: ValidationSummary
    flags: list[FlagRecord]
    meta: ResponseMeta


def _mask_uid(uid: str) -> str:
    """Mask UID keeping only the last 4 digits (e.g. XXXX XXXX 1234)."""
    clean_uid = uid.replace(" ", "")
    if len(clean_uid) == 12:
        last4 = clean_uid[-4:]
        return f"XXXX XXXX {last4}"
    return uid


def build_response(
    routing: dict[str, RoutingDecision],
    validation: dict[str, ValidationResult],
    corrections: dict[str, Any],
    extraction_method: Literal['text_layer', 'ocr'],
    request_id: str,
    processing_time_ms: float,
    doc_type: str = "unknown",
    template_version: str | None = None
) -> ExtractResponse:
    """Map pipeline states into the final Pydantic response schema."""
    
    # 1. Map fields
    field_values: dict[str, FieldValue] = {}
    flags: list[FlagRecord] = []
    correction_count = 0
    escalation_count = 0
    total_conf = 0.0
    field_count = 0
    
    for field_name, decision in routing.items():
        val = decision.value
        
        # Map aadhaar_number to uid for Pydantic schema compatibility
        mapped_field_name = "uid" if field_name == "aadhaar_number" else field_name
        
        # Mask Aadhaar UID
        if "uid" in mapped_field_name.lower() or "aadhaar_number" in field_name.lower():
            val = _mask_uid(val)
            
        field_values[mapped_field_name] = FieldValue(
            value=val,
            confidence=decision.confidence,
            status=decision.status,
            flag_reason=decision.flag_reason,
            raw_ocr_value=decision.raw_ocr_value,
            was_corrected=decision.was_corrected,
            original_ocr_value=decision.raw_ocr_value
        )
        
        if decision.was_corrected:
            correction_count += 1
            
        if decision.status == 'escalate':
            escalation_count += 1
            
        if decision.status in ('flag', 'escalate'):
            flags.append(
                FlagRecord(
                    field_name=field_name,
                    flag_reason=decision.flag_reason,
                    raw_ocr_value=decision.raw_ocr_value,
                    corrected_value=decision.value,
                    confidence=decision.confidence
                )
            )
            
        total_conf += decision.confidence
        field_count += 1
        
    overall_confidence = (total_conf / field_count) if field_count > 0 else 0.0
    
    # 2. Select the correct document schema
    if doc_type == 'aadhaar':
        fields_model = AadhaarFields(**field_values)
    elif doc_type == 'pan':
        fields_model = PANFields(**field_values)
    elif doc_type == 'cv':
        fields_model = CVFields(**field_values)
    else:
        fields_model = field_values  # type: ignore

    # 3. Validation Summary
    all_fields_valid = all(v.is_valid for v in validation.values()) if validation else False
    
    per_field_val = {}
    checksum_passed = None
    for k, v in validation.items():
        per_field_val[k] = {"is_valid": v.is_valid, "error_message": v.error_message}
        if k == "uid" or k == "pan_number":
            # For primary IDs, track checksum specifically
            checksum_passed = v.is_valid

    val_summary = ValidationSummary(
        all_fields_valid=all_fields_valid,
        checksum_passed=checksum_passed,
        per_field=per_field_val
    )
    
    # 4. Meta
    meta = ResponseMeta(
        extraction_method=extraction_method,
        parsability_score=overall_confidence,  # For now, tie parsability to confidence
        cache_hit_rate=None,  # Passed dynamically if cached_engine was used
        correction_count=correction_count,
        escalation_count=escalation_count,
        processing_time_ms=processing_time_ms,
        ocr_engine="paddleocr" if extraction_method == 'ocr' else "pymupdf"
    )
    
    return ExtractResponse(
        request_id=request_id,
        doc_type=doc_type,  # type: ignore
        template_version=template_version,
        overall_confidence=overall_confidence,
        fields=fields_model,
        validation=val_summary,
        flags=flags,
        meta=meta
    )
