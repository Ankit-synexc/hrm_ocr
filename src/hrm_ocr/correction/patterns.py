"""
hrm_ocr.correction.patterns
===========================
Field-specific post-correction rules based on domain constraints.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from hrm_ocr.correction.substitutions import ALPHA_SUBS, DIGIT_SUBS


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    was_changed: bool
    change_description: str


def correct_uid(raw: str) -> str:
    """Correct Aadhaar UID formatting and OCR errors.
    
    Rules:
      - Strip non-alphanumerics
      - Apply DIGIT_SUBS
      - Format as XXXX XXXX XXXX if 12 digits
    """
    stripped = re.sub(r"[^\w]", "", raw)
    corrected_chars = [DIGIT_SUBS.get(c, c) for c in stripped]
    candidate = "".join(corrected_chars)
    
    if len(candidate) == 12 and candidate.isdigit():
        return f"{candidate[:4]} {candidate[4:8]} {candidate[8:12]}"
    return raw


def correct_pan(raw: str) -> str:
    """Correct PAN number OCR errors based on positional constraints.
    
    Rules:
      - Strip spaces, uppercase
      - Pos 0-4: Alpha
      - Pos 5-8: Digits
      - Pos 9: Alpha
    """
    cleaned = raw.replace(" ", "").upper()
    if len(cleaned) != 10:
        return raw
        
    corrected = []
    for i, c in enumerate(cleaned):
        if i < 5:  # Alpha
            corrected.append(ALPHA_SUBS.get(c, c))
        elif i < 9:  # Digit
            corrected.append(DIGIT_SUBS.get(c, c))
        else:  # Alpha
            corrected.append(ALPHA_SUBS.get(c, c))
            
    candidate = "".join(corrected)
    if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", candidate):
        return candidate
    return raw


def correct_date(raw: str) -> str:
    """Correct dates (DOB) and validate.
    
    Rules:
      - Apply DIGIT_SUBS
      - Reconstruct DD/MM/YYYY
    """
    # Replace common slash mistakes
    s = raw.replace("-", "/").replace(".", "/").replace("\\", "/")
    
    corrected = []
    for c in s:
        if c == "/":
            corrected.append(c)
        else:
            corrected.append(DIGIT_SUBS.get(c, c))
    
    candidate = "".join(corrected)
    
    match = re.search(r"(\d{2})/(\d{2})/(\d{4})", candidate)
    if match:
        dd, mm, yyyy = match.groups()
        
        # Basic validation
        d, m, y = int(dd), int(mm), int(yyyy)
        if 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100:
            return f"{dd}/{mm}/{yyyy}"
            
    return raw


def correct_name(raw: str) -> str:
    """Clean and title-case names."""
    # Remove non-alphabetic/non-space
    s = re.sub(r"[^a-zA-Z\s]", "", raw)
    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def correct_pincode(raw: str) -> str:
    """Correct PIN code (6 digits)."""
    stripped = re.sub(r"[^\w]", "", raw)
    corrected_chars = [DIGIT_SUBS.get(c, c) for c in stripped]
    candidate = "".join(corrected_chars)
    
    if len(candidate) == 6 and candidate.isdigit():
        return candidate
    return raw


def correct_field(field_name: str, raw: str) -> CorrectionResult:
    """Dispatcher to route a field to its corresponding correction function."""
    original = raw
    corrected = raw
    
    # Simple routing based on field name
    fn_lower = field_name.lower()
    if "uid" in fn_lower or "aadhaar" in fn_lower:
        corrected = correct_uid(raw)
    elif "pan" in fn_lower:
        corrected = correct_pan(raw)
    elif "dob" in fn_lower or "date" in fn_lower:
        corrected = correct_date(raw)
    elif "name" in fn_lower:
        corrected = correct_name(raw)
    elif "pin" in fn_lower:
        corrected = correct_pincode(raw)
        
    # By default, trim whitespace if no specific rule matched
    if corrected == raw and field_name not in ["uid", "pan", "dob", "name", "pincode"]:
        corrected = raw.strip()
        
    was_changed = original != corrected
    desc = f"{field_name} correction applied" if was_changed else ""
    
    return CorrectionResult(
        original=original,
        corrected=corrected,
        was_changed=was_changed,
        change_description=desc
    )


def correct_all_fields(doc_type: str, fields: dict[str, str]) -> dict[str, CorrectionResult]:
    """Run correction on every field and return detailed results."""
    results = {}
    for k, v in fields.items():
        results[k] = correct_field(k, v)
    return results
