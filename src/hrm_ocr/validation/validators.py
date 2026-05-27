"""
hrm_ocr.validation.validators
=============================
Strict field validation logic for OCR outputs. Runs after post-correction.
Includes regex, checksums, and reference database lookups.
"""
from __future__ import annotations

import datetime
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Verhoeff algorithm tables for UID validation
d = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
)

p = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8)
)

inv = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


@dataclass
class ValidationResult:
    is_valid: bool
    error_message: str = ""


def _get_db_connection(db_name: str) -> sqlite3.Connection | None:
    """Helper to safely connect to reference SQLite databases."""
    repo_root = Path(__file__).resolve().parents[3]
    db_path = repo_root / "data" / "reference" / db_name
    
    if not db_path.exists():
        return None
        
    # Read-only connection using URI
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        return conn
    except sqlite3.Error as e:
        logger.warning("Database connection failed for %s: %s", db_name, e)
        return None


def validate_uid(uid: str) -> ValidationResult:
    """Validate Aadhaar UID using Verhoeff checksum."""
    clean_uid = uid.replace(" ", "").strip()
    
    if not clean_uid.isdigit():
        return ValidationResult(False, "UID contains non-numeric characters")
        
    if len(clean_uid) != 12:
        return ValidationResult(False, f"UID must be exactly 12 digits (found {len(clean_uid)})")
        
    if clean_uid.startswith("0") or clean_uid.startswith("1"):
        return ValidationResult(False, "UID cannot start with 0 or 1")
        
    # Calculate Verhoeff
    c = 0
    num_array = [int(x) for x in reversed(clean_uid)]
    for i, n in enumerate(num_array):
        c = d[c][p[i % 8][n]]
        
    if c == 0:
        return ValidationResult(True)
    return ValidationResult(False, "Verhoeff checksum failed")


def validate_pan(pan: str) -> ValidationResult:
    """Validate PAN format and entity code."""
    clean_pan = pan.strip().upper()
    
    if not re.match(r"^[A-Z]{3}[ABCFGHLJPTFG][A-Z][0-9]{4}[A-Z]$", clean_pan):
        return ValidationResult(False, "PAN format is invalid or contains an invalid entity code")
        
    return ValidationResult(True)


def validate_ifsc(ifsc: str) -> ValidationResult:
    """Validate IFSC against regex and local SQLite database."""
    clean_ifsc = ifsc.strip().upper()
    
    if not re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", clean_ifsc):
        return ValidationResult(False, "IFSC format is invalid")
        
    conn = _get_db_connection("ifsc.db")
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM ifsc_codes WHERE code = ?", (clean_ifsc,))
            if not cursor.fetchone():
                return ValidationResult(False, "IFSC not found in master database")
        finally:
            conn.close()
    else:
        logger.warning("ifsc.db not found. Skipping DB lookup.")
        
    return ValidationResult(True)


def validate_pincode(pin: str) -> ValidationResult:
    """Validate Pincode length and local SQLite database."""
    clean_pin = pin.strip()
    
    if not re.match(r"^[1-9][0-9]{5}$", clean_pin):
        return ValidationResult(False, "Pincode format is invalid (must be 6 digits not starting with 0)")
        
    conn = _get_db_connection("pincodes.db")
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM pincodes WHERE code = ?", (clean_pin,))
            if not cursor.fetchone():
                return ValidationResult(False, "Pincode not found in master database")
        finally:
            conn.close()
    else:
        logger.warning("pincodes.db not found. Skipping DB lookup.")
        
    return ValidationResult(True)


def validate_date(date_str: str) -> ValidationResult:
    """Validate calendar date and logical bounds."""
    clean_date = date_str.strip()
    
    # Check for YOB format first
    if re.match(r"^(19\d{2}|20\d{2})$", clean_date):
        age = datetime.date.today().year - int(clean_date)
        if 0 <= age <= 120:
             return ValidationResult(True)
             
    match = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", clean_date)
    if not match:
        return ValidationResult(False, "Date must be DD/MM/YYYY or YYYY")
        
    dd, mm, yyyy = map(int, match.groups())
    
    try:
        date_obj = datetime.date(yyyy, mm, dd)
    except ValueError:
        return ValidationResult(False, "Invalid calendar date")
        
    today = datetime.date.today()
    
    if yyyy < 1900:
        return ValidationResult(False, "Year before 1900")
        
    if date_obj > today:
        return ValidationResult(False, "Date is in the future")
        
    age = today.year - yyyy - ((today.month, today.day) < (mm, dd))
    if age > 120:
        return ValidationResult(False, "Age exceeds 120 years")
        
    return ValidationResult(True)


def validate_email(email: str) -> ValidationResult:
    """Validate email using standard RFC 5322 regex approximation."""
    clean_email = email.strip()
    # Basic comprehensive regex
    pattern = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    if re.match(pattern, clean_email):
        return ValidationResult(True)
    return ValidationResult(False, "Invalid email format")


def validate_phone(phone: str) -> ValidationResult:
    """Validate Indian or international phone numbers."""
    clean_phone = phone.strip()
    # Indian mobile: 10 digits starting with 6-9
    # International: + followed by 10-15 digits
    if re.match(r"^[6-9]\d{9}$", clean_phone) or re.match(r"^\+?[1-9]\d{9,14}$", clean_phone):
        return ValidationResult(True)
    return ValidationResult(False, "Invalid phone number format")


def validate_fields(doc_type: str, fields: dict[str, str]) -> dict[str, ValidationResult]:
    """Run all applicable validators against the extracted field map."""
    results: dict[str, ValidationResult] = {}
    
    for key, value in fields.items():
        k = key.lower()
        if not value:
            # Skip empty fields, they are handled separately as 'missing' upstream
            results[key] = ValidationResult(False, "Field is empty")
            continue
            
        if "uid" in k or "aadhaar" in k:
            results[key] = validate_uid(value)
        elif "pan_number" in k or "pan" == k:
            results[key] = validate_pan(value)
        elif "ifsc" in k:
            results[key] = validate_ifsc(value)
        elif "pin" in k:
            results[key] = validate_pincode(value)
        elif "dob" in k or "date" in k:
            results[key] = validate_date(value)
        elif "email" in k:
            results[key] = validate_email(value)
        elif "phone" in k or "mobile" in k:
            results[key] = validate_phone(value)
        else:
            # Fields with no strict validator (like name, address text) pass implicitly
            results[key] = ValidationResult(True)
            
    return results
