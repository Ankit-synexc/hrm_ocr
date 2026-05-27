"""
hrm_ocr.correction.substitutions
================================
Deterministic character substitution tables for OCR post-correction.
"""
from __future__ import annotations

# Characters that look like digits when OCR makes mistakes.
DIGIT_SUBS: dict[str, str] = {
    'O': '0', 'o': '0', 'Q': '0', 'D': '0',
    'I': '1', 'l': '1', 'i': '1', '|': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'G': '6', 'b': '6',
    'T': '7',
    'B': '8',
    'g': '9', 'q': '9',
}

# Characters that look like letters when OCR makes mistakes.
ALPHA_SUBS: dict[str, str] = {
    '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G',
}
