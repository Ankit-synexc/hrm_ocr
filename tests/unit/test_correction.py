"""
tests/unit/test_correction.py
==============================
Unit tests for the rule-based post-correction layer.
"""
from __future__ import annotations

import pytest

from hrm_ocr.correction.patterns import (
    correct_all_fields,
    correct_date,
    correct_name,
    correct_pan,
    correct_pincode,
    correct_uid,
)
from hrm_ocr.correction.substitutions import DIGIT_SUBS


class TestUIDCorrection:
    def test_correct_uid_with_ocr_noise(self):
        # 8O24 -> 8024, 56l2 -> 5612
        assert correct_uid("8O24 56l2 3847") == "8024 5612 3847"

    def test_correct_uid_removes_junk(self):
        assert correct_uid("1234-5678-9012") == "1234 5678 9012"
        assert correct_uid("1234_5678_9012") == "1234 5678 9012"
        
    def test_correct_uid_leaves_invalid_alone(self):
        # Too short
        assert correct_uid("1234 5678") == "1234 5678"


class TestPANCorrection:
    def test_correct_pan_unchanged(self):
        assert correct_pan("ABCDE1234F") == "ABCDE1234F"

    def test_correct_pan_fixes_digits(self):
        # O in digit position -> 0
        assert correct_pan("ABCDE123OF") == "ABCDE1230F"
        # I in digit position -> 1
        assert correct_pan("ABCDEI234F") == "ABCDE1234F"

    def test_correct_pan_fixes_letters(self):
        # 0 in alpha position -> O
        assert correct_pan("ABCD01234F") == "ABCDO1234F"
        # 5 in alpha position -> S
        assert correct_pan("ABCDE12345") == "ABCDE1234S"
        
    def test_correct_pan_ignores_invalid_length(self):
        assert correct_pan("ABCDE1234") == "ABCDE1234"


class TestDateCorrection:
    def test_correct_date_with_ocr_noise(self):
        # O1/O6/1992 -> 01/06/1992
        assert correct_date("O1/O6/1992") == "01/06/1992"
        
    def test_correct_date_normalises_slashes(self):
        assert correct_date("01-06-1992") == "01/06/1992"
        assert correct_date("01.06.1992") == "01/06/1992"
        
    def test_correct_date_invalid_date_unchanged(self):
        # 32nd day is invalid
        assert correct_date("32/01/1992") == "32/01/1992"


class TestNameCorrection:
    def test_correct_name_formatting(self):
        assert correct_name("  RAHUL  SHARMA  ") == "Rahul Sharma"
        assert correct_name("J0HN D0E") == "Jhn De"  # Digits are stripped


class TestPincodeCorrection:
    def test_correct_pincode_with_noise(self):
        assert correct_pincode("11O0OI") == "110001"
        
    def test_correct_pincode_invalid_length(self):
        assert correct_pincode("11000") == "11000"


class TestAllDigitSubs:
    def test_all_digit_subs(self):
        """Ensure every single digit substitution works via UID correction."""
        for error_char, correct_digit in DIGIT_SUBS.items():
            # Build a string like '0000 0000 000X'
            raw = f"0000 0000 000{error_char}"
            expected = f"0000 0000 000{correct_digit}"
            assert correct_uid(raw) == expected


class TestCorrectionDispatcher:
    def test_correct_all_fields(self):
        raw_fields = {
            "uid": "8O24 56l2 3847",
            "pan_number": "ABCDE123OF",
            "dob": "O1/O6/1992",
            "name": "  RAHUL  SHARMA  ",
            "unrelated": " NOISE "
        }
        
        results = correct_all_fields("unknown_doc", raw_fields)
        
        assert results["uid"].corrected == "8024 5612 3847"
        assert results["uid"].was_changed is True
        
        assert results["pan_number"].corrected == "ABCDE1230F"
        assert results["pan_number"].was_changed is True
        
        assert results["dob"].corrected == "01/06/1992"
        assert results["dob"].was_changed is True
        
        assert results["name"].corrected == "Rahul Sharma"
        assert results["name"].was_changed is True
        
        assert results["unrelated"].corrected == "NOISE"
        assert results["unrelated"].was_changed is True
