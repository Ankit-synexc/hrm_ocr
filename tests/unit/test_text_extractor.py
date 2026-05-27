"""
Unit tests for hrm_ocr.pipeline.text_extractor
"""
from __future__ import annotations

from hrm_ocr.pipeline.text_extractor import extract_fields_from_text


class TestTextExtractor:
    def test_cv_extraction(self):
        cv_text = """
John Doe
Software Engineer
john.doe@example.com
+91 9876543210
linkedin.com/in/johndoe

Skills
Python, Machine Learning, Computer Vision

Experience
Software Developer
ABC Corp
Jan 2020 - Present

Education
B.Tech in Computer Science
XYZ University
        """
        result = extract_fields_from_text(cv_text, "cv")
        assert result.fields["name"] == "John Doe"
        assert result.fields["email"] == "john.doe@example.com"
        assert result.fields["phone"] == "+91 9876543210"
        assert "linkedin.com/in/johndoe" in result.fields["linkedin"]
        assert result.fields["skills"] == "Python, Machine Learning, Computer Vision"
        assert result.fields["education"] == "B.Tech in Computer Science"
        # "Jan 2020 - Present" matches the regex
        assert result.fields["total_experience_years"] == "1"
        assert len(result.warnings) == 1
        assert result.warnings[0] == "current_title missing"

    def test_cv_missing_phone(self):
        cv_text = "Jane Doe\njane@example.com\n"
        result = extract_fields_from_text(cv_text, "cv")
        assert result.fields["phone"] == ""
        assert "phone missing" in result.warnings

    def test_aadhaar_extraction(self):
        aadhaar_text = """
Name: Ravi Kumar
DOB: 15/08/1985
Gender: Male
1234 5678 9012
        """
        result = extract_fields_from_text(aadhaar_text, "aadhaar")
        assert result.fields["name"] == "Ravi Kumar"
        assert result.fields["dob"] == "15/08/1985"
        assert result.fields["gender"] == "Male"
        assert result.fields["uid"] == "1234 5678 9012"
        assert not result.warnings

    def test_pan_extraction(self):
        pan_text = """
INCOME TAX DEPARTMENT
Name
Ankit Sharma
Father's Name
Rajesh Sharma
Date of Birth
10/05/1992
Permanent Account Number Card
ABCDE1234F
        """
        result = extract_fields_from_text(pan_text, "pan")
        assert result.fields["pan_number"] == "ABCDE1234F"
        assert result.fields["name"] == "Ankit Sharma"
        assert result.fields["father_name"] == "Rajesh Sharma"
        assert result.fields["dob"] == "10/05/1992"
        assert not result.warnings
