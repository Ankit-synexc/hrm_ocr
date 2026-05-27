"""
tests/unit/test_pan_gen.py
==========================
Unit tests for the synthetic PAN generator.
"""
from __future__ import annotations

import re
from pathlib import Path

from hrm_ocr.data.pan_gen import generate_valid_pan, generate_pan


class TestPANGen:
    def test_pan_regex_validity(self):
        # 10 random PANs
        for _ in range(10):
            pan = generate_valid_pan(entity_type="P", surname="Sharma")
            assert len(pan) == 10
            # RegEx: 5 letters, 4 digits, 1 letter
            assert re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", pan)
            # Entity type is P
            assert pan[3] == "P"
            # Surname letter is S
            assert pan[4] == "S"

    def test_generate_pan(self, tmp_path: Path):
        records = generate_pan("pan_v3", 2, tmp_path)
        assert len(records) == 2
        
        # Check files were created
        images_dir = tmp_path / "images"
        assert images_dir.exists()
        images = list(images_dir.glob("*.jpg"))
        assert len(images) == 2
        
        gt_file = tmp_path / "gt_pan_v3.jsonl"
        assert gt_file.exists()
        
        # Check record structure
        rec = records[0]
        assert "image_path" in rec
        assert rec["template_version"] == "pan_v3"
        fields = rec["fields"]
        assert "name" in fields
        assert "father_name" in fields
        assert "dob" in fields
        assert "pan_number" in fields
        assert "entity_type" in fields
        assert fields["entity_type"] == "P"
        
        # Ensure image dimensions are correct (1012x638)
        from PIL import Image
        img = Image.open(images[0])
        assert img.size == (1012, 638)
