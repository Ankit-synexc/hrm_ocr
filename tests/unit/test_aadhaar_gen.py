"""
tests/unit/test_aadhaar_gen.py
==============================
Unit tests for the synthetic Aadhaar generator.
"""
from __future__ import annotations

from pathlib import Path

from hrm_ocr.data.aadhaar_gen import calc_checksum, generate_valid_uid, generate_aadhaar


class TestAadhaarGen:
    def test_verhoeff_checksum(self):
        # Known valid Verhoeff string: '236' -> checksum is '3' to make '2363' valid
        # Actually '12345678901' -> let's test generate_valid_uid directly
        uid = generate_valid_uid().replace(" ", "")
        assert len(uid) == 12
        base_num = uid[:11]
        expected_checksum = calc_checksum(base_num)
        assert uid[11] == expected_checksum
        
        # Test 10 random UIDs
        for _ in range(10):
            u = generate_valid_uid().replace(" ", "")
            assert u[11] == calc_checksum(u[:11])

    def test_uid_formatting(self):
        uid = generate_valid_uid()
        assert len(uid) == 14
        assert uid[4] == " "
        assert uid[9] == " "

    def test_generate_aadhaar(self, tmp_path: Path):
        records = generate_aadhaar("aadhaar_v3", 2, tmp_path)
        assert len(records) == 2
        
        # Check files were created
        images_dir = tmp_path / "images"
        assert images_dir.exists()
        images = list(images_dir.glob("*.jpg"))
        assert len(images) == 2
        
        gt_file = tmp_path / "gt_aadhaar_v3.jsonl"
        assert gt_file.exists()
        
        # Check record structure
        rec = records[0]
        assert "image_path" in rec
        assert rec["template_version"] == "aadhaar_v3"
        fields = rec["fields"]
        assert "name" in fields
        assert "uid" in fields
        assert "address" in fields
        
        # Ensure image dimensions are correct (1012x638)
        from PIL import Image
        img = Image.open(images[0])
        assert img.size == (1012, 638)
