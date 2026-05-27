"""
hrm_ocr.data.aadhaar_gen
========================
Synthetic Aadhaar card generator for training and evaluation.

Generates 4 template versions:
- v1 (pre-2012): blue background, no QR
- v2 (2012-2017): redesigned blue-white, small QR
- v3 (2018+): saffron/white background, large QR
- v4 (mAadhaar): secure print

Coordinates match `configs/field_coords.yaml`.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import qrcode
import yaml
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Canonical size
CARD_W = 1012
CARD_H = 638

# Verhoeff multiplication table
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

# Verhoeff permutation table
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

# Verhoeff inverse table
inv = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def calc_checksum(num: str) -> str:
    """Calculate Verhoeff checksum digit."""
    c = 0
    num_array = [int(x) for x in reversed(num)]
    for i, n in enumerate(num_array):
        c = d[c][p[(i + 1) % 8][n]]
    return str(inv[c])


def generate_valid_uid() -> str:
    """Generate a valid 12-digit Aadhaar UID matching the Verhoeff algorithm.
    
    Aadhaar UIDs don't start with 0 or 1.
    """
    # 11 random digits, first digit 2-9
    first = str(random.randint(2, 9))
    rest = "".join(str(random.randint(0, 9)) for _ in range(10))
    base_num = first + rest
    checksum = calc_checksum(base_num)
    uid = base_num + checksum
    # Format as XXXX XXXX XXXX
    return f"{uid[:4]} {uid[4:8]} {uid[8:12]}"


def get_font(size: int, is_hindi: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a true type font, or fallback to default if not found."""
    # Try common paths or fallback
    try:
        font_name = "Mangal.ttf" if is_hindi else "arial.ttf"
        return ImageFont.truetype(font_name, size)
    except IOError:
        return ImageFont.load_default()


def _draw_text_in_box(draw: ImageDraw.ImageDraw, text: str, box: list[int], font: Any, fill: str = "black") -> None:
    """Draw text inside a bounding box (x_min, y_min, x_max, y_max)."""
    # Just draw at the top-left of the box for synthetic data purposes
    x_min, y_min, _, _ = box
    draw.text((x_min, y_min), text, font=font, fill=fill)


def generate_aadhaar(template_version: str, n: int, output_dir: Path) -> list[dict[str, Any]]:
    """Generate n synthetic Aadhaar cards of the specified version.
    
    Saves images to output_dir/images and JSONL annotations to output_dir/annotations.jsonl.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Load field coordinates
    repo_root = Path(__file__).resolve().parents[3]
    coords_path = repo_root / "configs" / "field_coords.yaml"
    with open(coords_path, "r", encoding="utf-8") as f:
        all_coords = yaml.safe_load(f)
        
    coords = all_coords.get(template_version)
    if not coords:
        raise ValueError(f"Unknown template version: {template_version}")
        
    fake = Faker("en_IN")
    fake_hi = Faker("hi_IN")
    
    font_large = get_font(40)
    font_medium = get_font(30)
    font_small = get_font(20)
    font_hi_large = get_font(35, is_hindi=True)
    
    records = []
    
    for i in range(n):
        # 1. Base background
        if template_version in ["aadhaar_v1", "aadhaar_v2"]:
            bg_color = "#E6F3FF"  # Light blue
        else:
            bg_color = "#FFFFFF"  # White
            
        img = Image.new("RGB", (CARD_W, CARD_H), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Saffron stripe for v3/v4
        if "saffron_stripe" in coords:
            x1, y1, x2, y2 = coords["saffron_stripe"]
            draw.rectangle([x1, y1, x2, y2], fill="#FF9933")
            
        # 2. Generate Fake Data
        gender_en = random.choice(["Male", "Female", "Transgender"])
        gender_hi = {"Male": "पुरुष", "Female": "महिला", "Transgender": "ट्रांसजेंडर"}[gender_en]
        
        # Gender string
        gender_str = f"Gender: {gender_en}"
        
        # Name
        if gender_en == "Male":
            name_en = fake.name_male()
            name_hi = fake_hi.name_male()
        else:
            name_en = fake.name_female()
            name_hi = fake_hi.name_female()
            
        dob_str = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d/%m/%Y")
        uid_str = generate_valid_uid()
        
        address_line1 = fake.street_address()
        address_line2 = fake.street_name()
        city = fake.city()
        state = fake.state()
        pincode = fake.postcode()
        
        # 3. Draw fields
        # Name (with Hindi above English in the same bounding box or nearby)
        name_box = coords.get("name")
        if name_box:
            _draw_text_in_box(draw, name_hi, name_box, font_hi_large)
            # Offset English name slightly down
            eng_box = [name_box[0], name_box[1] + 35, name_box[2], name_box[3]]
            _draw_text_in_box(draw, name_en, eng_box, font_large)
            
        # DOB
        dob_box = coords.get("dob")
        if dob_box:
            _draw_text_in_box(draw, f"DOB: {dob_str}", dob_box, font_medium)
            
        # Gender
        gender_box = coords.get("gender")
        if gender_box:
            _draw_text_in_box(draw, gender_str, gender_box, font_medium)
            
        # UID
        uid_box = coords.get("aadhaar_number") or coords.get("masked_aadhaar")
        if uid_box:
            _draw_text_in_box(draw, uid_str, uid_box, font_large, fill="black")
            
        # Address
        if "address_line1" in coords:
            _draw_text_in_box(draw, address_line1, coords["address_line1"], font_small)
        if "address_line2" in coords:
            _draw_text_in_box(draw, address_line2, coords["address_line2"], font_small)
        if "address_city" in coords:
            _draw_text_in_box(draw, city, coords["address_city"], font_small)
        if "address_state" in coords:
            _draw_text_in_box(draw, state, coords["address_state"], font_small)
        if "address_pincode" in coords:
            _draw_text_in_box(draw, pincode, coords["address_pincode"], font_small)
            
        # 4. Draw QR Code for v2, v3, v4
        qr_box = coords.get("qr_code")
        if qr_box and (qr_box[2] - qr_box[0]) > 0:
            qr = qrcode.make(f"UID:{uid_str} Name:{name_en}")
            qr_w = qr_box[2] - qr_box[0]
            qr_h = qr_box[3] - qr_box[1]
            qr_resized = qr.resize((qr_w, qr_h))
            img.paste(qr_resized, (qr_box[0], qr_box[1]))
            
        # 5. Save image
        img_filename = f"{template_version}_{i:04d}.jpg"
        img_path = images_dir / img_filename
        img.save(img_path, "JPEG", quality=95)
        
        # 6. Add to records
        full_address = f"{address_line1}, {address_line2}, {city}, {state} - {pincode}"
        records.append({
            "image_path": f"images/{img_filename}",
            "template_version": template_version,
            "fields": {
                "name": name_en,
                "name_hi": name_hi,
                "dob": dob_str,
                "gender": gender_en,
                "uid": uid_str,
                "address": full_address,
                "pincode": pincode
            }
        })
        
    # Write annotations
    anno_path = output_dir / f"gt_{template_version}.jsonl"
    with open(anno_path, "a" if anno_path.exists() else "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    logger.info("Generated %d %s cards in %s", n, template_version, output_dir)
    return records
