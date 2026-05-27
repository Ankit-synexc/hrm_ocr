"""
hrm_ocr.data.pan_gen
====================
Synthetic PAN card generator for training and evaluation.

Generates 3 template versions:
- v1 (pre-2017 NSDL): cream background, standard tan card
- v2 (2017-2021 UTI): redesigned layout with right-side hologram
- v3 (2022-present): current Income Tax branding, e-PAN compatible

Coordinates match `configs/field_coords.yaml`.
"""
from __future__ import annotations

import json
import logging
import random
import string
from pathlib import Path
from typing import Any

import yaml
from faker import Faker
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Canonical size
CARD_W = 1012
CARD_H = 638


def generate_valid_pan(entity_type: str = "P", surname: str = "") -> str:
    """Generate a structurally valid PAN number.
    
    Structure:
    - Chars 1-3: Random alpha (AAA to ZZZ)
    - Char 4: Entity code (P=Person, C=Company, etc.)
    - Char 5: First letter of surname (or random if not provided)
    - Chars 6-9: Sequential/Random digits (0001 to 9999)
    - Char 10: Check character (alphabetic)
    """
    c13 = "".join(random.choices(string.ascii_uppercase, k=3))
    c4 = entity_type.upper()
    c5 = surname[0].upper() if surname else random.choice(string.ascii_uppercase)
    c69 = f"{random.randint(1, 9999):04d}"
    c10 = random.choice(string.ascii_uppercase)  # True check-char logic is complex; random alpha suffices for OCR
    return f"{c13}{c4}{c5}{c69}{c10}"


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a true type font, or fallback to default if not found."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        return ImageFont.load_default()


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: list[int],
    font: Any,
    fill: str = "black",
    embossed: bool = False
) -> None:
    """Draw text inside a bounding box (x_min, y_min, x_max, y_max)."""
    x_min, y_min, _, _ = box
    
    if embossed:
        # Draw dark shadow offset 2px down and right
        draw.text((x_min + 2, y_min + 2), text, font=font, fill="#555555")
        # Draw highlight offset 1px up and left
        draw.text((x_min - 1, y_min - 1), text, font=font, fill="#FFFFFF")
        
    draw.text((x_min, y_min), text, font=font, fill=fill)


def generate_pan(template_version: str, n: int, output_dir: Path) -> list[dict[str, Any]]:
    """Generate n synthetic PAN cards of the specified version.
    
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
    
    font_large = get_font(40)
    font_medium = get_font(30)
    
    records = []
    
    for i in range(n):
        # 1. Base background
        bg_color = "#FFF5E1"  # Cream/tan background typical for PAN
        img = Image.new("RGB", (CARD_W, CARD_H), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Header strip (blue)
        header_box = coords.get("header_strip")
        if header_box:
            draw.rectangle(header_box, fill="#003366")
            
        # Hologram strip (right side) if v2 or v3
        hologram_box = coords.get("hologram_strip")
        if hologram_box:
            draw.rectangle(hologram_box, fill="#D3D3D3")
            
        # 2. Generate Fake Data
        surname = fake.last_name()
        first_name = fake.first_name_male()
        name = f"{first_name} {surname}"
        
        father_name = f"{fake.first_name_male()} {surname}"
        dob_str = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%d/%m/%Y")
        pan_num = generate_valid_pan(entity_type="P", surname=surname)
        
        # 3. Draw fields
        # Name
        name_box = coords.get("name")
        if name_box:
            _draw_text_in_box(draw, name.upper(), name_box, font_medium)
            
        # Father's Name
        father_box = coords.get("fathers_name")
        if father_box:
            _draw_text_in_box(draw, father_name.upper(), father_box, font_medium)
            
        # DOB
        dob_box = coords.get("dob")
        if dob_box:
            _draw_text_in_box(draw, dob_str, dob_box, font_medium)
            
        # PAN Number (Embossed)
        pan_box = coords.get("pan_number")
        if pan_box:
            _draw_text_in_box(draw, pan_num, pan_box, font_large, fill="black", embossed=True)
            
        # Visual placeholders
        photo_box = coords.get("photo_box")
        if photo_box:
            draw.rectangle(photo_box, fill="#CCCCCC", outline="#000000", width=2)
            
        sig_box = coords.get("signature_box")
        if sig_box:
            draw.rectangle(sig_box, fill="#EEEEEE", outline="#000000", width=1)
            
        # 4. Save image
        img_filename = f"{template_version}_{i:04d}.jpg"
        img_path = images_dir / img_filename
        img.save(img_path, "JPEG", quality=95)
        
        # 5. Add to records
        records.append({
            "image_path": f"images/{img_filename}",
            "template_version": template_version,
            "fields": {
                "name": name,
                "father_name": father_name,
                "dob": dob_str,
                "pan_number": pan_num,
                "entity_type": "P"
            }
        })
        
    # Write annotations
    anno_path = output_dir / f"gt_{template_version}.jsonl"
    with open(anno_path, "a" if anno_path.exists() else "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    logger.info("Generated %d %s cards in %s", n, template_version, output_dir)
    return records
