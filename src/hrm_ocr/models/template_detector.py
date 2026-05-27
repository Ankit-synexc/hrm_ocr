"""
hrm_ocr.models.template_detector
================================
Rule-based template detector using pure OpenCV logic.
Runs in <5ms. No ML inference overhead.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TemplateDetectionResult:
    doc_type: Literal['aadhaar', 'pan', 'unknown']
    template_version: str
    confidence: float
    field_coordinate_map: dict
    detection_method: str


def _load_field_coords(template_version: str) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    coords_path = repo_root / "configs" / "field_coords.yaml"
    if not coords_path.exists():
        logger.warning(f"field_coords.yaml not found at {coords_path}")
        return {}
        
    with open(coords_path, "r", encoding="utf-8") as f:
        coords = yaml.safe_load(f)
    return coords.get(template_version, {})


def detect_template(image: np.ndarray, extracted_text: str = "") -> TemplateDetectionResult:
    """Detect the specific ID template version from visual rules (and optional text).
    
    Rules applied in cascade:
    1. Aspect ratio
    2. Color histogram (HSV)
    3. QR code presence and location
    4. Text regex matching (if extracted_text is provided)
    """
    h, w = image.shape[:2]
    aspect_ratio = w / max(h, 1)
    
    # Pre-load all coords for fallback
    repo_root = Path(__file__).resolve().parents[3]
    coords_path = repo_root / "configs" / "field_coords.yaml"
    all_coords = {}
    if coords_path.exists():
        with open(coords_path, "r", encoding="utf-8") as f:
            all_coords = yaml.safe_load(f)

    # ---------------------------------------------------------
    # RULE 4: Text-based explicit override (if text is provided)
    # ---------------------------------------------------------
    import re
    if extracted_text:
        if re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", extracted_text):
            return TemplateDetectionResult(
                doc_type="pan",
                template_version="pan_v2", # Default to newest PAN layout for spatial extraction
                confidence=0.99,
                field_coordinate_map=all_coords.get("pan_v2", {}),
                detection_method="text_regex_pan"
            )
        elif re.search(r"\d{4}\s?\d{4}\s?\d{4}", extracted_text):
            return TemplateDetectionResult(
                doc_type="aadhaar",
                template_version="aadhaar_v3", # Default to standard Aadhaar layout for spatial extraction
                confidence=0.99,
                field_coordinate_map=all_coords.get("aadhaar_v3", {}),
                detection_method="text_regex_aadhaar"
            )

    # ---------------------------------------------------------
    # RULE 1: Aspect Ratio
    # ---------------------------------------------------------
    # mAadhaar portrait mode
    if abs(aspect_ratio - 0.63) < 0.15:
        return TemplateDetectionResult(
            doc_type="aadhaar",
            template_version="aadhaar_v4",
            confidence=0.9,
            field_coordinate_map=all_coords.get("aadhaar_v4", {}),
            detection_method="aspect_ratio_portrait"
        )
        
    # If it's not landscape, we don't know what it is
    if not (1.4 < aspect_ratio < 1.7):
        return TemplateDetectionResult(
            doc_type="unknown",
            template_version="unknown",
            confidence=0.0,
            field_coordinate_map={},
            detection_method="aspect_ratio_out_of_bounds"
        )

    # ---------------------------------------------------------
    # RULE 2: HSV Color Analysis
    # ---------------------------------------------------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # High blue saturation (Aadhaar v1, v2)
    # H: 100-130 (OpenCV H is 0-179, so 100-130 is 200-260 degrees, which is blue)
    blue_mask = cv2.inRange(hsv, np.array([90, 100, 50]), np.array([130, 255, 255]))
    blue_ratio = np.sum(blue_mask > 0) / (h * w)
    
    # Cream/off-white (PAN v1)
    cream_mask = cv2.inRange(hsv, np.array([10, 20, 150]), np.array([40, 60, 255]))
    cream_ratio = np.sum(cream_mask > 0) / (h * w)
    
    # Low saturation / White (Aadhaar v3, PAN v2, PAN v3)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 35, 255]))
    white_ratio = np.sum(white_mask > 0) / (h * w)
    
    # ---------------------------------------------------------
    # RULE 3: QR Code
    # ---------------------------------------------------------
    qr_detector = cv2.QRCodeDetector()
    has_qr = False
    qr_bbox = None
    
    # Optimize QR detection by only checking the bottom-right and right edge where QRs usually are
    qr_roi = image[int(h*0.3):, int(w*0.5):]
    try:
        retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(qr_roi)
        if retval:
            has_qr = True
            qr_bbox = points
    except Exception:
        pass

    # ---------------------------------------------------------
    # CASCADE LOGIC
    # ---------------------------------------------------------
    
    logger.info(f"Template detector ratios -> Blue: {blue_ratio:.3f}, White: {white_ratio:.3f}, Cream: {cream_ratio:.3f}, QR: {has_qr}")
    
    # Blue cards (Aadhaar v1/v2)
    if blue_ratio > 0.05:  # Highly relaxed threshold for real-world lighting
        if has_qr:
            return TemplateDetectionResult(
                doc_type="aadhaar",
                template_version="aadhaar_v2",
                confidence=0.8,
                field_coordinate_map=all_coords.get("aadhaar_v2", {}),
                detection_method="hsv_blue_with_qr"
            )
        else:
            return TemplateDetectionResult(
                doc_type="aadhaar",
                template_version="aadhaar_v1",
                confidence=0.7,
                field_coordinate_map=all_coords.get("aadhaar_v1", {}),
                detection_method="hsv_blue_no_qr"
            )
            
    # Cream cards (PAN v1)
    if cream_ratio > 0.1:
        return TemplateDetectionResult(
            doc_type="pan",
            template_version="pan_v1",
            confidence=0.7,
            field_coordinate_map=all_coords.get("pan_v1", {}),
            detection_method="hsv_cream"
        )
        
    # White cards (Aadhaar v3 / PAN v2 / PAN v3)
    if white_ratio > 0.1:
        if has_qr:
            return TemplateDetectionResult(
                doc_type="aadhaar",
                template_version="aadhaar_v3",
                confidence=0.8,
                field_coordinate_map=all_coords.get("aadhaar_v3", {}),
                detection_method="hsv_white_with_qr"
            )
        else:
            return TemplateDetectionResult(
                doc_type="pan",
                template_version="pan_v2",
                confidence=0.7,
                field_coordinate_map=all_coords.get("pan_v2", {}),
                detection_method="hsv_white_no_qr"
            )

    # ---------------------------------------------------------
    # DEFINITIVE FALLBACK (If lighting completely ruins HSV)
    # ---------------------------------------------------------
    if extracted_text:
        if re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", extracted_text):
            return TemplateDetectionResult(
                doc_type="pan", template_version="pan_v3", confidence=0.5,
                field_coordinate_map=all_coords.get("pan_v3", {}), detection_method="text_pan_fallback"
            )
        elif re.search(r"\d{4}\s\d{4}\s\d{4}", extracted_text):
            return TemplateDetectionResult(
                doc_type="aadhaar", template_version="aadhaar_v3", confidence=0.5,
                field_coordinate_map=all_coords.get("aadhaar_v3", {}), detection_method="text_aadhaar_fallback"
            )
            
    # Absolute fallback guess (assume modern Aadhaar if completely unknown)
    return TemplateDetectionResult(
        doc_type="aadhaar",
        template_version="aadhaar_v3",
        confidence=0.2,
        field_coordinate_map=all_coords.get("aadhaar_v3", {}),
        detection_method="absolute_fallback_guess"
    )
