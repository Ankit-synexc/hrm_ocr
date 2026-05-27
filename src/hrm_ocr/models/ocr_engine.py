"""
hrm_ocr.models.ocr_engine
=========================
PaddleOCR engine wrapper.

Handles OCR execution on full cards or field-level crops using lightweight
(<= 8MB) detection and recognition models. Supports English and regional languages.
"""
from __future__ import annotations

import os
# Disable PIR API to fallback to the stable static graph executor in Paddle v3.3+
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"  # Also disable OneDNN just in case
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRRegion:
    bbox: list[list[int]]
    text: str
    confidence: float


@dataclass
class FieldOCRResult:
    text: str
    confidence: float
    raw_regions: list[OCRRegion]


class OCREngine:
    def __init__(
        self,
        model_dir: Path,
        use_gpu: bool = False,
        use_angle_cls: bool = True,
        lang: str = "en"
    ) -> None:
        """Initialize PaddleOCR engine."""
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError("paddleocr package is required") from exc
            
        self.lang = lang
        # Map lang to model subdirectories
        det_dir = model_dir / f"det_{lang}"
        rec_dir = model_dir / f"rec_{lang}"
        cls_dir = model_dir / "cls"
        
        # If models are not explicitly downloaded in our custom path, fallback to default download
        kwargs: dict[str, Any] = {
            "use_angle_cls": use_angle_cls,
            "use_gpu": use_gpu,
            "lang": lang,
            "show_log": False,
        }
        
        if det_dir.exists():
            kwargs["det_model_dir"] = str(det_dir)
        if rec_dir.exists():
            kwargs["rec_model_dir"] = str(rec_dir)
        if cls_dir.exists() and use_angle_cls:
            kwargs["cls_model_dir"] = str(cls_dir)
            
        logger.info("Initializing PaddleOCR (lang=%s, use_gpu=%s)", lang, use_gpu)
        self.ocr = PaddleOCR(**kwargs)

    def recognize_full_card(self, image: np.ndarray) -> list[OCRRegion]:
        """Run OCR on the entire image.
        
        Returns
        -------
        list[OCRRegion]
            Bounding boxes, text strings, and confidences for all detected text.
        """
        # PaddleOCR returns a list of results (one per batch item).
        # For single image, it's a list containing a list of lines.
        # Line format: [ [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('text', confidence) ]
        results = self.ocr.ocr(image, cls=True)
        
        regions = []
        if not results or not results[0]:
            return regions
            
        for line in results[0]:
            if not line:
                continue
            bbox = line[0]
            text = line[1][0]
            conf = float(line[1][1])
            
            # Ensure bbox is a list of lists of ints
            int_bbox = [[int(pt[0]), int(pt[1])] for pt in bbox]
            
            regions.append(OCRRegion(bbox=int_bbox, text=text, confidence=conf))
            
        return regions

    def recognize_field_crop(self, crop: np.ndarray) -> FieldOCRResult:
        """Run OCR on a single pre-cropped field image."""
        regions = self.recognize_full_card(crop)
        
        if not regions:
            return FieldOCRResult(text="", confidence=0.0, raw_regions=[])
            
        # Sort regions: top-to-bottom, then left-to-right
        # Use center Y and center X for sorting
        def _get_center(bbox: list[list[int]]) -> tuple[float, float]:
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            return sum(xs)/len(xs), sum(ys)/len(ys)
            
        # Add centers for sorting
        sorted_regions = sorted(
            regions,
            key=lambda r: (_get_center(r.bbox)[1], _get_center(r.bbox)[0])
        )
        
        text_parts = []
        total_weight = 0.0
        weighted_conf_sum = 0.0
        
        for r in sorted_regions:
            text_parts.append(r.text)
            weight = len(r.text)
            total_weight += weight
            weighted_conf_sum += r.confidence * weight
            
        full_text = " ".join(text_parts).strip()
        final_conf = (weighted_conf_sum / total_weight) if total_weight > 0 else 0.0
        
        return FieldOCRResult(
            text=full_text,
            confidence=final_conf,
            raw_regions=sorted_regions
        )

    def recognize_all_fields(self, image: np.ndarray, field_coordinate_map: dict[str, Any]) -> dict[str, FieldOCRResult]:
        """Crop and recognize each field sequentially with Anchor Registration fallback."""
        
        # 1. Standard execution
        results = self._execute_coordinate_crops(image, field_coordinate_map)
        
        # 2. Anchor Registration Fallback
        # If the primary fields (name, aadhaar_number) are empty, the coordinate map missed.
        # We find the anchor text and shift the map!
        needs_shift = False
        if "aadhaar_number" in results and not results["aadhaar_number"].text.strip():
            needs_shift = True
        elif "name" in results and not results["name"].text.strip():
            needs_shift = True
            
        anchor_text = field_coordinate_map.get("__anchor_text__")
        anchor_target_y = field_coordinate_map.get("__anchor_y__")
        
        if needs_shift and anchor_text and anchor_target_y is not None:
            logger.info("Coordinates missed target. Executing Anchor Registration...")
            full_regions = self.recognize_full_card(image)
            
            # Find the anchor region
            actual_y = None
            for r in full_regions:
                # Fuzzy match for OCR mistakes
                text_up = r.text.upper()
                anchor_up = str(anchor_text).upper()
                if anchor_up in text_up or "GOVERNMENT" in text_up or "INDIA" in text_up:
                    # Calculate center Y of this region
                    actual_y = sum(pt[1] for pt in r.bbox) / len(r.bbox)
                    break
                    
            if actual_y is not None:
                dy = int(actual_y - anchor_target_y)
                logger.info("Found anchor '%s' at Y=%d. Shifting map by dy=%d", anchor_text, actual_y, dy)
                
                # Shift all boxes
                shifted_map = {}
                for k, coords in field_coordinate_map.items():
                    if k.startswith("__") or len(coords) != 4:
                        continue
                    x1, y1, x2, y2 = coords
                    # Only shift Y for now (simplest and most robust for ID cards)
                    shifted_map[k] = [x1, y1 + dy, x2, y2 + dy]
                    
                # Re-run crops with the perfectly snapped map
                results = self._execute_coordinate_crops(image, shifted_map)
            else:
                logger.warning("Anchor Registration failed: anchor text '%s' not found in image.", anchor_text)
                
        return results

    def _execute_coordinate_crops(self, image: np.ndarray, field_map: dict[str, Any]) -> dict[str, FieldOCRResult]:
        """Internal helper to safely crop and extract."""
        results = {}
        h, w = image.shape[:2]
        
        for field_name, coords in field_map.items():
            if field_name.startswith("__") or len(coords) != 4:
                continue
                
            x1, y1, x2, y2 = coords
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            
            if x2 <= x1 or y2 <= y1:
                results[field_name] = FieldOCRResult(text="", confidence=0.0, raw_regions=[])
                continue
                
            crop = image[y1:y2, x1:x2]
            results[field_name] = self.recognize_field_crop(crop)
            
        return results

    def recognize_spatial_from_regions(self, full_regions: list[OCRRegion], doc_type: str) -> dict[str, FieldOCRResult]:
        """ML-Driven Spatial Extraction for wild, uncropped photos."""
        import re
        results = {}
        
        logger.info("Executing ML Spatial Extraction for %s...", doc_type)
        if not full_regions:
            return results
            
        full_text = " ".join([r.text for r in full_regions])
        
        if doc_type == "aadhaar":
            # 1. Aadhaar Number (12 digits)
            uid_match = re.search(r"(\d{4}\s?\d{4}\s?\d{4})", full_text)
            if uid_match:
                # Mock a FieldOCRResult
                results["aadhaar_number"] = FieldOCRResult(
                    text=uid_match.group(1).replace(" ", ""),
                    confidence=0.99,
                    raw_regions=[]
                )
                
            # 2. DOB (highly robust to OCR artifacts like D0B, Year of 8irth)
            dob_match = re.search(r"(?:DOB|D0B|YOB|Birth|Date).*?([0-9]{2}[/-][0-9]{2}[/-][0-9]{4}|[0-9]{4})", full_text, re.IGNORECASE)
            if dob_match:
                results["dob"] = FieldOCRResult(text=dob_match.group(1), confidence=0.95, raw_regions=[])
                
            # 3. Gender
            gender_match = re.search(r"(MALE|FEMALE|TRANSGENDER|M a l e|F e m a l e)", full_text, re.IGNORECASE)
            if gender_match:
                results["gender"] = FieldOCRResult(text=gender_match.group(1).replace(" ", "").capitalize(), confidence=0.95, raw_regions=[])
                
            # 4. Name (Spatial heuristic: Name is usually the line right before DOB)
            # Find the region that contains the DOB
            dob_y = None
            for r in full_regions:
                if "DOB" in r.text.upper() or "YOB" in r.text.upper() or "YEAR" in r.text.upper() or (dob_match and dob_match.group(1) in r.text):
                    dob_y = sum(pt[1] for pt in r.bbox) / len(r.bbox)
                    break
                    
            if dob_y is not None:
                # Find regions above DOB
                candidates = []
                for r in full_regions:
                    cy = sum(pt[1] for pt in r.bbox) / len(r.bbox)
                    if cy < dob_y - 10:  # strictly above
                        # Filter out known header text
                        up = r.text.upper()
                        if "GOVERNMENT" not in up and "INDIA" not in up and "CHANDUKHA" not in up:
                            candidates.append((cy, r.text))
                
                # Sort by Y descending (closest to DOB)
                candidates.sort(key=lambda x: x[0], reverse=True)
                if candidates:
                    results["name"] = FieldOCRResult(text=candidates[0][1], confidence=0.90, raw_regions=[])
                    
        elif doc_type == "pan":
            pan_match = re.search(r"([A-Z]{5}[0-9]{4}[A-Z])", full_text)
            if pan_match:
                results["pan_number"] = FieldOCRResult(
                    text=pan_match.group(1), confidence=0.99, raw_regions=[]
                )
            dob_match = re.search(r"([0-9]{2}[/-][0-9]{2}[/-][0-9]{4})", full_text)
            if dob_match:
                results["dob"] = FieldOCRResult(text=dob_match.group(1), confidence=0.95, raw_regions=[])
                
        elif doc_type == "cv":
            # Re-use text_extractor logic for the raw OCR text!
            from hrm_ocr.pipeline.text_extractor import extract_fields_from_text
            cv_res = extract_fields_from_text(full_text, "cv")
            for field_name, field_val in cv_res.fields.items():
                results[field_name] = FieldOCRResult(text=field_val, confidence=0.85, raw_regions=[])
        
        return results

# -----------------------------------------------------------------------------
# Module-level Engine Registry (Singleton Management)
# -----------------------------------------------------------------------------

_engine_registry: dict[str, OCREngine] = {}

def get_engine(lang: str = "en") -> OCREngine:
    """Get or initialize the OCREngine for the specified language.
    
    Models are loaded exactly once per language during application lifecycle.
    """
    if lang not in _engine_registry:
        repo_root = Path(__file__).resolve().parents[3]
        model_dir = repo_root / "models" / "paddleocr"
        model_dir.mkdir(parents=True, exist_ok=True)
        _engine_registry[lang] = OCREngine(model_dir=model_dir, lang=lang)
    return _engine_registry[lang]


def recognize_regional_field(crop: np.ndarray, script: str) -> FieldOCRResult:
    """Helper to route a crop to a specific regional script engine.
    
    Parameters
    ----------
    crop : np.ndarray
        The pre-cropped field image.
    script : str
        Language code (e.g., 'hi' for Hindi).
    """
    engine = get_engine(lang=script)
    return engine.recognize_field_crop(crop)
