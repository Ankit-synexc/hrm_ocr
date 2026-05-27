#!/usr/bin/env python3
"""
scripts/augment_dataset.py
==========================
CLI to augment an existing synthetic dataset with real-world camera noise.

Usage:
  python scripts/augment_dataset.py \\
    --mode augment \\
    --input_dir data/synthetic/ \\
    --output_dir data/augmented/ \\
    --n_variants 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hrm_ocr.pipeline.augment import augment_sample

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment image datasets.")
    parser.add_argument("--mode", choices=["augment"], required=True,
                        help="Operation mode.")
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="Input directory containing 'images' folder.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output directory for augmented images.")
    parser.add_argument("--n_variants", type=int, default=10,
                        help="Number of augmented variants per input image.")
    parser.add_argument("--prob", type=float, default=0.5,
                        help="Probability of applying each augmentation (0.0 to 1.0).")
                        
    args = parser.parse_args()
    
    if args.mode == "augment":
        in_img_dir = args.input_dir / "images"
        if not in_img_dir.exists():
            logger.error("Input images directory not found: %s", in_img_dir)
            sys.exit(1)
            
        out_img_dir = args.output_dir / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = args.output_dir / "augmentation_log.jsonl"
        
        image_files = list(in_img_dir.glob("*.jpg")) + list(in_img_dir.glob("*.png"))
        logger.info("Found %d images to augment.", len(image_files))
        
        total_generated = 0
        
        with open(log_path, "w", encoding="utf-8") as f_log:
            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning("Failed to read image: %s", img_path)
                    continue
                    
                records = augment_sample(img, args.n_variants, args.prob)
                
                for idx, record in enumerate(records):
                    out_filename = f"{img_path.stem}_aug_{idx:03d}.jpg"
                    out_path = out_img_dir / out_filename
                    
                    cv2.imwrite(str(out_path), record.image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    
                    log_entry = {
                        "original_image": img_path.name,
                        "augmented_image": out_filename,
                        "augmentations": record.augmentations_applied,
                        "severity_params": record.severity_params
                    }
                    f_log.write(json.dumps(log_entry) + "\n")
                    total_generated += 1
                    
        logger.info("Successfully generated %d augmented images.", total_generated)
        logger.info("Saved log to %s", log_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
