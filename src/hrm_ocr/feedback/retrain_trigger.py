"""
hrm_ocr.feedback.retrain_trigger
================================
Analyze human corrections to automatically suggest rule updates in substitutions.py
and threshold adjustments, avoiding heavy model retraining.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from hrm_ocr.feedback.logger import FeedbackLogger, LowConfidenceExtraction

logger = logging.getLogger(__name__)


def check_and_update_rules() -> None:
    """Analyze accumulated corrections to suggest deterministic rule updates."""
    logger_instance = FeedbackLogger()
    
    with logger_instance.Session() as session:
        corrected_records = session.query(LowConfidenceExtraction).filter(
            LowConfidenceExtraction.corrected_by_human == True
        ).all()
        
        if len(corrected_records) < 50:
            logger.info(
                "Not enough corrections to trigger rule analysis. "
                "(Found %d, need 50)", len(corrected_records)
            )
            return
            
        logger.info("Analyzing %d corrections...", len(corrected_records))
        
        # Character-level error analysis for substitutions.py
        char_replacements = Counter()
        
        for record in corrected_records:
            raw = record.raw_ocr_value or ""
            human = record.human_value or ""
            
            # Simple 1-to-1 length alignment mapping
            if len(raw) == len(human) and len(raw) > 0:
                for r_char, h_char in zip(raw, human):
                    if r_char != h_char:
                        char_replacements[(r_char, h_char)] += 1
                        
        # Identify the most common (raw_ocr, human) errors
        suggested_digit_subs = {}
        suggested_alpha_subs = {}
        
        for (r_char, h_char), count in char_replacements.most_common():
            if count >= 3:  # Must have been seen at least 3 times
                if h_char.isdigit():
                    suggested_digit_subs[r_char] = h_char
                elif h_char.isalpha():
                    suggested_alpha_subs[r_char] = h_char
                    
        repo_root = Path(__file__).resolve().parents[3]
        logs_dir = repo_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        suggested_path = logs_dir / "suggested_rule_updates.json"
        
        payload = {
            "total_records_analyzed": len(corrected_records),
            "suggested_digit_subs": suggested_digit_subs,
            "suggested_alpha_subs": suggested_alpha_subs
        }
        
        with open(suggested_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
            
        logger.info("Rule analysis complete. Suggested updates written to %s", suggested_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    check_and_update_rules()
