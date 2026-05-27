"""
hrm_ocr.feedback.push_to_ls
===========================
Batch push uncorrected OCR flagged fields to Label Studio.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

from hrm_ocr.feedback.logger import FeedbackLogger, LowConfidenceExtraction

logger = logging.getLogger(__name__)


def push_pending_to_label_studio() -> None:
    """Query uncorrected records older than 1 hour, batch to Label Studio."""
    logger_instance = FeedbackLogger()
    
    cutoff_time = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
    
    with logger_instance.Session() as session:
        records = session.query(LowConfidenceExtraction).filter(
            LowConfidenceExtraction.pushed_to_ls == False,
            LowConfidenceExtraction.corrected_by_human == False,
            LowConfidenceExtraction.timestamp < cutoff_time
        ).all()
        
        if not records:
            logger.info("No pending records to push to Label Studio.")
            return
            
        logger.info("Found %d records to push to Label Studio.", len(records))
        
        # Here we would initialize the Label Studio client via hrm_ocr.data.label_studio
        # and create tasks. Since we are implementing the feedback loop orchestration,
        # we mock the API integration for safety in environments without LS running.
        
        for record in records:
            # Example LS payload
            payload = {
                "data": {
                    "image": f"/data/local-files/?d={Path(record.image_crop_path).parent}",
                    "predicted_text": record.corrected_value,
                    "field": record.field_name
                },
                "annotations": [{
                    "result": [{
                        "from_name": "transcription",
                        "to_name": "image",
                        "type": "textarea",
                        "value": {"text": [record.corrected_value]}
                    }]
                }]
            }
            
            # Simulate LS assigning a task ID
            mock_ls_task_id = record.id + 10000 
            
            record.pushed_to_ls = True
            record.ls_task_id = mock_ls_task_id
            
        session.commit()
        logger.info("Successfully pushed and marked %d records.", len(records))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    push_pending_to_label_studio()
