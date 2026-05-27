"""
tests/integration/test_feedback_loop.py
=======================================
Integration tests for the Active Learning Feedback Loop.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np

from hrm_ocr.feedback.logger import FeedbackLogger, LowConfidenceExtraction, ValidationError
from hrm_ocr.feedback.pull_corrections import pull_corrections
from hrm_ocr.feedback.push_to_ls import push_pending_to_label_studio
from hrm_ocr.feedback.retrain_trigger import check_and_update_rules


class TestFeedbackLoop:
    def test_end_to_end_feedback_loop(self, tmp_path: Path):
        # 1. Setup DB
        db_path = tmp_path / "feedback.db"
        logger = FeedbackLogger(db_path=db_path)
        
        # 2. Log OCR Issue
        dummy_crop = np.zeros((32, 32, 3), dtype=np.uint8)
        logger.log_ocr_issue(
            request_id="req_123",
            doc_type="pan",
            field_name="pan_number",
            crop=dummy_crop,
            raw_ocr_value="ABCDE123OF",
            corrected_value="ABCDE1230F",
            was_corrected=True,
            confidence=0.65,
            flag_reason="low_conf_validation_fail"
        )
        
        # Manually alter the timestamp to be > 1 hour old to trigger push
        with logger.Session() as session:
            record = session.query(LowConfidenceExtraction).first()
            assert record is not None
            record.timestamp = datetime.datetime.utcnow() - datetime.timedelta(hours=2)
            session.commit()
            
        # 3. Push to LS
        # Monkey patch FeedbackLogger in the modules to use our tmp_path DB
        import hrm_ocr.feedback.push_to_ls as p2ls
        import hrm_ocr.feedback.pull_corrections as pull
        import hrm_ocr.feedback.retrain_trigger as retrain
        
        orig_push = p2ls.FeedbackLogger
        orig_pull = pull.FeedbackLogger
        orig_retrain = retrain.FeedbackLogger
        
        p2ls.FeedbackLogger = lambda: logger
        pull.FeedbackLogger = lambda: logger
        retrain.FeedbackLogger = lambda: logger
        
        try:
            push_pending_to_label_studio()
            
            with logger.Session() as session:
                record = session.query(LowConfidenceExtraction).first()
                assert record.pushed_to_ls is True
                assert record.ls_task_id is not None
                
            # 4. Pull Corrections (Mocks a human correcting it)
            # In pull_corrections mock logic, record.id % 2 == 0 is corrected. 
            # Auto-increment starts at 1, so id=1 is NOT corrected. Let's add another one.
            logger.log_ocr_issue(
                request_id="req_124", doc_type="pan", field_name="name", crop=dummy_crop,
                raw_ocr_value="R4HUL", corrected_value="R4HUL", was_corrected=False,
                confidence=0.6, flag_reason="val_fail"
            )
            with logger.Session() as session:
                record2 = session.query(LowConfidenceExtraction).filter_by(request_id="req_124").first()
                record2.timestamp = datetime.datetime.utcnow() - datetime.timedelta(hours=2)
                session.commit()
                
            push_pending_to_label_studio()
            pull_corrections()
            
            with logger.Session() as session:
                # id 2 should be corrected
                r2 = session.query(LowConfidenceExtraction).filter_by(id=2).first()
                assert r2.corrected_by_human is True
                assert r2.human_value == "R4HUL_FIXED"
                
            # 5. Retrain Trigger
            # We need 50+ records to trigger the analysis in the script, so we'll 
            # just execute it and ensure it doesn't crash (it will early exit).
            check_and_update_rules()
            
        finally:
            p2ls.FeedbackLogger = orig_push
            pull.FeedbackLogger = orig_pull
            retrain.FeedbackLogger = orig_retrain

    def test_log_validation_error(self, tmp_path: Path):
        db_path = tmp_path / "feedback_val.db"
        logger = FeedbackLogger(db_path=db_path)
        
        logger.log_validation_error(
            request_id="req_999",
            doc_type="aadhaar",
            field_name="uid",
            extracted_value="1234",
            extraction_method="text_layer",
            validation_reason="length_fail"
        )
        
        with logger.Session() as session:
            record = session.query(ValidationError).first()
            assert record is not None
            assert record.request_id == "req_999"
            assert record.extraction_method == "text_layer"
