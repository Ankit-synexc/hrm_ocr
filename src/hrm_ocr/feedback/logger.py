"""
hrm_ocr.feedback.logger
=======================
Active learning feedback logger using SQLAlchemy for structured SQLite storage.
Separates image-based OCR anomalies from raw text-layer validation errors.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class LowConfidenceExtraction(Base):
    __tablename__ = "low_confidence_extractions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    request_id = Column(String(50), index=True)
    doc_type = Column(String(50))
    field_name = Column(String(50))
    image_crop_path = Column(String(255))
    raw_ocr_value = Column(String(255))
    corrected_value = Column(String(255))
    was_corrected = Column(Boolean)
    confidence = Column(Float)
    flag_reason = Column(String(255))
    
    # Active learning fields
    corrected_by_human = Column(Boolean, default=False)
    human_value = Column(String(255), nullable=True)
    pushed_to_ls = Column(Boolean, default=False)
    ls_task_id = Column(Integer, nullable=True)
    extraction_method = Column(String(50))


class ValidationError(Base):
    __tablename__ = "validation_errors"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    request_id = Column(String(50), index=True)
    doc_type = Column(String(50))
    field_name = Column(String(50))
    extracted_value = Column(String(255))
    extraction_method = Column(String(50))
    validation_reason = Column(String(255))


class FeedbackLogger:
    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            repo_root = Path(__file__).resolve().parents[3]
            db_dir = repo_root / "logs"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "feedback.db"
            
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Set up raw crop storage dir
        repo_root = Path(__file__).resolve().parents[3]
        self.crop_dir = repo_root / "data" / "raw" / "feedback"
        
    def log_ocr_issue(
        self,
        request_id: str,
        doc_type: str,
        field_name: str,
        crop: np.ndarray,
        raw_ocr_value: str,
        corrected_value: str,
        was_corrected: bool,
        confidence: float,
        flag_reason: str,
        extraction_method: str = "ocr"
    ) -> None:
        """Log an OCR escalation/flag and save the crop image."""
        date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        daily_dir = self.crop_dir / date_str
        daily_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{request_id}_{field_name}_{datetime.datetime.utcnow().strftime('%H%M%S')}.jpg"
        img_path = daily_dir / filename
        cv2.imwrite(str(img_path), crop)
        
        with self.Session() as session:
            record = LowConfidenceExtraction(
                request_id=request_id,
                doc_type=doc_type,
                field_name=field_name,
                image_crop_path=str(img_path),
                raw_ocr_value=raw_ocr_value,
                corrected_value=corrected_value,
                was_corrected=was_corrected,
                confidence=confidence,
                flag_reason=flag_reason,
                extraction_method=extraction_method
            )
            session.add(record)
            session.commit()
            
        logger.info("Logged OCR feedback for %s/%s", request_id, field_name)

    def log_validation_error(
        self,
        request_id: str,
        doc_type: str,
        field_name: str,
        extracted_value: str,
        extraction_method: str,
        validation_reason: str
    ) -> None:
        """Log a validation error without saving a crop (used mostly for text_layer)."""
        with self.Session() as session:
            record = ValidationError(
                request_id=request_id,
                doc_type=doc_type,
                field_name=field_name,
                extracted_value=extracted_value,
                extraction_method=extraction_method,
                validation_reason=validation_reason
            )
            session.add(record)
            session.commit()
            
        logger.info("Logged Validation error for %s/%s", request_id, field_name)
