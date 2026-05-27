"""
hrm_ocr.api.main
================
FastAPI application entrypoint.
"""
from __future__ import annotations

import os
# Disable PIR API to fallback to the stable static graph executor in Paddle v3.3+
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import datetime
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from hrm_ocr.api.middleware.request_id import RequestIDMiddleware
from hrm_ocr.api.routes.extract import router as extract_router
from hrm_ocr.feedback.logger import FeedbackLogger
from hrm_ocr.models.ocr_engine import get_engine

logger = logging.getLogger("hrm_ocr.api")
limiter = Limiter(key_func=get_remote_address)


# App-level state
STARTUP_TIME: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and Shutdown logic."""
    global STARTUP_TIME
    STARTUP_TIME = time.time()
    
    logger.info("Initializing HRM OCR API...")
    
    # 1. Pre-load OCREngine models into the singleton registry
    try:
        _ = get_engine("en")
        _ = get_engine("hi")
        logger.info("PaddleOCR engines loaded successfully.")
    except Exception as e:
        logger.error("Failed to load PaddleOCR models: %s", e)
        
    # 2. Load field coordinates into memory (warmup)
    repo_root = Path(__file__).resolve().parents[3]
    coords_path = repo_root / "configs" / "field_coords.yaml"
    if coords_path.exists():
        with open(coords_path, "r", encoding="utf-8") as f:
            yaml.safe_load(f)
            
    # 3. Initialize active learning logger
    FeedbackLogger()
    
    logger.info("HRM OCR API startup complete.")
    
    yield
    
    # Shutdown logic
    logger.info("Shutting down HRM OCR API. Releasing resources...")
    # Python GC takes care of the PaddleOCR instances, but we could explicitly clear the registry here


app = FastAPI(
    title="HRM OCR API",
    description="Production-grade rule-based HRM Document OCR.",
    version="0.1.0",
    lifespan=lifespan
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Attach Request ID middleware
app.add_middleware(RequestIDMiddleware)


@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes/Docker probes."""
    uptime_seconds = int(time.time() - STARTUP_TIME) if STARTUP_TIME else 0
    return {
        "status": "ok",
        "model_type": "paddleocr",
        "uptime_seconds": uptime_seconds
    }


# Register main API router
app.include_router(extract_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    # Structured JSON logging can be configured in production via Uvicorn config
    uvicorn.run("hrm_ocr.api.main:app", host="0.0.0.0", port=8000, reload=True)
