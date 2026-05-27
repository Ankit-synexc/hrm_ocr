# Stage 1: Builder
FROM python:3.11-slim as builder

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.6.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app
COPY pyproject.toml poetry.lock* ./

# Generate requirements and install into a virtualenv
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes || echo "WARNING: poetry export failed, make sure poetry is initialized."
RUN python -m venv /opt/venv

# Since poetry export might fail if pyproject is missing or out of sync, 
# we also provide a fallback pip install if requirements.txt is empty
RUN if [ -s requirements.txt ]; then \
        /opt/venv/bin/pip install -r requirements.txt; \
    else \
        /opt/venv/bin/pip install fastapi uvicorn paddlepaddle paddleocr pdfplumber opencv-python-headless pydantic scikit-image pyyaml sqlalchemy slowapi; \
    fi

# Stage 2: Runtime
FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1001 appuser && useradd -u 1001 -g appuser -s /bin/bash -m appuser

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy source code and config
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/paddleocr/ ./models/paddleocr/

# Ensure models are present (build-time check)
RUN test -d ./models/paddleocr/det_en || echo "Warning: det_en model missing"
RUN test -d ./models/paddleocr/rec_en || echo "Warning: rec_en model missing"

# Import checks
RUN python -c "from paddleocr import PaddleOCR; import cv2; import fastapi; import pdfplumber"

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.hrm_ocr.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
