#!/usr/bin/env bash
set -e

echo "Building HRM OCR API image..."
docker build -t hrm-ocr-api:latest -f docker/Dockerfile .

# Assert image size under 300MB
SIZE_BYTES=$(docker inspect -f "{{ .Size }}" hrm-ocr-api:latest)
SIZE_MB=$((SIZE_BYTES / 1024 / 1024))

echo "Image built. Size: ${SIZE_MB}MB"

if [ "$SIZE_MB" -gt 300 ]; then
    echo "WARNING: Image size ($SIZE_MB MB) exceeds the 300MB target!"
else
    echo "SUCCESS: Image size ($SIZE_MB MB) is well within the 300MB target."
fi

echo "Starting services via docker-compose..."
cd docker
docker-compose up -d

echo "Polling API health..."
for i in {1..15}; do
    if curl -s http://localhost:8000/health | grep -q '"status":"ok"'; then
        echo "API is healthy and ready!"
        exit 0
    fi
    echo "Waiting for API to start..."
    sleep 3
done

echo "API failed to become healthy in time."
exit 1
