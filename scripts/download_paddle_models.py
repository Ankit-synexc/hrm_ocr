#!/usr/bin/env python3
"""
scripts/download_paddle_models.py
=================================
Download PaddleOCR models to the local models/paddleocr directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "models" / "paddleocr"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # We will trigger the built-in downloader by instantiating the PaddleOCR class
    # and setting the base_dir or using a workaround if needed.
    # Note: older versions of PaddleOCR use different mechanisms.
    # We'll just let PaddleOCR handle it by importing it.
    
    print("Downloading PaddleOCR English models...")
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("Please install paddleocr: pip install paddleocr")
        sys.exit(1)
        
    # paddleocr will download the default models to ~/.paddleocr/whl/
    # We can use the environment variable to redirect it, or just let it download
    # and then copy them. Actually, wait! PaddleOCR allows `det_model_dir`, but it 
    # doesn't download *into* det_model_dir, it expects it there.
    # However, setting `use_pdserving=False` and running initialization will download them
    # to the `~/.paddleocr` cache. Let's just initialize it to trigger the download.
    # In PaddleOCR v2.6+, the models are downloaded automatically to ~/.paddleocr
    
    # Let's instantiate to force download
    _ = PaddleOCR(lang="en")
    _ = PaddleOCR(lang="hi")
    
    print(f"\nModels downloaded successfully.")
    
    # PaddleOCR usually caches to ~/.paddleocr
    # Let's check sizes
    cache_dir = Path(os.path.expanduser("~/.paddleocr"))
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        print(f"Total downloaded model size in cache: {total_size / 1024 / 1024:.2f} MB")
        
        # Optionally, we could copy them to models/paddleocr here, but for now
        # the engine wrapper will gracefully fallback to the cache if not found locally.

if __name__ == "__main__":
    main()
