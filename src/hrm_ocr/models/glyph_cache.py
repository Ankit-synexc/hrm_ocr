"""
hrm_ocr.models.glyph_cache
==========================
Per-document glyph cache using HOG descriptors and cosine similarity.
Skips ML overhead for repeating text elements across a single document.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from skimage.feature import hog

from hrm_ocr.models.ocr_engine import OCREngine

logger = logging.getLogger(__name__)


@dataclass
class CacheQueryResult:
    char: str
    similarity: float


@dataclass
class CachedFieldOCRResult:
    text: str
    confidence: float
    cache_hit: bool


class GlyphCache:
    def __init__(self, min_samples: int = 50, similarity_threshold: float = 0.88):
        self.min_samples = min_samples
        self.similarity_threshold = similarity_threshold
        
        # Mapping from char (or text string) to list of HOG descriptors
        self.cache: dict[str, list[np.ndarray]] = {}
        
        self.total_cached = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _compute_hog(self, patch: np.ndarray) -> np.ndarray:
        """Compute a fixed-length HOG descriptor for a patch."""
        # Convert to grayscale and resize to a fixed dimension (e.g., 32x32) 
        # so HOG descriptors are always the same shape for cosine similarity.
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
            
        resized = cv2.resize(gray, (32, 32))
        
        descriptor = hog(
            resized,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            feature_vector=True
        )
        return descriptor

    def add(self, char: str, patch: np.ndarray, confidence: float) -> None:
        """Compute HOG and store if confidence is high."""
        if confidence >= 0.95:
            try:
                descriptor = self._compute_hog(patch)
                if char not in self.cache:
                    self.cache[char] = []
                self.cache[char].append(descriptor)
                self.total_cached += 1
            except Exception as e:
                logger.warning("Failed to compute HOG descriptor: %s", e)

    def query(self, patch: np.ndarray) -> CacheQueryResult | None:
        """Query the cache using cosine similarity."""
        if self.total_cached < self.min_samples:
            return None
            
        try:
            descriptor = self._compute_hog(patch)
        except Exception as e:
            logger.warning("Failed to compute HOG descriptor for query: %s", e)
            return None

        best_char = None
        best_sim = -1.0
        
        # Compute cosine similarity: dot product / (norm * norm)
        desc_norm = np.linalg.norm(descriptor)
        if desc_norm == 0:
            return None
            
        for char, stored_descs in self.cache.items():
            for stored in stored_descs:
                stored_norm = np.linalg.norm(stored)
                if stored_norm == 0:
                    continue
                    
                sim = np.dot(descriptor, stored) / (desc_norm * stored_norm)
                if sim > best_sim:
                    best_sim = sim
                    best_char = char
                    
        if best_sim >= self.similarity_threshold and best_char is not None:
            self.cache_hits += 1
            return CacheQueryResult(char=best_char, similarity=float(best_sim))
            
        self.cache_misses += 1
        return None

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_queries) if total_queries > 0 else 0.0
        return {
            "total_cached": self.total_cached,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }

    def reset(self) -> None:
        """Clear all stored descriptors and reset counters."""
        self.cache.clear()
        self.total_cached = 0
        self.cache_hits = 0
        self.cache_misses = 0


class CachedOCREngine:
    def __init__(self, engine: OCREngine, cache: GlyphCache | None = None):
        """Wrap an OCREngine with a per-document GlyphCache."""
        self.engine = engine
        self.cache = cache or GlyphCache()

    def recognize_field_crop(self, crop: np.ndarray) -> CachedFieldOCRResult:
        """Try cache first; on miss, call underlying engine and cache result."""
        # Try cache
        cache_res = self.cache.query(crop)
        if cache_res is not None:
            return CachedFieldOCRResult(
                text=cache_res.char,
                confidence=cache_res.similarity,
                cache_hit=True
            )
            
        # Cache miss
        ocr_res = self.engine.recognize_field_crop(crop)
        
        # Add to cache (the prompt signature implies we pass the whole text/patch)
        self.cache.add(char=ocr_res.text, patch=crop, confidence=ocr_res.confidence)
        
        return CachedFieldOCRResult(
            text=ocr_res.text,
            confidence=ocr_res.confidence,
            cache_hit=False
        )

    def reset_session(self) -> None:
        """Reset the cache between documents."""
        self.cache.reset()
