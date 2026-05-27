"""
tests/unit/test_glyph_cache.py
==============================
Unit tests for the HOG-based Glyph Cache.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from hrm_ocr.models.glyph_cache import CachedOCREngine, GlyphCache
from hrm_ocr.models.ocr_engine import FieldOCRResult


class TestGlyphCache:
    def _get_dummy_patch(self, val: int = 128) -> np.ndarray:
        return np.full((32, 32, 3), val, dtype=np.uint8)

    def test_cache_miss_before_min_samples(self):
        cache = GlyphCache(min_samples=5)
        
        # Add 3 samples
        for i in range(3):
            cache.add("A", self._get_dummy_patch(), 0.99)
            
        assert cache.total_cached == 3
        # Should return None because total_cached (3) < min_samples (5)
        res = cache.query(self._get_dummy_patch())
        assert res is None

    def test_cache_hit_after_min_samples(self):
        cache = GlyphCache(min_samples=2)
        
        # Add a specific pattern for 'A'
        patch_A = np.zeros((32, 32, 3), dtype=np.uint8)
        patch_A[10:20, 10:20] = 255
        cache.add("A", patch_A, 0.99)
        
        # Add a specific pattern for 'B'
        patch_B = np.zeros((32, 32, 3), dtype=np.uint8)
        patch_B[0:10, 0:10] = 255
        cache.add("B", patch_B, 0.99)
        
        assert cache.total_cached == 2
        
        # Query with exact A pattern
        res = cache.query(patch_A)
        assert res is not None
        assert res.char == "A"
        assert res.similarity > 0.95
        
        # Hit rate check
        stats = cache.stats()
        assert stats["cache_hits"] == 1
        assert stats["hit_rate"] == 1.0

    def test_cache_reset(self):
        cache = GlyphCache(min_samples=1)
        cache.add("A", self._get_dummy_patch(), 0.99)
        
        cache.query(self._get_dummy_patch())
        assert cache.stats()["cache_hits"] == 1
        
        cache.reset()
        assert cache.total_cached == 0
        assert cache.stats()["cache_hits"] == 0


class TestCachedOCREngine:
    def test_recognize_field_crop_integration(self):
        # Mock underlying OCREngine
        mock_engine = MagicMock()
        mock_engine.recognize_field_crop.return_value = FieldOCRResult(
            text="HELLO",
            confidence=0.98,
            raw_regions=[]
        )
        
        # Use min_samples=1 so it caches and returns immediately
        cache = GlyphCache(min_samples=1)
        cached_engine = CachedOCREngine(engine=mock_engine, cache=cache)
        
        dummy_crop = np.zeros((32, 32, 3), dtype=np.uint8)
        
        # First call: cache miss
        res1 = cached_engine.recognize_field_crop(dummy_crop)
        assert not res1.cache_hit
        assert res1.text == "HELLO"
        mock_engine.recognize_field_crop.assert_called_once()
        
        # Second call: cache hit
        res2 = cached_engine.recognize_field_crop(dummy_crop)
        assert res2.cache_hit
        assert res2.text == "HELLO"
        # The underlying engine shouldn't be called a second time
        assert mock_engine.recognize_field_crop.call_count == 1
        
        # Reset session
        cached_engine.reset_session()
        assert cached_engine.cache.total_cached == 0
