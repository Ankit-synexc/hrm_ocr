"""
tests/unit/test_augment.py
==========================
Unit tests for the augmentation pipeline.
"""
from __future__ import annotations

import numpy as np

from hrm_ocr.pipeline.augment import (
    AUGMENTATION_FUNCS,
    augment_sample,
)


def _get_dummy_image() -> np.ndarray:
    """Returns a solid white 100x100 BGR image."""
    return np.full((100, 100, 3), 255, dtype=np.uint8)


class TestAugmentations:
    def test_all_functions_execute_without_crashing(self):
        img = _get_dummy_image()
        for name, func in AUGMENTATION_FUNCS.items():
            out = func(img.copy())
            assert out is not None
            assert out.shape == img.shape
            assert out.dtype == np.uint8

    def test_augment_sample_returns_correct_variants(self):
        img = _get_dummy_image()
        records = augment_sample(img, n_variants=3, augmentation_probability=0.5)
        
        assert len(records) == 3
        for rec in records:
            assert rec.image.shape == img.shape
            assert isinstance(rec.augmentations_applied, list)
            assert "p" in rec.severity_params
            
    def test_augment_sample_zero_probability(self):
        """If probability is 0, no augmentations should be applied."""
        img = _get_dummy_image()
        records = augment_sample(img, n_variants=1, augmentation_probability=0.0)
        
        assert len(records) == 1
        assert len(records[0].augmentations_applied) == 0
        # The output image should exactly match the input
        np.testing.assert_array_equal(records[0].image, img)
        
    def test_augment_sample_one_probability(self):
        """If probability is 1, ALL augmentations should be applied."""
        img = _get_dummy_image()
        records = augment_sample(img, n_variants=1, augmentation_probability=1.0)
        
        assert len(records) == 1
        # All available functions should have run
        assert len(records[0].augmentations_applied) == len(AUGMENTATION_FUNCS)
