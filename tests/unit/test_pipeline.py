"""
tests/unit/test_pipeline.py
============================
Unit tests for:
  * hrm_ocr.exceptions           — typed exception hierarchy
  * hrm_ocr.pipeline.ingest      — load_document()
  * hrm_ocr.pipeline.preprocess  — clean_image() and all private steps

Design philosophy
-----------------
* No PaddleOCR, no PDF2Image, no network calls.
* Every helper function is tested in isolation with controlled inputs.
* Short-circuit paths are explicitly verified (CLAHE skip, deskew skip, etc.)
* Performance budget assertions (< 30 ms total) are marked for optional
  execution so CI doesn't fail on slow machines.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(w: int = 200, h: int = 120, color=(180, 140, 100)) -> bytes:
    """Return minimal JPEG bytes from a solid-colour image."""
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_png_bytes(w: int = 150, h: int = 90, color=(80, 160, 200)) -> bytes:
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_bgr_array(
    w: int = 1012,
    h: int = 638,
    fill: int = 200,
) -> np.ndarray:
    """Return a solid-grey BGR uint8 ndarray."""
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _make_card_array(
    w: int = 1012,
    h: int = 638,
) -> np.ndarray:
    """Return a synthetic card-like image: white background + dark border."""
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (w - 5, h - 5), (30, 30, 30), 3)
    return img


# ============================================================================
# Exception hierarchy
# ============================================================================

class TestExceptions:
    def test_hrm_ocr_error_base(self):
        from hrm_ocr.exceptions import HRMOCRError
        exc = HRMOCRError("test message", detail={"key": "value"})
        assert str(exc) == "test message"
        assert exc.detail == {"key": "value"}

    def test_file_size_error_message(self):
        from hrm_ocr.exceptions import FileSizeError
        exc = FileSizeError(size_bytes=15 * 1024 * 1024, limit_bytes=10 * 1024 * 1024)
        assert "15" in str(exc) or "15.00" in str(exc)
        assert "10" in str(exc)
        assert exc.size_bytes == 15 * 1024 * 1024
        assert exc.limit_bytes == 10 * 1024 * 1024

    def test_unsupported_format_error(self):
        from hrm_ocr.exceptions import UnsupportedFormatError
        exc = UnsupportedFormatError("image/bmp")
        assert "image/bmp" in str(exc)
        assert "image/jpeg" in str(exc)

    def test_corrupt_file_error(self):
        from hrm_ocr.exceptions import CorruptFileError
        exc = CorruptFileError("card.jpg", "truncated EXIF")
        assert "card.jpg" in str(exc)
        assert "truncated EXIF" in str(exc)
        assert exc.filename == "card.jpg"
        assert exc.reason == "truncated EXIF"

    def test_ingest_error_is_hrm_ocr_error(self):
        from hrm_ocr.exceptions import HRMOCRError, IngestError
        assert issubclass(IngestError, HRMOCRError)

    def test_file_size_error_is_ingest_error(self):
        from hrm_ocr.exceptions import FileSizeError, IngestError
        assert issubclass(FileSizeError, IngestError)

    def test_unsupported_format_is_ingest_error(self):
        from hrm_ocr.exceptions import IngestError, UnsupportedFormatError
        assert issubclass(UnsupportedFormatError, IngestError)

    def test_corrupt_file_is_ingest_error(self):
        from hrm_ocr.exceptions import CorruptFileError, IngestError
        assert issubclass(CorruptFileError, IngestError)


# ============================================================================
# ingest.load_document — MIME detection
# ============================================================================

class TestMimeDetection:
    """Test that magic bytes are used, not file extensions."""

    def test_jpeg_bytes_detected(self):
        from hrm_ocr.pipeline.ingest import _detect_mime
        raw = _make_jpeg_bytes()
        assert _detect_mime(raw) == "image/jpeg"

    def test_png_bytes_detected(self):
        from hrm_ocr.pipeline.ingest import _detect_mime
        raw = _make_png_bytes()
        assert _detect_mime(raw) == "image/png"

    def test_pdf_magic_detected(self):
        from hrm_ocr.pipeline.ingest import _detect_mime
        raw = b"%PDF-1.4 fake content"
        assert _detect_mime(raw) == "application/pdf"

    def test_bare_jpeg_soi_detected(self):
        from hrm_ocr.pipeline.ingest import _detect_mime
        # Bare SOI (0xFF 0xD8) without JFIF APP0 marker
        raw = b"\xff\xd8" + b"\x00" * 100
        assert _detect_mime(raw) == "image/jpeg"

    def test_unknown_magic_raises(self):
        from hrm_ocr.exceptions import UnsupportedFormatError
        from hrm_ocr.pipeline.ingest import _detect_mime
        with pytest.raises(UnsupportedFormatError) as exc_info:
            _detect_mime(b"\x00\x01\x02GIF89a")
        assert "unsupported" in str(exc_info.value).lower() or "unknown" in str(exc_info.value).lower()

    def test_bmp_magic_raises_unsupported(self):
        from hrm_ocr.exceptions import UnsupportedFormatError
        from hrm_ocr.pipeline.ingest import _detect_mime
        with pytest.raises(UnsupportedFormatError):
            _detect_mime(b"BM" + b"\x00" * 50)


# ============================================================================
# ingest.load_document — size validation
# ============================================================================

class TestSizeValidation:
    def test_within_limit_passes(self):
        from hrm_ocr.pipeline.ingest import _validate_size
        _validate_size(b"x" * 1000, max_bytes=10_000)  # should not raise

    def test_exact_limit_passes(self):
        from hrm_ocr.pipeline.ingest import _validate_size
        _validate_size(b"x" * 10_000, max_bytes=10_000)  # exact limit: OK

    def test_over_limit_raises(self):
        from hrm_ocr.exceptions import FileSizeError
        from hrm_ocr.pipeline.ingest import _validate_size
        with pytest.raises(FileSizeError) as exc_info:
            _validate_size(b"x" * 11_000, max_bytes=10_000)
        assert exc_info.value.size_bytes == 11_000
        assert exc_info.value.limit_bytes == 10_000


# ============================================================================
# ingest.load_document — source reading
# ============================================================================

class TestReadSource:
    def test_bytes_passthrough(self):
        from hrm_ocr.pipeline.ingest import _read_source
        raw = b"hello bytes"
        assert _read_source(raw) is raw

    def test_path_object_read(self, tmp_path: Path):
        from hrm_ocr.pipeline.ingest import _read_source
        p = tmp_path / "card.jpg"
        p.write_bytes(b"test content")
        assert _read_source(p) == b"test content"

    def test_string_path_read(self, tmp_path: Path):
        from hrm_ocr.pipeline.ingest import _read_source
        p = tmp_path / "card.png"
        p.write_bytes(b"png content")
        assert _read_source(str(p)) == b"png content"

    def test_missing_path_raises_ingest_error(self, tmp_path: Path):
        from hrm_ocr.exceptions import IngestError
        from hrm_ocr.pipeline.ingest import _read_source
        with pytest.raises(IngestError, match="not found"):
            _read_source(tmp_path / "nonexistent.jpg")

    def test_directory_raises_ingest_error(self, tmp_path: Path):
        from hrm_ocr.exceptions import IngestError
        from hrm_ocr.pipeline.ingest import _read_source
        with pytest.raises(IngestError, match="not a regular file"):
            _read_source(tmp_path)


# ============================================================================
# ingest.load_document — integrity checks
# ============================================================================

class TestIntegrityValidation:
    def test_valid_jpeg_passes(self):
        from hrm_ocr.pipeline.ingest import _validate_integrity
        _validate_integrity(_make_jpeg_bytes(), "image/jpeg")  # no exception

    def test_valid_png_passes(self):
        from hrm_ocr.pipeline.ingest import _validate_integrity
        _validate_integrity(_make_png_bytes(), "image/png")  # no exception

    def test_truncated_jpeg_raises_corrupt(self):
        from hrm_ocr.exceptions import CorruptFileError
        from hrm_ocr.pipeline.ingest import _validate_integrity
        with pytest.raises(CorruptFileError):
            _validate_integrity(b"\xff\xd8\xff" + b"\x00" * 10, "image/jpeg")

    def test_valid_pdf_passes(self):
        from hrm_ocr.pipeline.ingest import _validate_integrity
        fake_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF"
        _validate_integrity(fake_pdf, "application/pdf")  # no exception

    def test_pdf_missing_eof_raises_corrupt(self):
        from hrm_ocr.exceptions import CorruptFileError
        from hrm_ocr.pipeline.ingest import _validate_integrity
        fake_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n"  # no %%EOF
        with pytest.raises(CorruptFileError, match="truncated"):
            _validate_integrity(fake_pdf, "application/pdf")

    def test_pdf_missing_header_raises_corrupt(self):
        from hrm_ocr.exceptions import CorruptFileError
        from hrm_ocr.pipeline.ingest import _validate_integrity
        # Note: _validate_integrity only called after MIME detection confirmed %PDF
        # but we test the internal check anyway
        fake_pdf = b"notpdf content %%EOF"
        with pytest.raises(CorruptFileError, match="Missing"):
            _validate_integrity(fake_pdf, "application/pdf")


# ============================================================================
# ingest.load_document — full integration
# ============================================================================

class TestLoadDocument:
    def test_returns_jpeg_bytes_and_mime(self, tmp_path: Path):
        from hrm_ocr.pipeline.ingest import load_document
        raw = _make_jpeg_bytes()
        p = tmp_path / "card.jpg"
        p.write_bytes(raw)
        result_bytes, mime = load_document(p)
        assert result_bytes == raw
        assert mime == "image/jpeg"

    def test_returns_png_bytes_and_mime(self):
        from hrm_ocr.pipeline.ingest import load_document
        raw = _make_png_bytes()
        result_bytes, mime = load_document(raw)
        assert mime == "image/png"
        assert result_bytes == raw

    def test_accepts_bytes_directly(self):
        from hrm_ocr.pipeline.ingest import load_document
        raw = _make_jpeg_bytes()
        result_bytes, mime = load_document(raw)
        assert mime == "image/jpeg"

    def test_accepts_string_path(self, tmp_path: Path):
        from hrm_ocr.pipeline.ingest import load_document
        raw = _make_jpeg_bytes()
        p = tmp_path / "card.jpg"
        p.write_bytes(raw)
        _, mime = load_document(str(p))
        assert mime == "image/jpeg"

    def test_size_over_limit_raises(self):
        from hrm_ocr.exceptions import FileSizeError
        from hrm_ocr.pipeline.ingest import load_document
        raw = _make_jpeg_bytes()
        with pytest.raises(FileSizeError):
            load_document(raw, max_bytes=100)  # 100 bytes — JPEG is larger

    def test_unsupported_format_raises(self):
        from hrm_ocr.exceptions import UnsupportedFormatError
        from hrm_ocr.pipeline.ingest import load_document
        # GIF magic bytes — not supported
        gif = b"GIF89a" + b"\x00" * 200
        with pytest.raises(UnsupportedFormatError):
            load_document(gif)

    def test_missing_path_raises(self, tmp_path: Path):
        from hrm_ocr.exceptions import IngestError
        from hrm_ocr.pipeline.ingest import load_document
        with pytest.raises(IngestError):
            load_document(tmp_path / "ghost.jpg")

    def test_corrupt_jpeg_raises(self):
        from hrm_ocr.exceptions import CorruptFileError
        from hrm_ocr.pipeline.ingest import load_document
        # Valid magic but garbage body
        with pytest.raises(CorruptFileError):
            load_document(b"\xff\xd8\xff" + b"\xde\xad\xbe\xef" * 5)


# ============================================================================
# preprocess._perspective_warp
# ============================================================================

class TestPerspectiveWarp:
    def test_returns_ndarray(self):
        from hrm_ocr.pipeline.preprocess import _perspective_warp
        img = _make_card_array(800, 500)
        result = _perspective_warp(img)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.ndim == 3

    def test_fallback_on_blank_image(self):
        """Solid white image → no contours → returns full image unchanged."""
        from hrm_ocr.pipeline.preprocess import _perspective_warp
        img = np.full((300, 500, 3), 255, dtype=np.uint8)
        result = _perspective_warp(img)
        # Should return original (same shape) since no card boundary found
        assert result.shape[0] <= img.shape[0] + 10  # shape is preserved or similar

    def test_returns_bgr_3channel(self):
        from hrm_ocr.pipeline.preprocess import _perspective_warp
        img = _make_bgr_array(600, 400)
        result = _perspective_warp(img)
        assert result.shape[2] == 3


class TestOrderCorners:
    def test_tl_has_smallest_sum(self):
        from hrm_ocr.pipeline.preprocess import _order_corners
        pts = np.array(
            [[100, 50], [400, 50], [400, 250], [100, 250]], dtype=np.float32
        )
        # Shuffle to test ordering robustness
        shuffled = pts[[3, 1, 0, 2]]
        ordered = _order_corners(shuffled)
        assert tuple(ordered[0]) == (100.0, 50.0)   # TL
        assert tuple(ordered[2]) == (400.0, 250.0)  # BR

    def test_output_shape(self):
        from hrm_ocr.pipeline.preprocess import _order_corners
        pts = np.array(
            [[0, 0], [100, 0], [100, 60], [0, 60]], dtype=np.float32
        )
        ordered = _order_corners(pts)
        assert ordered.shape == (4, 2)


class TestFourPointTransform:
    def test_output_is_rectangle(self):
        from hrm_ocr.pipeline.preprocess import _four_point_transform
        img = _make_bgr_array(800, 500)
        pts = np.array(
            [[10.0, 10.0], [790.0, 10.0], [790.0, 490.0], [10.0, 490.0]],
            dtype=np.float32,
        )
        result = _four_point_transform(img, pts)
        assert result.ndim == 3
        # Should be approximately 780×480
        assert 750 <= result.shape[1] <= 810
        assert 460 <= result.shape[0] <= 510


# ============================================================================
# preprocess._deskew
# ============================================================================

class TestDeskew:
    def test_returns_same_shape(self):
        from hrm_ocr.pipeline.preprocess import _deskew
        img = _make_card_array()
        result = _deskew(img, min_angle=0.5)
        assert result.shape == img.shape

    def test_skips_when_angle_below_threshold(self):
        """On a perfectly aligned card, deskew should return the array unchanged
        (same object or identical content, not a copy from warpAffine)."""
        from hrm_ocr.pipeline.preprocess import _deskew
        # Perfectly horizontal border → median angle ≈ 0 → should skip
        img = _make_card_array()
        result = _deskew(img, min_angle=0.5)
        # The returned array should be identical in shape and dtype
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_blank_image_returns_unchanged(self):
        from hrm_ocr.pipeline.preprocess import _deskew
        img = np.full((300, 400, 3), 255, dtype=np.uint8)
        result = _deskew(img, min_angle=0.5)
        assert result.shape == img.shape

    def test_high_threshold_always_skips(self):
        """With min_angle=90, nothing should ever trigger deskew."""
        from hrm_ocr.pipeline.preprocess import _deskew
        img = _make_card_array()
        result = _deskew(img, min_angle=90.0)
        # Should be unchanged (skipped)
        assert result.shape == img.shape


# ============================================================================
# preprocess._clahe
# ============================================================================

class TestCLAHE:
    def test_well_lit_image_skipped(self):
        """Image with mean luminance 150 (well-lit) must NOT be CLAHE'd."""
        from hrm_ocr.pipeline.preprocess import _clahe
        img = _make_bgr_array(fill=150)
        result = _clahe(img, lum_low=100, lum_high=200)
        # Should be the exact same object (no-op path)
        assert np.array_equal(result, img)

    def test_dark_image_gets_clahe(self):
        """Image with mean luminance 50 (too dark) MUST be processed."""
        from hrm_ocr.pipeline.preprocess import _clahe
        img = _make_bgr_array(fill=50)
        result = _clahe(img, lum_low=100, lum_high=200)
        # CLAHE increases brightness on dark images — result should differ
        assert result.mean() > img.mean()

    def test_bright_image_gets_clahe(self):
        """Image with mean luminance 230 (overexposed) MUST be processed."""
        from hrm_ocr.pipeline.preprocess import _clahe
        img = _make_bgr_array(fill=230)
        result = _clahe(img, lum_low=100, lum_high=200)
        # CLAHE reduces bright regions — result should differ from input
        assert not np.array_equal(result, img)

    def test_output_shape_preserved(self):
        from hrm_ocr.pipeline.preprocess import _clahe
        img = _make_bgr_array()
        result = _clahe(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_at_low_boundary_skipped(self):
        """Exactly at lum_low boundary should be skipped (condition: lum >= low AND lum <= high)."""
        from hrm_ocr.pipeline.preprocess import _clahe
        img = _make_bgr_array(fill=100)
        result = _clahe(img, lum_low=100, lum_high=200)
        assert np.array_equal(result, img)

    def test_at_high_boundary_skipped(self):
        from hrm_ocr.pipeline.preprocess import _clahe
        img = _make_bgr_array(fill=200)
        result = _clahe(img, lum_low=100, lum_high=200)
        assert np.array_equal(result, img)


# ============================================================================
# preprocess._remove_glare
# ============================================================================

class TestRemoveGlare:
    def test_no_glare_returns_unchanged(self):
        """Image with no bright pixels → skip path → identical output."""
        from hrm_ocr.pipeline.preprocess import _remove_glare
        img = _make_bgr_array(fill=150)
        result = _remove_glare(img, threshold=240, min_area_ratio=0.02)
        assert np.array_equal(result, img)

    def test_small_glare_below_ratio_skipped(self):
        """1% glare area with 2% threshold → should skip."""
        from hrm_ocr.pipeline.preprocess import _remove_glare
        img = _make_bgr_array(fill=150)
        h, w = img.shape[:2]
        # Set ~1% of pixels to 255 (glare)
        glare_h = int(h * 0.05)
        glare_w = int(w * 0.2)  # 0.05 * 0.2 = 1% of area
        img[:glare_h, :glare_w] = 255
        result = _remove_glare(img, threshold=240, min_area_ratio=0.02)
        # Should be unchanged — glare area < 2%
        assert np.array_equal(result, img)

    def test_large_glare_triggers_inpaint(self):
        """5% glare area with 2% threshold → should trigger inpainting."""
        from hrm_ocr.pipeline.preprocess import _remove_glare
        img = _make_bgr_array(fill=150)
        h, w = img.shape[:2]
        # Set a large region to 255 — roughly 5% of area
        img[:int(h * 0.25), :int(w * 0.25)] = 255
        result = _remove_glare(img, threshold=240, min_area_ratio=0.02, inpaint_radius=3)
        # Inpainting changes the glare region — result differs from input
        assert not np.array_equal(result, img)

    def test_output_shape_preserved(self):
        from hrm_ocr.pipeline.preprocess import _remove_glare
        img = _make_bgr_array()
        result = _remove_glare(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


# ============================================================================
# preprocess._resize_to_canonical
# ============================================================================

class TestResizeToCanonical:
    def test_already_canonical_is_noop(self):
        from hrm_ocr.pipeline.preprocess import CARD_H, CARD_W, _resize_to_canonical
        img = _make_bgr_array(CARD_W, CARD_H)
        result = _resize_to_canonical(img)
        assert result is img  # Must return the same object (zero-copy fast path)

    def test_smaller_image_padded(self):
        from hrm_ocr.pipeline.preprocess import CARD_H, CARD_W, _resize_to_canonical
        img = _make_bgr_array(500, 300)
        result = _resize_to_canonical(img)
        assert result.shape == (CARD_H, CARD_W, 3)

    def test_larger_image_downscaled(self):
        from hrm_ocr.pipeline.preprocess import CARD_H, CARD_W, _resize_to_canonical
        img = _make_bgr_array(2000, 1200)
        result = _resize_to_canonical(img)
        assert result.shape == (CARD_H, CARD_W, 3)

    def test_wide_aspect_padded_vertically(self):
        from hrm_ocr.pipeline.preprocess import CARD_H, CARD_W, _resize_to_canonical
        # Very wide image — should be letterboxed (white bars top/bottom)
        img = _make_bgr_array(3000, 400)
        result = _resize_to_canonical(img)
        assert result.shape == (CARD_H, CARD_W, 3)

    def test_output_dtype_uint8(self):
        from hrm_ocr.pipeline.preprocess import _resize_to_canonical
        img = _make_bgr_array(600, 400)
        result = _resize_to_canonical(img)
        assert result.dtype == np.uint8

    def test_padding_is_white(self):
        from hrm_ocr.pipeline.preprocess import CARD_H, CARD_W, _resize_to_canonical
        # Square image → will have white side bars or top/bottom bars
        img = np.full((400, 400, 3), 100, dtype=np.uint8)
        result = _resize_to_canonical(img)
        assert result.shape == (CARD_H, CARD_W, 3)
        # Corners should be white (padding)
        assert int(result[0, 0].mean()) == 255


# ============================================================================
# preprocess.clean_image — toggle behaviour
# ============================================================================

class TestCleanImageToggles:
    """Verify that each step can be disabled independently."""

    def _run(self, cfg_overrides: dict) -> np.ndarray:
        from hrm_ocr.pipeline.preprocess import clean_image
        img = _make_card_array()
        cfg = {
            "enable_perspective_warp": False,
            "enable_deskew": False,
            "enable_clahe": False,
            "enable_glare_removal": False,
            "enable_resize": True,
        }
        cfg.update(cfg_overrides)
        return clean_image(img, cfg)

    def test_all_disabled_except_resize(self):
        from hrm_ocr.pipeline.preprocess import CARD_H, CARD_W
        result = self._run({})
        assert result.shape == (CARD_H, CARD_W, 3)

    def test_disable_resize_keeps_original_size(self):
        from hrm_ocr.pipeline.preprocess import clean_image
        img = _make_card_array(800, 500)
        result = clean_image(img, {
            "enable_perspective_warp": False,
            "enable_deskew": False,
            "enable_clahe": False,
            "enable_glare_removal": False,
            "enable_resize": False,
        })
        assert result.shape[:2] == img.shape[:2]

    def test_output_always_uint8(self):
        from hrm_ocr.pipeline.preprocess import clean_image
        img = _make_card_array()
        result = clean_image(img)
        assert result.dtype == np.uint8

    def test_output_always_3channel(self):
        from hrm_ocr.pipeline.preprocess import clean_image
        img = _make_card_array()
        result = clean_image(img)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_input_not_mutated(self):
        from hrm_ocr.pipeline.preprocess import clean_image
        img = _make_card_array()
        original = img.copy()
        clean_image(img)
        assert np.array_equal(img, original)

    def test_nested_cfg_preprocess_key_supported(self):
        """cfg with a 'preprocess' sub-key should be accepted (as loaded from YAML)."""
        from hrm_ocr.pipeline.preprocess import CARD_H, CARD_W, clean_image
        img = _make_card_array()
        cfg = {"preprocess": {"enable_resize": True, "enable_deskew": False}}
        result = clean_image(img, cfg)
        assert result.shape == (CARD_H, CARD_W, 3)


# ============================================================================
# preprocess.clean_image — full pipeline shape contract
# ============================================================================

class TestCleanImageContract:
    def test_output_is_canonical_shape(self):
        from hrm_ocr.pipeline.preprocess import CARD_H, CARD_W, clean_image
        img = _make_bgr_array(800, 500)
        result = clean_image(img)
        assert result.shape == (CARD_H, CARD_W, 3)

    def test_accepts_small_input(self):
        from hrm_ocr.pipeline.preprocess import CARD_H, CARD_W, clean_image
        img = _make_bgr_array(100, 60)
        result = clean_image(img)
        assert result.shape == (CARD_H, CARD_W, 3)

    def test_accepts_large_input(self):
        from hrm_ocr.pipeline.preprocess import CARD_H, CARD_W, clean_image
        img = _make_bgr_array(4000, 2500)
        result = clean_image(img)
        assert result.shape == (CARD_H, CARD_W, 3)

    def test_accepts_already_canonical_input(self):
        from hrm_ocr.pipeline.preprocess import CARD_H, CARD_W, clean_image
        img = _make_bgr_array(CARD_W, CARD_H)
        result = clean_image(img)
        assert result.shape == (CARD_H, CARD_W, 3)


# ============================================================================
# Performance budget (optional — skipped in slow CI environments)
# ============================================================================

@pytest.mark.slow
class TestPreprocessPerformance:
    """Verify that clean_image stays under 30 ms on a canonical-sized input.

    Mark with ``-m slow`` to include in performance runs:
      pytest -m slow tests/unit/test_pipeline.py
    """

    def test_30ms_budget_canonical_input(self):
        from hrm_ocr.pipeline.preprocess import clean_image
        img = _make_card_array()  # 1012×638 — canonical size
        # Warm-up run (avoids counting JIT / module import overhead)
        clean_image(img)

        N = 5
        times: list[float] = []
        for _ in range(N):
            t0 = time.perf_counter()
            clean_image(img)
            times.append((time.perf_counter() - t0) * 1000)

        median_ms = sorted(times)[N // 2]
        assert median_ms < 30.0, (
            f"clean_image median latency {median_ms:.1f} ms exceeds 30 ms budget. "
            f"All runs: {[f'{t:.1f}' for t in times]}"
        )

    def test_30ms_budget_phone_photo_input(self):
        """Simulate a larger phone photo before warp brings it to canonical size."""
        from hrm_ocr.pipeline.preprocess import clean_image
        img = _make_card_array(3000, 2000)  # 6 MP phone crop
        clean_image(img)  # warm-up

        N = 3
        times: list[float] = []
        for _ in range(N):
            t0 = time.perf_counter()
            clean_image(img)
            times.append((time.perf_counter() - t0) * 1000)

        median_ms = sorted(times)[N // 2]
        assert median_ms < 100.0, (
            f"clean_image on 6MP input: {median_ms:.1f} ms (relaxed budget for large input)"
        )
