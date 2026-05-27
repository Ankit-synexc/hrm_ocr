"""
hrm_ocr.pipeline.preprocess
============================
Image preprocessing pipeline tuned for phone-photographed Indian ID cards.

Design contract
---------------
``clean_image(image, cfg) -> np.ndarray``

Every step earns its place: each is individually togglable via
``configs/default.yaml`` so you can A/B test its contribution to OCR
accuracy.  Steps that detect they are unnecessary (e.g. already-straight
image, well-lit image) **short-circuit immediately** — zero wasted CPU.

Step order (canonical, optimised for quality × speed)
------------------------------------------------------
1. **Boundary detection + perspective warp** (highest-value step; do first)
   Canny edges → largest quadrilateral contour → four-point transform.
   Fallback: full image if no card boundary found.

2. **Deskew** — Hough line transform.
   *Skipped* if |rotation| ≤ 0.5 ° (already straight).

3. **CLAHE** on L channel (LAB space), clipLimit=2.0, tileGridSize=8×8.
   *Skipped* if mean luminance is in [100, 200] (well-lit image).

4. **Glare removal** — cv2.inpaint TELEA on luminance > 240 mask.
   *Skipped* if glare covers ≤ 2 % of card area.

5. **Resize to 1012 × 638 px** (300 DPI standard ID-1 card).
   Uses ``cv2.INTER_LANCZOS4`` for highest quality downscale.

Performance budget: all 5 steps together must run under 30 ms on CPU
(measured on a modern x86-64 core at 300 DPI input).

Config flags (``preprocess:`` section of ``default.yaml``)
-----------------------------------------------------------
::

    preprocess:
      enable_perspective_warp: true
      enable_deskew:           true
      enable_clahe:            true
      enable_glare_removal:    true
      enable_resize:           true

      # Tuning knobs
      deskew_min_angle:        0.5     # degrees — skip below this
      clahe_clip_limit:        2.0
      clahe_tile_grid:         [8, 8]
      clahe_lum_low:           100     # skip CLAHE if mean lum in (low, high)
      clahe_lum_high:          200
      glare_threshold:         240     # pixel value considered glare
      glare_area_ratio:        0.02    # fraction of area — skip below this
      glare_inpaint_radius:    5       # inpaint neighbourhood radius (px)
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical output dimensions
# ---------------------------------------------------------------------------
#: ISO/IEC 7810 ID-1 card at 300 DPI: 85.6 mm × 53.98 mm
CARD_W: int = 1012
CARD_H: int = 638

# ---------------------------------------------------------------------------
# Default config values (mirror configs/default.yaml preprocess section)
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    # Feature toggles
    "enable_perspective_warp": True,
    "enable_deskew": True,
    "enable_clahe": True,
    "enable_glare_removal": True,
    "enable_resize": True,
    # Deskew
    "deskew_min_angle": 0.5,
    # CLAHE
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": [8, 8],
    "clahe_lum_low": 100,
    "clahe_lum_high": 200,
    # Glare
    "glare_threshold": 240,
    "glare_area_ratio": 0.02,
    "glare_inpaint_radius": 5,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_image(
    image: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    """Run the full preprocessing pipeline on a BGR card image.

    Parameters
    ----------
    image:
        BGR uint8 ndarray from the ingest / pdf_router layer.
        Any input resolution is accepted; the final step canonicalises to
        ``(638, 1012, 3)``.
    cfg:
        Flat or nested dict of preprocessing options.  Keys are resolved
        from the ``preprocess`` sub-section of ``default.yaml``.
        Missing keys fall back to :data:`_DEFAULTS`.

    Returns
    -------
    np.ndarray
        BGR uint8, shape ``(638, 1012, 3)``.

    Notes
    -----
    The function is **pure** (no global state) and safe to call from
    multiple threads simultaneously.
    """
    c = _merge_cfg(cfg)

    img = image.copy()  # never mutate the caller's array

    # ── Step 1: Perspective warp ──────────────────────────────────────────────
    if c["enable_perspective_warp"]:
        img = _perspective_warp(img)

    # ── Step 2: Deskew ────────────────────────────────────────────────────────
    if c["enable_deskew"]:
        img = _deskew(img, min_angle=float(c["deskew_min_angle"]))

    # ── Step 3: CLAHE ─────────────────────────────────────────────────────────
    if c["enable_clahe"]:
        img = _clahe(
            img,
            clip_limit=float(c["clahe_clip_limit"]),
            tile_grid=tuple(c["clahe_tile_grid"]),          # type: ignore[arg-type]
            lum_low=float(c["clahe_lum_low"]),
            lum_high=float(c["clahe_lum_high"]),
        )

    # ── Step 4: Glare removal ─────────────────────────────────────────────────
    if c["enable_glare_removal"]:
        img = _remove_glare(
            img,
            threshold=int(c["glare_threshold"]),
            min_area_ratio=float(c["glare_area_ratio"]),
            inpaint_radius=int(c["glare_inpaint_radius"]),
        )

    # ── Step 5: Resize to canonical dimensions ────────────────────────────────
    if c["enable_resize"]:
        img = _resize_to_canonical(img)

    return img


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _perspective_warp(img: np.ndarray) -> np.ndarray:
    """Detect card boundary and apply four-point perspective correction.

    Algorithm
    ---------
    1. Convert to grayscale → Gaussian blur → Canny edge detection.
    2. Find all external contours; keep the one closest to a quadrilateral
       (largest area, 4-corner approximation).
    3. Order corners (top-left, top-right, bottom-right, bottom-left) and
       compute an output rectangle that preserves the card's aspect ratio.
    4. Apply ``cv2.getPerspectiveTransform`` + ``cv2.warpPerspective``.

    Fallback: if no suitable quadrilateral is found (e.g. image already
    cropped to the card boundary), return *img* unchanged.

    Performance note
    ----------------
    Canny + contour finding on a ~1 MP image runs in ~3–5 ms on CPU.
    The warp itself is ~2 ms.  Total budget: ~8 ms.
    """
    # 1. Resize for faster, more robust contour detection
    orig_h, orig_w = img.shape[:2]
    work_scale = 800.0 / orig_w
    work_img = cv2.resize(img, (800, int(orig_h * work_scale)))
    
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Scharr Gradient Magnitude: This is the gold standard for document scanning.
    # It finds the physical structural boundary of the card by looking at the rate of 
    # pixel change, making it completely immune to color similarities between the card 
    # and the wooden table.
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = np.uint8(255 * grad_mag / np.max(grad_mag))
    
    # 3. Otsu Threshold + Morphological Close to connect the edges into a solid loop
    _, thresh = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 4. Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.debug("perspective_warp: no contours found — using full image")
        return img

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    card_contour = None
    
    work_area = work_img.shape[0] * work_img.shape[1]
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 4. Strict area filter: The card must be between 10% and 95% of the frame.
        # This prevents selecting the entire image border as the card.
        if 0.10 * work_area < area < 0.95 * work_area:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            # Scale the detected coordinates back up to the original high-res image
            box = box / work_scale
            card_contour = np.array(box, dtype=np.float32)
            break

    if card_contour is None:
        logger.debug(
            "perspective_warp: no suitable quadrilateral found — using full image"
        )
        return img

    pts = card_contour.reshape(4, 2).astype(np.float32)
    ordered = _order_corners(pts)
    warped = _four_point_transform(img, ordered)
    logger.debug(
        "perspective_warp: applied transform, output shape %s",
        warped.shape,
    )
    return warped


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as [TL, TR, BR, BL].

    Uses the sum/diff trick:
    * Top-left     → smallest (x+y)
    * Bottom-right → largest  (x+y)
    * Top-right    → smallest (y−x)
    * Bottom-left  → largest  (y−x)
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered[0] = pts[np.argmin(s)]   # TL
    ordered[2] = pts[np.argmax(s)]   # BR
    ordered[1] = pts[np.argmin(d)]   # TR
    ordered[3] = pts[np.argmax(d)]   # BL
    return ordered


def _four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp image so that *pts* (TL, TR, BR, BL) map to a rectangle."""
    tl, tr, br, bl = pts

    # Compute output width: max of top/bottom edge lengths
    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    out_w = max(int(w_top), int(w_bot))

    # Compute output height: max of left/right edge lengths
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))
    out_h = max(int(h_left), int(h_right))

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def _deskew(img: np.ndarray, min_angle: float = 0.5) -> np.ndarray:
    """Correct small rotations using Hough probabilistic line transform.

    Short-circuit
    -------------
    Skips the warp entirely if the detected rotation is below *min_angle*
    degrees (default 0.5 °).  This avoids an unnecessary affine warp for
    images that are already aligned — a 3–5 ms saving per card.

    Implementation notes
    --------------------
    * ``HoughLinesP`` (probabilistic) is used instead of the full Hough
      transform: it is ~3× faster and returns line segments directly,
      making angle extraction trivial.
    * Only near-horizontal lines (|angle| < 45 °) are considered to avoid
      confusing card text lines with vertical card edges.
    * The median angle over all accepted segments is used — robust to
      outliers caused by diagonal text or logos.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    h, w = img.shape[:2]

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=w // 4,   # Line must span at least 25% of card width
        maxLineGap=20,
    )
    if lines is None:
        logger.debug("deskew: no lines detected — skipping")
        return img

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        # Keep only near-horizontal lines (card edges and text baselines)
        if abs(angle) < 45.0:
            angles.append(angle)

    if not angles:
        logger.debug("deskew: no valid angles — skipping")
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < min_angle:
        logger.debug(
            "deskew: angle=%.3f° < threshold=%.1f° — skipping",
            median_angle, min_angle,
        )
        return img

    logger.debug("deskew: correcting %.3f°", median_angle)
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), median_angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
    lum_low: float = 100.0,
    lum_high: float = 200.0,
) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB colour space.

    Short-circuit
    -------------
    Skips processing if mean luminance is in the "good" range
    ``(lum_low, lum_high)`` — a well-lit, even image gains nothing from
    histogram equalisation and would only add ~2 ms of wasted work.

    Colour correctness
    ------------------
    Working in LAB space means only luminance is equalised; the a/b
    (colour) channels are untouched.  This prevents the colour shifts
    that would occur if CLAHE were applied channel-by-channel in BGR.
    """
    # Measure luminance from grayscale (fast, single channel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_lum = float(gray.mean())

    if lum_low <= mean_lum <= lum_high:
        logger.debug(
            "clahe: mean_lum=%.1f in [%.0f, %.0f] — skipping",
            mean_lum, lum_low, lum_high,
        )
        return img

    logger.debug(
        "clahe: mean_lum=%.1f, applying CLAHE (clip=%.1f, grid=%s)",
        mean_lum, clip_limit, tile_grid,
    )
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_ch = clahe_obj.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _remove_glare(
    img: np.ndarray,
    threshold: int = 240,
    min_area_ratio: float = 0.02,
    inpaint_radius: int = 5,
) -> np.ndarray:
    """Inpaint specular glare hotspots from flash or overhead lighting.

    Short-circuit
    -------------
    Skips inpainting if the glare mask covers ≤ *min_area_ratio* of the
    card area (default 2 %).  Minor glare (<2%) has negligible impact on
    OCR and inpainting wastes ~5–10 ms.

    Algorithm
    ---------
    1. Convert to LAB; threshold L channel at *threshold* to create a
       binary glare mask.
    2. Dilate the mask by 2 px to bleed slightly beyond the hotspot edge.
    3. ``cv2.inpaint`` with ``cv2.INPAINT_TELEA`` to reconstruct the
       occluded pixel values using fast marching.

    Notes
    -----
    TELEA is chosen over NS (Navier-Stokes) because it is faster (~2×)
    and produces sharper reconstructions on text regions.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]

    # Glare mask: pixels brighter than threshold in L channel
    _, mask = cv2.threshold(l_ch, threshold, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    glare_ratio = float(mask.sum()) / (mask.size * 255)
    if glare_ratio <= min_area_ratio:
        logger.debug(
            "glare_removal: glare=%.3f%% ≤ threshold=%.0f%% — skipping",
            glare_ratio * 100, min_area_ratio * 100,
        )
        return img

    logger.debug(
        "glare_removal: glare=%.2f%% — inpainting (radius=%d)",
        glare_ratio * 100, inpaint_radius,
    )
    # Dilate mask: heal slightly beyond the bright boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)


def _resize_to_canonical(img: np.ndarray) -> np.ndarray:
    """Resize to ``(CARD_H, CARD_W)`` = ``(638, 1012)`` using LANCZOS4.

    Preserves aspect ratio: if the input does not match the card's 1.585:1
    ratio the image is scaled to fit and padded with white (255, 255, 255)
    on the shorter axis.

    Uses ``cv2.INTER_LANCZOS4`` (4-lobe Lanczos resampling) for the highest
    quality downscale — noticeably sharper than AREA or CUBIC at card-like
    scale factors, with minimal aliasing near character strokes.
    """
    h, w = img.shape[:2]
    if w == CARD_W and h == CARD_H:
        return img  # Already canonical — zero-copy fast path

    scale = min(CARD_W / w, CARD_H / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Pad with white to fill canonical canvas
    canvas = np.full((CARD_H, CARD_W, 3), 255, dtype=np.uint8)
    y_off = (CARD_H - new_h) // 2
    x_off = (CARD_W - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    logger.debug(
        "resize: (%d×%d) → (%d×%d) → padded to (%d×%d)",
        w, h, new_w, new_h, CARD_W, CARD_H,
    )
    return canvas


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _merge_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Merge caller-supplied config with defaults.

    Supports both flat dicts and nested dicts with a ``preprocess`` key
    (as loaded directly from ``default.yaml``).
    """
    base = dict(_DEFAULTS)
    if cfg is None:
        return base
    # If caller passed the full YAML dict, drill into the preprocess section
    source = cfg.get("preprocess", cfg)
    base.update({k: v for k, v in source.items() if k in base})
    return base
