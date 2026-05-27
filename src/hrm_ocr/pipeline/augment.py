"""
hrm_ocr.pipeline.augment
=========================
Data augmentation pipeline for generating realistic training data.

Applies exclusively to the image path to simulate real-world phone photography
degradations (glare, shadows, perspective skew, noise, folds).

These functions receive and return BGR uint8 NumPy arrays.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class AugmentationRecord:
    image: np.ndarray
    augmentations_applied: list[str] = field(default_factory=list)
    severity_params: dict[str, Any] = field(default_factory=dict)


def perspective_distort(img: np.ndarray) -> np.ndarray:
    """Apply a random four-corner perspective shift (max 5%)."""
    h, w = img.shape[:2]
    max_shift = int(min(h, w) * 0.05)
    
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [random.randint(0, max_shift), random.randint(0, max_shift)],
        [w - random.randint(0, max_shift), random.randint(0, max_shift)],
        [w - random.randint(0, max_shift), h - random.randint(0, max_shift)],
        [random.randint(0, max_shift), h - random.randint(0, max_shift)],
    ])
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def motion_blur(img: np.ndarray) -> np.ndarray:
    """Apply directional motion blur (3-7px kernel)."""
    size = random.randint(3, 7)
    kernel = np.zeros((size, size), dtype=np.float32)
    
    # Random direction: horizontal, vertical, or diagonal
    direction = random.choice(["horizontal", "vertical", "diagonal1", "diagonal2"])
    if direction == "horizontal":
        kernel[size // 2, :] = 1.0 / size
    elif direction == "vertical":
        kernel[:, size // 2] = 1.0 / size
    elif direction == "diagonal1":
        for i in range(size):
            kernel[i, i] = 1.0 / size
    else:
        for i in range(size):
            kernel[i, size - 1 - i] = 1.0 / size
            
    return cv2.filter2D(img, -1, kernel)


def jpeg_compress(img: np.ndarray) -> np.ndarray:
    """Simulate severe JPEG compression artifacts (quality 40-75)."""
    quality = random.randint(40, 75)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encimg, cv2.IMREAD_COLOR)


def add_gaussian_noise(img: np.ndarray) -> np.ndarray:
    """Add Gaussian noise with sigma 5-20."""
    sigma = random.uniform(5, 20)
    gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    return noisy


def brightness_contrast_shift(img: np.ndarray) -> np.ndarray:
    """Shift brightness by ±30 and contrast by 0.8-1.2."""
    brightness = random.randint(-30, 30)
    contrast = random.uniform(0.8, 1.2)
    
    # formula: img * contrast + brightness
    shifted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    return shifted


def shadow_overlay(img: np.ndarray) -> np.ndarray:
    """Overlay 1-3 dark polygons to simulate shadows."""
    out = img.copy()
    h, w = out.shape[:2]
    num_shadows = random.randint(1, 3)
    
    for _ in range(num_shadows):
        # Create a random polygon
        pts = np.array([[
            [random.randint(0, w), random.randint(0, h)],
            [random.randint(0, w), random.randint(0, h)],
            [random.randint(0, w), random.randint(0, h)],
            [random.randint(0, w), random.randint(0, h)]
        ]], dtype=np.int32)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, pts, 255)
        
        # Darken the shadowed area
        shadow_intensity = random.uniform(0.4, 0.7)
        shadowed = cv2.convertScaleAbs(out, alpha=shadow_intensity, beta=0)
        
        # Apply mask
        np.copyto(out, shadowed, where=(mask[..., None] == 255))
        
    return out


def glare_patch(img: np.ndarray) -> np.ndarray:
    """Add 1-2 elliptical bright patches to simulate flash glare."""
    out = img.copy().astype(np.float32)
    h, w = out.shape[:2]
    num_glares = random.randint(1, 2)
    
    for _ in range(num_glares):
        center = (random.randint(0, w), random.randint(0, h))
        axes = (random.randint(50, 200), random.randint(20, 100))
        angle = random.randint(0, 180)
        
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
        
        # Blur the mask to create a soft glare edge
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        # Add glare
        glare_intensity = random.uniform(100, 200)
        out += mask[..., None] * glare_intensity
        
    return np.clip(out, 0, 255).astype(np.uint8)


def fold_lines(img: np.ndarray) -> np.ndarray:
    """Add 1-2 brightness bands to simulate folded paper."""
    out = img.copy().astype(np.float32)
    h, w = out.shape[:2]
    num_folds = random.randint(1, 2)
    
    for _ in range(num_folds):
        is_horizontal = random.choice([True, False])
        if is_horizontal:
            y = random.randint(int(h * 0.1), int(h * 0.9))
            fold_w = random.randint(10, 30)
            
            # Create a 1D gradient for the fold
            grad = np.linspace(-1, 1, fold_w)
            intensity = 50 * np.exp(-4 * grad**2)  # Brightness bump
            
            y_start = max(0, y - fold_w // 2)
            y_end = min(h, y + fold_w // 2)
            actual_w = y_end - y_start
            
            if actual_w > 0:
                intensity = intensity[:actual_w]
                out[y_start:y_end, :] += intensity[:, None, None]
        else:
            x = random.randint(int(w * 0.1), int(w * 0.9))
            fold_w = random.randint(10, 30)
            
            grad = np.linspace(-1, 1, fold_w)
            intensity = 50 * np.exp(-4 * grad**2)
            
            x_start = max(0, x - fold_w // 2)
            x_end = min(w, x + fold_w // 2)
            actual_w = x_end - x_start
            
            if actual_w > 0:
                intensity = intensity[:actual_w]
                out[:, x_start:x_end] += intensity[None, :, None]
                
    return np.clip(out, 0, 255).astype(np.uint8)


def partial_occlusion(img: np.ndarray) -> np.ndarray:
    """Overlay 1-3 small black rectangles to simulate occlusions (e.g., fingers)."""
    out = img.copy()
    h, w = out.shape[:2]
    num_occlusions = random.randint(1, 3)
    
    for _ in range(num_occlusions):
        occ_w = random.randint(20, 80)
        occ_h = random.randint(20, 80)
        x = random.randint(0, w - occ_w)
        y = random.randint(0, h - occ_h)
        
        cv2.rectangle(out, (x, y), (x + occ_w, y + occ_h), (0, 0, 0), -1)
        
    return out


# Mapping of function names to their callable implementation
AUGMENTATION_FUNCS = {
    "perspective_distort": perspective_distort,
    "motion_blur": motion_blur,
    "jpeg_compress": jpeg_compress,
    "add_gaussian_noise": add_gaussian_noise,
    "brightness_contrast_shift": brightness_contrast_shift,
    "shadow_overlay": shadow_overlay,
    "glare_patch": glare_patch,
    "fold_lines": fold_lines,
    "partial_occlusion": partial_occlusion,
}


def augment_sample(
    image: np.ndarray,
    n_variants: int,
    augmentation_probability: float = 0.5
) -> list[AugmentationRecord]:
    """Generate multiple augmented variants from a single source image."""
    variants = []
    
    for _ in range(n_variants):
        aug_img = image.copy()
        applied = []
        
        for name, func in AUGMENTATION_FUNCS.items():
            if random.random() < augmentation_probability:
                aug_img = func(aug_img)
                applied.append(name)
                
        variants.append(AugmentationRecord(
            image=aug_img,
            augmentations_applied=applied,
            severity_params={"p": augmentation_probability}
        ))
        
    return variants
