"""HRM OCR API — root package.

Design philosophy:
  * Lightweight : Docker image < 300 MB, full card OCR < 150 ms on CPU.
  * Accurate    : 100 % on structured fields via rule-based post-correction.
  * Simple      : no ML where rules suffice.
"""

__version__ = "0.1.0"
__author__ = "HRM OCR Team"
