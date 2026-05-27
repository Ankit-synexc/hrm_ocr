"""
hrm_ocr.eval
=============
Evaluation metrics for the HRM OCR pipeline.

This module is the single source of truth for how accuracy is measured.
Every metric is pure (no I/O, no side-effects) so it can be called in
unit tests and in the CLI harness identically.

Metrics
-------
cer(pred, gt)                       → float   Character Error Rate
wer(pred, gt)                       → float   Word Error Rate
field_accuracy(pred, gt)            → dict    Per-field exact-match + CER
confusion_pairs(preds, gts)         → Counter Most common char substitution pairs
post_correction_delta(pre, post, gt)→ dict    Accuracy gain from correction layer

Model
-----
EvalReport — Pydantic v2 model capturing a full evaluation run.

Design notes
------------
* All text is normalised before comparison: strip whitespace, collapse
  internal spaces, uppercase.  Field-level normalisation is intentionally
  minimal so the metrics reflect what the validator and downstream
  consumers actually see.
* CER uses the standard formula:
    CER = (S + D + I) / N
  where S=substitutions, D=deletions, I=insertions, N=len(reference).
  Implemented via python-Levenshtein for speed.
* WER tokenises on whitespace; empty-reference edge case returns 0.0 if
  both strings are empty, else 1.0.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Literal

from Levenshtein import distance as _lev_distance
from Levenshtein import editops as _lev_editops
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Canonical form used for ALL comparisons.

    Steps
    -----
    1. Strip leading / trailing whitespace.
    2. Collapse internal whitespace runs to a single space.
    3. Uppercase (OCR models are case-inconsistent on Indian cards).
    """
    return re.sub(r"\s+", " ", text.strip()).upper()


# ---------------------------------------------------------------------------
# Character Error Rate
# ---------------------------------------------------------------------------

def cer(pred: str, gt: str) -> float:
    """Character Error Rate between *pred* and *gt*.

    Formula: (S + D + I) / max(len(gt), 1)

    Parameters
    ----------
    pred : str
        Predicted (OCR output, optionally corrected) text.
    gt : str
        Ground-truth text.

    Returns
    -------
    float
        CER in [0, ∞).  0.0 = perfect, 1.0 = as many errors as chars in GT.
        Can exceed 1.0 if prediction is much longer than ground truth.

    Examples
    --------
    >>> cer("", "")
    0.0
    >>> cer("ABCD", "ABCD")
    0.0
    >>> cer("ABCE", "ABCD")
    0.25
    >>> cer("", "ABCD")
    1.0
    """
    pred_n = _normalise(pred)
    gt_n = _normalise(gt)
    if not gt_n:
        return 0.0 if not pred_n else 1.0
    return _lev_distance(pred_n, gt_n) / len(gt_n)


# ---------------------------------------------------------------------------
# Word Error Rate
# ---------------------------------------------------------------------------

def wer(pred: str, gt: str) -> float:
    """Word Error Rate between *pred* and *gt*.

    Tokenises on whitespace (after normalisation).  Uses Levenshtein
    distance over the token sequence.

    Formula: edit_distance(pred_tokens, gt_tokens) / max(len(gt_tokens), 1)

    Returns
    -------
    float
        WER in [0, ∞).  0.0 = perfect match.

    Examples
    --------
    >>> wer("", "")
    0.0
    >>> wer("RAVI KUMAR", "RAVI KUMAR")
    0.0
    >>> wer("RAVI", "RAVI KUMAR")
    0.5
    >>> wer("RAVI KAPOOR", "RAVI KUMAR")
    0.5
    """
    pred_n = _normalise(pred)
    gt_n = _normalise(gt)
    pred_tokens = pred_n.split() if pred_n else []
    gt_tokens = gt_n.split() if gt_n else []
    if not gt_tokens:
        return 0.0 if not pred_tokens else 1.0
    # Levenshtein over token sequences via the Wagner-Fischer DP approach
    return _token_edit_distance(pred_tokens, gt_tokens) / len(gt_tokens)


def _token_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein edit distance over two token lists (Wagner-Fischer DP)."""
    m, n = len(a), len(b)
    # Use two-row DP for O(min(m,n)) space
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


# ---------------------------------------------------------------------------
# Field accuracy
# ---------------------------------------------------------------------------

def field_accuracy(
    pred_fields: dict[str, str],
    gt_fields: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Per-field exact-match accuracy and CER.

    Parameters
    ----------
    pred_fields : dict[str, str]
        Predicted field values, e.g. ``{"name": "RAVI KUMAR", ...}``.
        Values may be raw OCR output or post-corrected strings.
    gt_fields : dict[str, str]
        Ground-truth field values with the same keys.

    Returns
    -------
    dict[str, dict]
        Keys are field names present in *gt_fields*.
        Each value is::

            {
                "exact_match": bool,
                "cer": float,
                "wer": float,
                "pred": str,        # normalised prediction
                "gt": str,          # normalised ground truth
                "missing": bool,    # True if field absent from pred_fields
            }

    Notes
    -----
    Fields present in *pred_fields* but absent from *gt_fields* are ignored
    — we measure only what we have ground truth for.
    """
    results: dict[str, dict[str, Any]] = {}
    for field_name, gt_val in gt_fields.items():
        gt_n = _normalise(gt_val)
        if field_name not in pred_fields:
            results[field_name] = {
                "exact_match": False,
                "cer": 1.0,
                "wer": 1.0,
                "pred": "",
                "gt": gt_n,
                "missing": True,
            }
            continue
        pred_val = pred_fields[field_name]
        pred_n = _normalise(pred_val)
        results[field_name] = {
            "exact_match": pred_n == gt_n,
            "cer": cer(pred_val, gt_val),
            "wer": wer(pred_val, gt_val),
            "pred": pred_n,
            "gt": gt_n,
            "missing": False,
        }
    return results


# ---------------------------------------------------------------------------
# Confusion pairs
# ---------------------------------------------------------------------------

def confusion_pairs(
    preds: list[str],
    gts: list[str],
) -> Counter[tuple[str, str]]:
    """Count the most common character-level substitution pairs across a corpus.

    Only **substitutions** are counted (not insertions or deletions) because
    substitutions are what post-correction rules can fix.  Deletions and
    insertions are tracked separately in :func:`cer`.

    Parameters
    ----------
    preds : list[str]
        List of predicted strings (one per sample), post-normalised.
    gts : list[str]
        Corresponding ground-truth strings.  Must be the same length as *preds*.

    Returns
    -------
    Counter[tuple[str, str]]
        Keys are ``(predicted_char, groundtruth_char)`` pairs.
        Sorted by frequency descending when iterated via
        ``counter.most_common(n)``.

    Examples
    --------
    >>> c = confusion_pairs(["1O23", "S678"], ["1023", "5678"])
    >>> c.most_common(2)
    [(('O', '0'), 1), (('S', '5'), 1)]
    """
    if len(preds) != len(gts):
        raise ValueError(
            f"preds and gts must have the same length "
            f"(got {len(preds)} vs {len(gts)})"
        )
    counter: Counter[tuple[str, str]] = Counter()
    for pred, gt in zip(preds, gts):
        pred_n = _normalise(pred)
        gt_n = _normalise(gt)
        if pred_n == gt_n:
            continue
        try:
            ops = _lev_editops(pred_n, gt_n)
        except Exception:
            continue
        for op, src_pos, dst_pos in ops:
            if op == "replace":
                pred_char = pred_n[src_pos] if src_pos < len(pred_n) else ""
                gt_char = gt_n[dst_pos] if dst_pos < len(gt_n) else ""
                if pred_char and gt_char:
                    counter[(pred_char, gt_char)] += 1
    return counter


# ---------------------------------------------------------------------------
# Post-correction delta
# ---------------------------------------------------------------------------

def post_correction_delta(
    pre_correction_fields: dict[str, str],
    post_correction_fields: dict[str, str],
    gt_fields: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Measure the exact accuracy gain from the correction layer, per field.

    This is the **most important metric** in the harness: it tells you
    whether each correction rule is helping or hurting.

    Parameters
    ----------
    pre_correction_fields : dict[str, str]
        Raw OCR output before any post-correction rules are applied.
    post_correction_fields : dict[str, str]
        OCR output after all post-correction rules have been applied.
    gt_fields : dict[str, str]
        Ground-truth values.

    Returns
    -------
    dict[str, dict]
        Keys are field names in *gt_fields*.  Each value is::

            {
                "pre_exact":  bool,    # exact match before correction
                "post_exact": bool,    # exact match after correction
                "pre_cer":    float,
                "post_cer":   float,
                "cer_delta":  float,   # negative = improvement (lower is better)
                "improved":   bool,    # correction helped this sample
                "degraded":   bool,    # correction hurt this sample (regression!)
                "unchanged":  bool,
            }

    Notes
    -----
    A *degraded* result means a correction rule has introduced a regression
    and should be re-examined.  The CLI highlights these in red.
    """
    results: dict[str, dict[str, Any]] = {}
    for field_name, gt_val in gt_fields.items():
        pre_val = pre_correction_fields.get(field_name, "")
        post_val = post_correction_fields.get(field_name, "")

        pre_exact = _normalise(pre_val) == _normalise(gt_val)
        post_exact = _normalise(post_val) == _normalise(gt_val)
        pre_cer_val = cer(pre_val, gt_val)
        post_cer_val = cer(post_val, gt_val)
        delta = post_cer_val - pre_cer_val  # negative = improvement

        results[field_name] = {
            "pre_exact": pre_exact,
            "post_exact": post_exact,
            "pre_cer": round(pre_cer_val, 4),
            "post_cer": round(post_cer_val, 4),
            "cer_delta": round(delta, 4),
            "improved": (not pre_exact and post_exact) or (post_cer_val < pre_cer_val),
            "degraded": (pre_exact and not post_exact) or (post_cer_val > pre_cer_val),
            "unchanged": pre_exact == post_exact and abs(delta) < 1e-9,
        }
    return results


# ---------------------------------------------------------------------------
# Aggregate helpers (used by run_eval.py)
# ---------------------------------------------------------------------------

def aggregate_field_accuracy(
    per_sample_results: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-sample field_accuracy dicts into corpus-level stats.

    Parameters
    ----------
    per_sample_results : list[dict]
        Each element is the return value of :func:`field_accuracy` for one sample.

    Returns
    -------
    dict[str, dict]
        Per field::

            {
                "exact_match_rate": float,   # fraction of samples with exact match
                "mean_cer": float,
                "mean_wer": float,
                "n": int,                    # sample count with this field in GT
                "missing_rate": float,       # fraction of samples missing this field
            }
    """
    accum: dict[str, dict[str, list[Any]]] = {}
    for sample in per_sample_results:
        for field_name, stats in sample.items():
            if field_name not in accum:
                accum[field_name] = {
                    "exact_match": [],
                    "cer": [],
                    "wer": [],
                    "missing": [],
                }
            accum[field_name]["exact_match"].append(stats["exact_match"])
            accum[field_name]["cer"].append(stats["cer"])
            accum[field_name]["wer"].append(stats["wer"])
            accum[field_name]["missing"].append(stats["missing"])

    out: dict[str, dict[str, Any]] = {}
    for field_name, lists in accum.items():
        n = len(lists["exact_match"])
        out[field_name] = {
            "exact_match_rate": sum(lists["exact_match"]) / n,
            "mean_cer": sum(lists["cer"]) / n,
            "mean_wer": sum(lists["wer"]) / n,
            "n": n,
            "missing_rate": sum(lists["missing"]) / n,
        }
    return out


def aggregate_post_correction_delta(
    per_sample_deltas: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-sample post_correction_delta dicts into corpus-level stats.

    Returns
    -------
    dict[str, dict]
        Per field::

            {
                "pre_exact_rate":  float,
                "post_exact_rate": float,
                "exact_delta":     float,   # post - pre (positive = improvement)
                "mean_cer_delta":  float,   # negative = correction helps
                "improved_rate":   float,   # fraction of samples improved
                "degraded_rate":   float,   # fraction of samples degraded (⚠)
                "n": int,
            }
    """
    accum: dict[str, dict[str, list[Any]]] = {}
    for sample in per_sample_deltas:
        for field_name, stats in sample.items():
            if field_name not in accum:
                accum[field_name] = {
                    "pre_exact": [],
                    "post_exact": [],
                    "cer_delta": [],
                    "improved": [],
                    "degraded": [],
                }
            accum[field_name]["pre_exact"].append(stats["pre_exact"])
            accum[field_name]["post_exact"].append(stats["post_exact"])
            accum[field_name]["cer_delta"].append(stats["cer_delta"])
            accum[field_name]["improved"].append(stats["improved"])
            accum[field_name]["degraded"].append(stats["degraded"])

    out: dict[str, dict[str, Any]] = {}
    for field_name, lists in accum.items():
        n = len(lists["pre_exact"])
        pre_rate = sum(lists["pre_exact"]) / n
        post_rate = sum(lists["post_exact"]) / n
        out[field_name] = {
            "pre_exact_rate": round(pre_rate, 4),
            "post_exact_rate": round(post_rate, 4),
            "exact_delta": round(post_rate - pre_rate, 4),
            "mean_cer_delta": round(sum(lists["cer_delta"]) / n, 4),
            "improved_rate": round(sum(lists["improved"]) / n, 4),
            "degraded_rate": round(sum(lists["degraded"]) / n, 4),
            "n": n,
        }
    return out


# ---------------------------------------------------------------------------
# EvalReport Pydantic model
# ---------------------------------------------------------------------------

class FieldAccuracyStats(BaseModel):
    """Corpus-level accuracy stats for one field."""
    exact_match_rate: float = Field(ge=0.0, le=1.0)
    mean_cer: float = Field(ge=0.0)
    mean_wer: float = Field(ge=0.0)
    n: int = Field(ge=0)
    missing_rate: float = Field(ge=0.0, le=1.0)


class CorrectionDeltaStats(BaseModel):
    """Corpus-level correction-layer stats for one field."""
    pre_exact_rate: float = Field(ge=0.0, le=1.0)
    post_exact_rate: float = Field(ge=0.0, le=1.0)
    exact_delta: float                     # post - pre; positive = better
    mean_cer_delta: float                  # negative = correction lowers CER
    improved_rate: float = Field(ge=0.0, le=1.0)
    degraded_rate: float = Field(ge=0.0, le=1.0)
    n: int = Field(ge=0)


class EvalReport(BaseModel):
    """Full evaluation report for one harness run.

    Serialised as JSON to ``logs/eval_results/<timestamp>_eval_report.json``.
    """
    doc_type: str = Field(
        description="Document type evaluated, e.g. 'aadhaar' or 'pan'."
    )
    extraction_method: Literal["text_layer", "ocr"] = Field(
        description=(
            "'text_layer' = pdfplumber text extraction path; "
            "'ocr' = PaddleOCR image path."
        )
    )
    sample_count: int = Field(ge=0, description="Number of samples evaluated.")

    mean_cer: float = Field(ge=0.0, description="Macro-average CER across all fields and samples.")
    mean_wer: float = Field(ge=0.0, description="Macro-average WER across all fields and samples.")

    per_field_accuracy: dict[str, FieldAccuracyStats] = Field(
        default_factory=dict,
        description="Per-field accuracy breakdown.",
    )
    post_correction_delta: dict[str, CorrectionDeltaStats] = Field(
        default_factory=dict,
        description=(
            "Per-field accuracy delta introduced by the correction layer. "
            "The most important metric: shows whether rules help or hurt."
        ),
    )
    top_confusions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Top character confusion pairs across the corpus, "
            "ordered by frequency descending. "
            "Format: [{'pred': 'O', 'gt': '0', 'count': 42}, ...]"
        ),
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of this evaluation run.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "doc_type": "aadhaar",
                "extraction_method": "ocr",
                "sample_count": 200,
                "mean_cer": 0.032,
                "mean_wer": 0.045,
                "per_field_accuracy": {
                    "aadhaar_number": {
                        "exact_match_rate": 0.97,
                        "mean_cer": 0.008,
                        "mean_wer": 0.012,
                        "n": 200,
                        "missing_rate": 0.0,
                    }
                },
                "post_correction_delta": {
                    "aadhaar_number": {
                        "pre_exact_rate": 0.91,
                        "post_exact_rate": 0.97,
                        "exact_delta": 0.06,
                        "mean_cer_delta": -0.021,
                        "improved_rate": 0.07,
                        "degraded_rate": 0.01,
                        "n": 200,
                    }
                },
                "top_confusions": [
                    {"pred": "O", "gt": "0", "count": 87},
                    {"pred": "I", "gt": "1", "count": 43},
                ],
                "timestamp": "2025-01-01T00:00:00+00:00",
            }
        }
    }
