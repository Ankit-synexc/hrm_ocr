#!/usr/bin/env python3
"""
scripts/run_eval.py
====================
Evaluation harness CLI — the single tool used to prove accuracy targets.

Run this after every pipeline change to see whether things got better or worse.

Usage examples
--------------
# Evaluate Aadhaar OCR extractions against ground truth:
  python scripts/run_eval.py \\
      --predictions  data/annotations/pred_aadhaar.jsonl \\
      --ground_truth data/annotations/gt_aadhaar.jsonl \\
      --doc_type     aadhaar \\
      --extraction_method ocr \\
      --output_dir   logs/eval_results

# Evaluate PAN text-layer extractions (digital PDFs):
  python scripts/run_eval.py \\
      --predictions  data/annotations/pred_pan.jsonl \\
      --ground_truth data/annotations/gt_pan.jsonl \\
      --doc_type     pan \\
      --extraction_method text_layer \\
      --output_dir   logs/eval_results

Input JSONL format
------------------
Each line in --predictions must be a JSON object with:
  {
    "id": "sample_001",
    "fields": {                    # post-correction predicted values
      "name":           "RAVI KUMAR",
      "aadhaar_number": "1234 5678 9012"
    },
    "raw_fields": {                # pre-correction OCR output (optional)
      "name":           "RAVI KUMA R",
      "aadhaar_number": "1234 5678 9O12"
    }
  }

Each line in --ground_truth must be a JSON object with:
  {
    "id": "sample_001",
    "fields": {
      "name":           "RAVI KUMAR",
      "aadhaar_number": "1234 5678 9012"
    }
  }

Output
------
Prints a formatted table to stdout, then saves:
  <output_dir>/<YYYYMMDD_HHMMSS>_<doc_type>_eval_report.json
  <output_dir>/<YYYYMMDD_HHMMSS>_<doc_type>_eval_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src is on path when running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hrm_ocr.eval import (
    EvalReport,
    FieldAccuracyStats,
    CorrectionDeltaStats,
    aggregate_field_accuracy,
    aggregate_post_correction_delta,
    cer,
    confusion_pairs,
    field_accuracy,
    post_correction_delta,
    wer,
)

# ---------------------------------------------------------------------------
# ANSI colour codes (gracefully degraded on Windows without VT mode)
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"


def _c(text: str, code: str) -> str:
    """Wrap text in ANSI code if stdout is a TTY."""
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Load all non-empty lines from a JSONL file."""
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: Skipping malformed JSON on line {lineno} of {path}: {exc}",
                    file=sys.stderr,
                )
    return records


def _index_by_id(records: list[dict]) -> dict[str, dict]:
    """Return a dict keyed by record['id']."""
    indexed: dict[str, dict] = {}
    for rec in records:
        rec_id = rec.get("id")
        if rec_id is None:
            print("WARNING: Record missing 'id' field — skipped.", file=sys.stderr)
            continue
        if rec_id in indexed:
            print(f"WARNING: Duplicate id '{rec_id}' — last occurrence used.", file=sys.stderr)
        indexed[rec_id] = rec
    return indexed


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _col_width(values: list[str], header: str, min_w: int = 6) -> int:
    return max(min_w, len(header), max((len(v) for v in values), default=0))


def _render_accuracy_table(
    per_field: dict[str, dict],
    title: str = "Per-Field Accuracy",
) -> str:
    """Render per-field accuracy as an aligned plain-text table."""
    if not per_field:
        return f"  (no data for {title})\n"

    fields = sorted(per_field.keys())
    col_field  = _col_width(fields, "Field", 20)
    col_n      = _col_width([str(per_field[f]["n"]) for f in fields], "N", 5)
    col_exact  = 8
    col_cer    = 8
    col_wer    = 8
    col_miss   = 8

    header = (
        f"{'Field':<{col_field}}  "
        f"{'N':>{col_n}}  "
        f"{'Exact%':>{col_exact}}  "
        f"{'CER':>{col_cer}}  "
        f"{'WER':>{col_wer}}  "
        f"{'Miss%':>{col_miss}}"
    )
    sep = "-" * len(header)
    lines = [f"\n{_c(title, _BOLD)}", sep, header, sep]

    for field in fields:
        s = per_field[field]
        exact_pct = s["exact_match_rate"] * 100
        cer_val   = s["mean_cer"]
        wer_val   = s["mean_wer"]
        miss_pct  = s["missing_rate"] * 100

        # Colour exact match rate
        if exact_pct >= 97:
            exact_str = _c(f"{exact_pct:>7.1f}%", _GREEN)
        elif exact_pct >= 85:
            exact_str = _c(f"{exact_pct:>7.1f}%", _YELLOW)
        else:
            exact_str = _c(f"{exact_pct:>7.1f}%", _RED)

        lines.append(
            f"{field:<{col_field}}  "
            f"{s['n']:>{col_n}}  "
            f"{exact_str}  "
            f"{cer_val:>{col_cer}.4f}  "
            f"{wer_val:>{col_wer}.4f}  "
            f"{miss_pct:>{col_miss}.1f}%"
        )
    lines.append(sep)
    return "\n".join(lines)


def _render_delta_table(
    deltas: dict[str, dict],
    title: str = "Post-Correction Delta (pre → post)",
) -> str:
    """Render post_correction_delta as an aligned table with ▲/▼ indicators."""
    if not deltas:
        return f"  (no pre-correction data available for {title})\n"

    fields = sorted(deltas.keys())
    col_field = _col_width(fields, "Field", 20)
    col_n     = _col_width([str(deltas[f]["n"]) for f in fields], "N", 5)

    header = (
        f"{'Field':<{col_field}}  "
        f"{'N':>{col_n}}  "
        f"{'Pre%':>8}  "
        f"{'Post%':>8}  "
        f"{'Δ Exact':>8}  "
        f"{'ΔCER':>8}  "
        f"{'Impr%':>7}  "
        f"{'Degr%':>7}"
    )
    sep = "-" * len(header)
    lines = [f"\n{_c(title, _BOLD)}", sep, header, sep]

    for field in fields:
        d = deltas[field]
        pre_pct   = d["pre_exact_rate"]  * 100
        post_pct  = d["post_exact_rate"] * 100
        delta_ex  = d["exact_delta"] * 100    # percentage points
        delta_cer = d["mean_cer_delta"]
        impr_pct  = d["improved_rate"] * 100
        degr_pct  = d["degraded_rate"] * 100

        # Delta colouring
        if delta_ex > 0.5:
            delta_str = _c(f"▲{delta_ex:+.1f}%", _GREEN)
        elif delta_ex < -0.5:
            delta_str = _c(f"▼{delta_ex:+.1f}%", _RED)
        else:
            delta_str = _c(f" {delta_ex:+.1f}%", _DIM)

        cer_str = (
            _c(f"{delta_cer:+.4f}", _GREEN)
            if delta_cer < -0.001
            else (_c(f"{delta_cer:+.4f}", _RED) if delta_cer > 0.001 else f"{delta_cer:+.4f}")
        )
        degr_str = _c(f"{degr_pct:.1f}%", _RED) if degr_pct > 1.0 else f"{degr_pct:.1f}%"

        lines.append(
            f"{field:<{col_field}}  "
            f"{d['n']:>{col_n}}  "
            f"{pre_pct:>7.1f}%  "
            f"{post_pct:>7.1f}%  "
            f"{delta_str:>8}  "
            f"{cer_str:>8}  "
            f"{impr_pct:>6.1f}%  "
            f"{degr_str:>7}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _render_confusion_table(
    top_confusions: list[dict],
    n: int = 20,
    title: str = "Top Character Confusions (pred → gt)",
) -> str:
    """Render the top-N confusion pairs as a compact table."""
    if not top_confusions:
        return ""
    shown = top_confusions[:n]
    lines = [f"\n{_c(title, _BOLD)}"]
    lines.append(f"  {'Pred':>6}  {'GT':>6}  {'Count':>8}  {'Bar'}")
    lines.append("  " + "-" * 50)
    max_count = shown[0]["count"] if shown else 1
    for entry in shown:
        bar_len = int(entry["count"] / max_count * 30)
        bar = "█" * bar_len
        lines.append(
            f"  {entry['pred']:>6}  {entry['gt']:>6}  {entry['count']:>8}  {bar}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _build_markdown(report: EvalReport, doc_type: str) -> str:
    ts = report.timestamp[:19].replace("T", " ") + " UTC"
    lines = [
        f"# HRM OCR Evaluation Report — {doc_type.upper()}",
        f"",
        f"**Generated**: {ts}  ",
        f"**Samples**: {report.sample_count}  ",
        f"**Extraction method**: {report.extraction_method}  ",
        f"**Mean CER**: {report.mean_cer:.4f}  ",
        f"**Mean WER**: {report.mean_wer:.4f}  ",
        f"",
        f"## Per-Field Accuracy",
        f"",
        f"| Field | N | Exact% | Mean CER | Mean WER | Miss% |",
        f"|-------|---|--------|----------|----------|-------|",
    ]
    for field, s in sorted(report.per_field_accuracy.items()):
        lines.append(
            f"| {field} | {s.n} | {s.exact_match_rate:.1%} "
            f"| {s.mean_cer:.4f} | {s.mean_wer:.4f} | {s.missing_rate:.1%} |"
        )

    lines += [
        f"",
        f"## Post-Correction Delta",
        f"",
        f"> Positive Δ Exact = correction layer improved accuracy. "
        f"Negative Δ Exact = regression — rule needs fixing.",
        f"",
        f"| Field | N | Pre% | Post% | Δ Exact | Δ CER | Impr% | Degr% |",
        f"|-------|---|------|-------|---------|-------|-------|-------|",
    ]
    for field, d in sorted(report.post_correction_delta.items()):
        arrow = "▲" if d.exact_delta > 0 else ("▼" if d.exact_delta < 0 else "=")
        lines.append(
            f"| {field} | {d.n} | {d.pre_exact_rate:.1%} | {d.post_exact_rate:.1%} "
            f"| {arrow}{abs(d.exact_delta * 100):.1f}pp "
            f"| {d.mean_cer_delta:+.4f} | {d.improved_rate:.1%} | {d.degraded_rate:.1%} |"
        )

    if report.top_confusions:
        lines += [
            f"",
            f"## Top Character Confusions",
            f"",
            f"| Pred | GT | Count |",
            f"|------|----|-------|",
        ]
        for entry in report.top_confusions[:30]:
            lines.append(f"| `{entry['pred']}` | `{entry['gt']}` | {entry['count']} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def run_evaluation(
    predictions: list[dict],
    ground_truths: list[dict],
    doc_type: str,
    extraction_method: str,
) -> EvalReport:
    """Execute the evaluation loop and build an EvalReport.

    Parameters
    ----------
    predictions : list[dict]
        Loaded prediction JSONL records (must contain 'id', 'fields',
        optionally 'raw_fields').
    ground_truths : list[dict]
        Loaded ground-truth JSONL records (must contain 'id', 'fields').
    doc_type : str
        Document type label (e.g. 'aadhaar', 'pan').
    extraction_method : str
        'text_layer' or 'ocr'.

    Returns
    -------
    EvalReport
    """
    gt_index = _index_by_id(ground_truths)
    pred_index = _index_by_id(predictions)

    matched_ids = [i for i in gt_index if i in pred_index]
    if not matched_ids:
        print("ERROR: No matching IDs between predictions and ground truth.", file=sys.stderr)
        sys.exit(1)

    unmatched_pred = set(pred_index) - set(gt_index)
    unmatched_gt   = set(gt_index)   - set(pred_index)
    if unmatched_pred:
        print(
            f"WARNING: {len(unmatched_pred)} prediction IDs have no ground truth — ignored.",
            file=sys.stderr,
        )
    if unmatched_gt:
        print(
            f"WARNING: {len(unmatched_gt)} ground-truth IDs have no prediction — "
            f"they will count as missing fields.",
            file=sys.stderr,
        )

    # Per-sample accumulators
    per_sample_accuracy:   list[dict] = []
    per_sample_deltas:     list[dict] = []
    all_pred_texts:        list[str]  = []
    all_gt_texts:          list[str]  = []
    all_cer_vals:          list[float] = []
    all_wer_vals:          list[float] = []

    for sample_id in sorted(matched_ids):
        gt_rec   = gt_index[sample_id]
        pred_rec = pred_index[sample_id]
        gt_fields:   dict[str, str] = gt_rec.get("fields", {})
        post_fields: dict[str, str] = pred_rec.get("fields", {})
        pre_fields:  dict[str, str] = pred_rec.get("raw_fields", {})

        # Field accuracy (post-correction)
        fa = field_accuracy(post_fields, gt_fields)
        per_sample_accuracy.append(fa)

        # Post-correction delta (only if raw_fields supplied)
        if pre_fields:
            delta = post_correction_delta(pre_fields, post_fields, gt_fields)
            per_sample_deltas.append(delta)

        # Corpus-level confusion pair inputs
        for field_name, gt_val in gt_fields.items():
            pred_val = post_fields.get(field_name, "")
            all_pred_texts.append(pred_val)
            all_gt_texts.append(gt_val)
            all_cer_vals.append(cer(pred_val, gt_val))
            all_wer_vals.append(wer(pred_val, gt_val))

    # Aggregate
    agg_accuracy = aggregate_field_accuracy(per_sample_accuracy)
    agg_delta    = aggregate_post_correction_delta(per_sample_deltas) if per_sample_deltas else {}

    # Confusion pairs
    confusion_counter = confusion_pairs(all_pred_texts, all_gt_texts)
    top_conf = [
        {"pred": p, "gt": g, "count": c}
        for (p, g), c in confusion_counter.most_common(50)
    ]

    # Global mean CER / WER
    mean_cer_val = sum(all_cer_vals) / len(all_cer_vals) if all_cer_vals else 0.0
    mean_wer_val = sum(all_wer_vals) / len(all_wer_vals) if all_wer_vals else 0.0

    # Build Pydantic model
    return EvalReport(
        doc_type=doc_type,
        extraction_method=extraction_method,  # type: ignore[arg-type]
        sample_count=len(matched_ids),
        mean_cer=round(mean_cer_val, 6),
        mean_wer=round(mean_wer_val, 6),
        per_field_accuracy={
            f: FieldAccuracyStats(**v) for f, v in agg_accuracy.items()
        },
        post_correction_delta={
            f: CorrectionDeltaStats(**v) for f, v in agg_delta.items()
        },
        top_confusions=top_conf,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval",
        description=(
            "HRM OCR evaluation harness. "
            "Computes CER, WER, field accuracy, confusion pairs, "
            "and post-correction delta from JSONL prediction/ground-truth files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--predictions", required=True, type=Path,
        metavar="PATH",
        help="JSONL file with predictions. Each line: {id, fields, raw_fields?}",
    )
    p.add_argument(
        "--ground_truth", required=True, type=Path,
        metavar="PATH",
        help="JSONL file with ground truth. Each line: {id, fields}",
    )
    p.add_argument(
        "--doc_type", required=True, type=str,
        choices=["aadhaar", "pan", "cv", "other"],
        help="Document type being evaluated.",
    )
    p.add_argument(
        "--extraction_method", required=True,
        choices=["text_layer", "ocr"],
        help="'text_layer' (pdfplumber) or 'ocr' (PaddleOCR).",
    )
    p.add_argument(
        "--output_dir", default="logs/eval_results", type=Path,
        metavar="DIR",
        help="Directory to save JSON and Markdown reports (default: logs/eval_results).",
    )
    p.add_argument(
        "--top_confusions", type=int, default=20, metavar="N",
        help="Number of confusion pairs to show in the table (default: 20).",
    )
    p.add_argument(
        "--no_color", action="store_true",
        help="Disable ANSI colour output.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Disable colour if requested
    if args.no_color:
        import hrm_ocr.eval as _eval_mod
        global _GREEN, _YELLOW, _RED, _CYAN, _BOLD, _DIM, _RESET
        _GREEN = _YELLOW = _RED = _CYAN = _BOLD = _DIM = _RESET = ""

    print(f"\n{_c('HRM OCR Evaluation Harness', _BOLD)}")
    print(f"  Predictions : {args.predictions}")
    print(f"  Ground truth: {args.ground_truth}")
    print(f"  Doc type    : {args.doc_type}")
    print(f"  Method      : {args.extraction_method}")

    predictions  = _load_jsonl(args.predictions)
    ground_truths = _load_jsonl(args.ground_truth)

    print(f"\n  Loaded {len(predictions)} predictions, {len(ground_truths)} ground-truth records.")
    print("  Evaluating …\n")

    report = run_evaluation(
        predictions=predictions,
        ground_truths=ground_truths,
        doc_type=args.doc_type,
        extraction_method=args.extraction_method,
    )

    # ── Print summary header ──────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(
        f"  {_c('Samples evaluated', _BOLD)}: {report.sample_count}  │  "
        f"{_c('Mean CER', _BOLD)}: {report.mean_cer:.4f}  │  "
        f"{_c('Mean WER', _BOLD)}: {report.mean_wer:.4f}"
    )
    print(f"{'─'*60}")

    # ── Per-field accuracy table ──────────────────────────────────────────────
    acc_data = {
        f: {
            "exact_match_rate": s.exact_match_rate,
            "mean_cer": s.mean_cer,
            "mean_wer": s.mean_wer,
            "n": s.n,
            "missing_rate": s.missing_rate,
        }
        for f, s in report.per_field_accuracy.items()
    }
    print(_render_accuracy_table(acc_data, "Per-Field Accuracy (post-correction)"))

    # ── Post-correction delta table ───────────────────────────────────────────
    if report.post_correction_delta:
        delta_data = {
            f: {
                "pre_exact_rate": d.pre_exact_rate,
                "post_exact_rate": d.post_exact_rate,
                "exact_delta": d.exact_delta,
                "mean_cer_delta": d.mean_cer_delta,
                "improved_rate": d.improved_rate,
                "degraded_rate": d.degraded_rate,
                "n": d.n,
            }
            for f, d in report.post_correction_delta.items()
        }
        print(_render_delta_table(delta_data))

        # Flag regressions
        regressions = [
            f for f, d in report.post_correction_delta.items()
            if d.degraded_rate > 0.01  # >1% of samples degraded
        ]
        if regressions:
            print(
                f"\n  {_c('⚠ Correction regressions detected (>1% samples degraded):', _RED)} "
                + ", ".join(regressions)
            )
    else:
        print(
            f"\n  {_c('(No raw_fields in predictions — post-correction delta not computed)', _DIM)}"
        )

    # ── Confusion pairs table ─────────────────────────────────────────────────
    if report.top_confusions:
        print(_render_confusion_table(report.top_confusions, n=args.top_confusions))

    # ── Save reports ──────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"{ts_str}_{args.doc_type}"

    json_path = args.output_dir / f"{stem}_eval_report.json"
    md_path   = args.output_dir / f"{stem}_eval_report.md"

    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(report.model_dump_json(indent=2))

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_build_markdown(report, args.doc_type))

    print(f"\n  {_c('✓', _GREEN)} JSON report → {json_path}")
    print(f"  {_c('✓', _GREEN)} MD  report  → {md_path}\n")


if __name__ == "__main__":
    main()
