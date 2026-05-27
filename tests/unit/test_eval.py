"""
tests/unit/test_eval.py
========================
Unit tests for hrm_ocr.eval — every metric function is tested in isolation
with controlled inputs so results are fully deterministic and reviewable.

Test organisation
-----------------
TestNormalise               — internal normalisation helper
TestCER                     — character error rate
TestWER                     — word error rate, incl. token edit distance
TestFieldAccuracy           — per-field exact-match + CER/WER
TestConfusionPairs          — substitution-pair counting
TestPostCorrectionDelta     — pre/post correction delta per field
TestAggregateFieldAccuracy  — corpus-level aggregation
TestAggregatePostCorrDelta  — corpus-level delta aggregation
TestEvalReport              — Pydantic model validation + serialisation
TestRunEvalIntegration      — run_evaluation() end-to-end with test fixtures
TestCLI                     — argparse + output-file smoke test (no OCR)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fa(pred_fields: dict, gt_fields: dict) -> dict:
    """Shortcut: call field_accuracy and return result."""
    from hrm_ocr.eval import field_accuracy
    return field_accuracy(pred_fields, gt_fields)


def _delta(pre: dict, post: dict, gt: dict) -> dict:
    from hrm_ocr.eval import post_correction_delta
    return post_correction_delta(pre, post, gt)


# ============================================================================
# _normalise (internal — accessed via known outputs)
# ============================================================================

class TestNormalise:
    """Test normalisation behaviour through the public functions."""

    def test_cer_strips_and_uppercases(self):
        from hrm_ocr.eval import cer
        # "ravi" should match "RAVI" after normalise
        assert cer("ravi", "RAVI") == 0.0

    def test_cer_collapses_internal_spaces(self):
        from hrm_ocr.eval import cer
        # "RAVI  KUMAR" (double space) == "RAVI KUMAR"
        assert cer("RAVI  KUMAR", "RAVI KUMAR") == 0.0

    def test_cer_strips_leading_trailing(self):
        from hrm_ocr.eval import cer
        assert cer("  ABCD  ", "ABCD") == 0.0


# ============================================================================
# CER
# ============================================================================

class TestCER:
    def test_identical_strings(self):
        from hrm_ocr.eval import cer
        assert cer("ABCDEF", "ABCDEF") == 0.0

    def test_both_empty(self):
        from hrm_ocr.eval import cer
        assert cer("", "") == 0.0

    def test_pred_empty_gt_nonempty(self):
        from hrm_ocr.eval import cer
        # All GT chars are deletions → CER = len(GT) / len(GT) = 1.0
        assert cer("", "ABCD") == 1.0

    def test_pred_nonempty_gt_empty(self):
        from hrm_ocr.eval import cer
        # GT is empty → return 1.0 (all insertions, nothing to normalise by)
        assert cer("ABCD", "") == 1.0

    def test_single_substitution(self):
        from hrm_ocr.eval import cer
        # "ABCE" vs "ABCD" — 1 substitution in 4 chars
        assert abs(cer("ABCE", "ABCD") - 0.25) < 1e-9

    def test_single_deletion(self):
        from hrm_ocr.eval import cer
        # "ABC" vs "ABCD" — 1 deletion, len(GT)=4
        assert abs(cer("ABC", "ABCD") - 0.25) < 1e-9

    def test_single_insertion(self):
        from hrm_ocr.eval import cer
        # "ABCDE" vs "ABCD" — 1 insertion, len(GT)=4
        assert abs(cer("ABCDE", "ABCD") - 0.25) < 1e-9

    def test_case_insensitive(self):
        from hrm_ocr.eval import cer
        assert cer("abcd", "ABCD") == 0.0

    def test_cer_can_exceed_one(self):
        from hrm_ocr.eval import cer
        # Very wrong prediction: many insertions relative to short GT
        result = cer("ABCDEFGHIJ", "AB")
        assert result > 1.0

    def test_typical_aadhaar_ocr_error(self):
        from hrm_ocr.eval import cer
        # "1234 5678 9O12" vs "1234 5678 9012" — 1 sub in 14 normalised chars
        pred = "1234 5678 9O12"
        gt   = "1234 5678 9012"
        # After normalise: "1234 5678 9O12" → "1234 5678 9O12" (14 chars)
        result = cer(pred, gt)
        assert result > 0.0
        assert result < 0.15  # Should be small (1 error in 14 chars ≈ 7%)

    def test_typical_pan_ocr_error(self):
        from hrm_ocr.eval import cer
        pred = "ABCDE1234G"  # last char wrong
        gt   = "ABCDE1234F"
        assert abs(cer(pred, gt) - 0.1) < 1e-9  # 1/10


# ============================================================================
# WER
# ============================================================================

class TestTokenEditDistance:
    def test_identical_tokens(self):
        from hrm_ocr.eval import _token_edit_distance
        assert _token_edit_distance(["A", "B"], ["A", "B"]) == 0

    def test_empty_lists(self):
        from hrm_ocr.eval import _token_edit_distance
        assert _token_edit_distance([], []) == 0

    def test_one_substitution(self):
        from hrm_ocr.eval import _token_edit_distance
        assert _token_edit_distance(["A", "X"], ["A", "B"]) == 1

    def test_one_deletion(self):
        from hrm_ocr.eval import _token_edit_distance
        assert _token_edit_distance(["A"], ["A", "B"]) == 1

    def test_one_insertion(self):
        from hrm_ocr.eval import _token_edit_distance
        assert _token_edit_distance(["A", "B", "C"], ["A", "B"]) == 1


class TestWER:
    def test_identical(self):
        from hrm_ocr.eval import wer
        assert wer("RAVI KUMAR", "RAVI KUMAR") == 0.0

    def test_both_empty(self):
        from hrm_ocr.eval import wer
        assert wer("", "") == 0.0

    def test_pred_empty(self):
        from hrm_ocr.eval import wer
        # All GT words are deletions
        assert wer("", "RAVI KUMAR") == 1.0

    def test_gt_empty_pred_nonempty(self):
        from hrm_ocr.eval import wer
        assert wer("RAVI", "") == 1.0

    def test_one_word_wrong(self):
        from hrm_ocr.eval import wer
        # "RAVI KAPOOR" vs "RAVI KUMAR" — 1 substitution in 2 tokens
        assert abs(wer("RAVI KAPOOR", "RAVI KUMAR") - 0.5) < 1e-9

    def test_missing_word(self):
        from hrm_ocr.eval import wer
        # "RAVI" vs "RAVI KUMAR" — 1 deletion in 2 tokens
        assert abs(wer("RAVI", "RAVI KUMAR") - 0.5) < 1e-9

    def test_case_insensitive(self):
        from hrm_ocr.eval import wer
        assert wer("ravi kumar", "RAVI KUMAR") == 0.0

    def test_extra_whitespace_ignored(self):
        from hrm_ocr.eval import wer
        assert wer("RAVI  KUMAR", "RAVI KUMAR") == 0.0


# ============================================================================
# field_accuracy
# ============================================================================

class TestFieldAccuracy:
    def test_exact_match(self):
        result = _fa({"name": "RAVI KUMAR"}, {"name": "RAVI KUMAR"})
        assert result["name"]["exact_match"] is True
        assert result["name"]["cer"] == 0.0
        assert result["name"]["wer"] == 0.0
        assert result["name"]["missing"] is False

    def test_inexact_match(self):
        result = _fa({"name": "RAVI KUMA"}, {"name": "RAVI KUMAR"})
        assert result["name"]["exact_match"] is False
        assert result["name"]["cer"] > 0.0

    def test_missing_field(self):
        result = _fa({}, {"name": "RAVI KUMAR"})
        assert result["name"]["exact_match"] is False
        assert result["name"]["cer"] == 1.0
        assert result["name"]["missing"] is True

    def test_extra_pred_field_ignored(self):
        """Fields in pred but not in gt are silently ignored."""
        result = _fa(
            {"name": "RAVI", "extra_field": "ignored"},
            {"name": "RAVI"},
        )
        assert "extra_field" not in result
        assert result["name"]["exact_match"] is True

    def test_multiple_fields(self):
        pred = {"name": "RAVI KUMAR", "dob": "15/08/1985", "pan_number": "ABCDE1234F"}
        gt   = {"name": "RAVI KUMAR", "dob": "15/08/1985", "pan_number": "ABCDE1234F"}
        result = _fa(pred, gt)
        for field in gt:
            assert result[field]["exact_match"] is True

    def test_normalisation_applied(self):
        """Pred 'ravi kumar' must match GT 'RAVI KUMAR' after normalise."""
        result = _fa({"name": "ravi kumar"}, {"name": "RAVI KUMAR"})
        assert result["name"]["exact_match"] is True

    def test_pred_and_gt_stored_normalised(self):
        result = _fa({"name": "  ravi  kumar  "}, {"name": "RAVI KUMAR"})
        assert result["name"]["pred"] == "RAVI KUMAR"
        assert result["name"]["gt"]   == "RAVI KUMAR"

    def test_returns_wer_for_each_field(self):
        result = _fa({"name": "RAVI"}, {"name": "RAVI KUMAR"})
        assert result["name"]["wer"] > 0.0


# ============================================================================
# confusion_pairs
# ============================================================================

class TestConfusionPairs:
    def test_no_errors_empty_counter(self):
        from hrm_ocr.eval import confusion_pairs
        c = confusion_pairs(["ABCD", "1234"], ["ABCD", "1234"])
        assert len(c) == 0

    def test_single_substitution(self):
        from hrm_ocr.eval import confusion_pairs
        c = confusion_pairs(["1O23"], ["1023"])
        # O substituted for 0
        assert c[("O", "0")] == 1

    def test_multiple_samples(self):
        from hrm_ocr.eval import confusion_pairs
        preds = ["1O23", "S678", "9O12"]
        gts   = ["1023", "5678", "9012"]
        c = confusion_pairs(preds, gts)
        assert c[("O", "0")] == 2
        assert c[("S", "5")] == 1

    def test_most_common_ordering(self):
        from hrm_ocr.eval import confusion_pairs
        preds = ["O", "O", "O", "I"]
        gts   = ["0", "0", "0", "1"]
        c = confusion_pairs(preds, gts)
        top = c.most_common(2)
        assert top[0][0] == ("O", "0")
        assert top[0][1] == 3

    def test_deletions_not_counted(self):
        """Deletions (missing chars) should NOT appear as substitutions."""
        from hrm_ocr.eval import confusion_pairs
        # "ABC" vs "ABCD" — D is deleted, not substituted
        c = confusion_pairs(["ABC"], ["ABCD"])
        # There should be no substitution pair for D
        for (p, g), _ in c.items():
            assert g != "D" or p != ""  # Deletion shouldn't create a pair

    def test_mismatched_lengths_raises(self):
        from hrm_ocr.eval import confusion_pairs
        with pytest.raises(ValueError, match="same length"):
            confusion_pairs(["A", "B"], ["A"])

    def test_empty_inputs(self):
        from hrm_ocr.eval import confusion_pairs
        c = confusion_pairs([], [])
        assert isinstance(c, Counter)
        assert len(c) == 0

    def test_returns_counter_type(self):
        from hrm_ocr.eval import confusion_pairs
        c = confusion_pairs(["AB"], ["AC"])
        assert isinstance(c, Counter)


# ============================================================================
# post_correction_delta
# ============================================================================

class TestPostCorrectionDelta:
    def test_correction_improves_exact_match(self):
        pre  = {"name": "RAVI KUMA R"}
        post = {"name": "RAVI KUMAR"}
        gt   = {"name": "RAVI KUMAR"}
        result = _delta(pre, post, gt)
        assert result["name"]["pre_exact"]  is False
        assert result["name"]["post_exact"] is True
        assert result["name"]["improved"]   is True
        assert result["name"]["degraded"]   is False
        assert result["name"]["cer_delta"]  < 0.0  # Lower CER = better

    def test_correction_degrades(self):
        """A bad rule that introduces an error: regression case."""
        pre  = {"pan_number": "ABCDE1234F"}   # was correct
        post = {"pan_number": "ABCDE1234G"}   # rule broke it
        gt   = {"pan_number": "ABCDE1234F"}
        result = _delta(pre, post, gt)
        assert result["pan_number"]["pre_exact"]  is True
        assert result["pan_number"]["post_exact"] is False
        assert result["pan_number"]["degraded"]   is True
        assert result["pan_number"]["improved"]   is False
        assert result["pan_number"]["cer_delta"]  > 0.0  # Higher CER = worse

    def test_no_change(self):
        pre  = {"dob": "15/08/1985"}
        post = {"dob": "15/08/1985"}
        gt   = {"dob": "15/08/1985"}
        result = _delta(pre, post, gt)
        assert result["dob"]["unchanged"] is True
        assert result["dob"]["improved"]  is False
        assert result["dob"]["degraded"]  is False
        assert result["dob"]["cer_delta"] == 0.0

    def test_missing_field_in_pre(self):
        """If field missing in pre, treat as empty string."""
        pre  = {}
        post = {"name": "RAVI KUMAR"}
        gt   = {"name": "RAVI KUMAR"}
        result = _delta(pre, post, gt)
        assert result["name"]["pre_exact"]  is False
        assert result["name"]["post_exact"] is True

    def test_missing_field_in_post(self):
        pre  = {"name": "RAVI KUMAR"}
        post = {}
        gt   = {"name": "RAVI KUMAR"}
        result = _delta(pre, post, gt)
        assert result["name"]["pre_exact"]  is True
        assert result["name"]["post_exact"] is False
        assert result["name"]["degraded"]   is True

    def test_cer_delta_rounded_to_4dp(self):
        pre  = {"name": "RAVI KUMA R"}
        post = {"name": "RAVI KUMAR"}
        gt   = {"name": "RAVI KUMAR"}
        result = _delta(pre, post, gt)
        # Should be rounded to 4 decimal places
        delta_val = result["name"]["cer_delta"]
        assert delta_val == round(delta_val, 4)

    def test_multiple_fields(self):
        pre  = {"name": "RAVI KUMA", "dob": "15/O8/1985"}
        post = {"name": "RAVI KUMAR", "dob": "15/08/1985"}
        gt   = {"name": "RAVI KUMAR", "dob": "15/08/1985"}
        result = _delta(pre, post, gt)
        assert result["name"]["improved"] is True
        assert result["dob"]["improved"]  is True


# ============================================================================
# aggregate_field_accuracy
# ============================================================================

class TestAggregateFieldAccuracy:
    def test_single_sample_passthrough(self):
        from hrm_ocr.eval import aggregate_field_accuracy, field_accuracy
        sample = field_accuracy({"name": "RAVI KUMAR"}, {"name": "RAVI KUMAR"})
        agg = aggregate_field_accuracy([sample])
        assert agg["name"]["exact_match_rate"] == 1.0
        assert agg["name"]["mean_cer"] == 0.0
        assert agg["name"]["n"] == 1

    def test_two_samples_average(self):
        from hrm_ocr.eval import aggregate_field_accuracy, field_accuracy
        s1 = field_accuracy({"name": "RAVI KUMAR"},  {"name": "RAVI KUMAR"})   # exact
        s2 = field_accuracy({"name": "RAVI KUMA R"}, {"name": "RAVI KUMAR"})   # not exact
        agg = aggregate_field_accuracy([s1, s2])
        assert agg["name"]["n"] == 2
        assert abs(agg["name"]["exact_match_rate"] - 0.5) < 1e-9

    def test_missing_field_counted(self):
        from hrm_ocr.eval import aggregate_field_accuracy, field_accuracy
        s1 = field_accuracy({"name": "RAVI"}, {"name": "RAVI"})    # present
        s2 = field_accuracy({},               {"name": "RAVI"})    # missing
        agg = aggregate_field_accuracy([s1, s2])
        assert abs(agg["name"]["missing_rate"] - 0.5) < 1e-9

    def test_multiple_fields(self):
        from hrm_ocr.eval import aggregate_field_accuracy, field_accuracy
        gt = {"name": "RAVI", "dob": "01/01/1990"}
        s  = field_accuracy({"name": "RAVI", "dob": "01/01/1990"}, gt)
        agg = aggregate_field_accuracy([s])
        assert "name" in agg
        assert "dob"  in agg


# ============================================================================
# aggregate_post_correction_delta
# ============================================================================

class TestAggregatePostCorrDelta:
    def test_single_improved_sample(self):
        from hrm_ocr.eval import aggregate_post_correction_delta, post_correction_delta
        d = post_correction_delta(
            {"name": "RAVI KUMA"},
            {"name": "RAVI KUMAR"},
            {"name": "RAVI KUMAR"},
        )
        agg = aggregate_post_correction_delta([d])
        assert agg["name"]["post_exact_rate"] == 1.0
        assert agg["name"]["pre_exact_rate"]  == 0.0
        assert agg["name"]["improved_rate"]   == 1.0
        assert agg["name"]["degraded_rate"]   == 0.0
        assert agg["name"]["exact_delta"]     == pytest.approx(1.0)

    def test_mix_improved_and_degraded(self):
        from hrm_ocr.eval import aggregate_post_correction_delta, post_correction_delta
        d1 = post_correction_delta(
            {"name": "RAVI KUMA"}, {"name": "RAVI KUMAR"}, {"name": "RAVI KUMAR"}
        )  # improved
        d2 = post_correction_delta(
            {"name": "RAVI KUMAR"}, {"name": "RAVI KUMA"}, {"name": "RAVI KUMAR"}
        )  # degraded
        agg = aggregate_post_correction_delta([d1, d2])
        assert abs(agg["name"]["improved_rate"] - 0.5) < 1e-9
        assert abs(agg["name"]["degraded_rate"] - 0.5) < 1e-9
        assert abs(agg["name"]["exact_delta"])   < 1e-9  # net zero

    def test_empty_list(self):
        from hrm_ocr.eval import aggregate_post_correction_delta
        agg = aggregate_post_correction_delta([])
        assert agg == {}


# ============================================================================
# EvalReport Pydantic model
# ============================================================================

class TestEvalReport:
    def _minimal_report(self) -> dict:
        return {
            "doc_type": "aadhaar",
            "extraction_method": "ocr",
            "sample_count": 100,
            "mean_cer": 0.05,
            "mean_wer": 0.08,
        }

    def test_valid_report(self):
        from hrm_ocr.eval import EvalReport
        r = EvalReport(**self._minimal_report())
        assert r.doc_type == "aadhaar"
        assert r.sample_count == 100

    def test_default_timestamp_is_utc_iso(self):
        from hrm_ocr.eval import EvalReport
        r = EvalReport(**self._minimal_report())
        assert "T" in r.timestamp or "+" in r.timestamp

    def test_extraction_method_literal(self):
        from hrm_ocr.eval import EvalReport
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EvalReport(**{**self._minimal_report(), "extraction_method": "invalid"})

    def test_per_field_accuracy_populated(self):
        from hrm_ocr.eval import EvalReport, FieldAccuracyStats
        r = EvalReport(
            **self._minimal_report(),
            per_field_accuracy={
                "name": FieldAccuracyStats(
                    exact_match_rate=0.95,
                    mean_cer=0.02,
                    mean_wer=0.03,
                    n=100,
                    missing_rate=0.0,
                )
            },
        )
        assert "name" in r.per_field_accuracy
        assert r.per_field_accuracy["name"].exact_match_rate == 0.95

    def test_post_correction_delta_populated(self):
        from hrm_ocr.eval import CorrectionDeltaStats, EvalReport
        r = EvalReport(
            **self._minimal_report(),
            post_correction_delta={
                "name": CorrectionDeltaStats(
                    pre_exact_rate=0.88,
                    post_exact_rate=0.95,
                    exact_delta=0.07,
                    mean_cer_delta=-0.02,
                    improved_rate=0.08,
                    degraded_rate=0.01,
                    n=100,
                )
            },
        )
        assert r.post_correction_delta["name"].exact_delta == pytest.approx(0.07)

    def test_top_confusions_stored(self):
        from hrm_ocr.eval import EvalReport
        r = EvalReport(
            **self._minimal_report(),
            top_confusions=[{"pred": "O", "gt": "0", "count": 42}],
        )
        assert r.top_confusions[0]["pred"] == "O"

    def test_serialises_to_json_and_back(self):
        from hrm_ocr.eval import EvalReport
        r = EvalReport(**self._minimal_report())
        json_str = r.model_dump_json()
        reloaded = EvalReport.model_validate_json(json_str)
        assert reloaded.doc_type == r.doc_type
        assert reloaded.mean_cer == r.mean_cer

    def test_field_accuracy_stats_bounds(self):
        from hrm_ocr.eval import FieldAccuracyStats
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FieldAccuracyStats(
                exact_match_rate=1.5,  # > 1.0
                mean_cer=0.0,
                mean_wer=0.0,
                n=10,
                missing_rate=0.0,
            )


# ============================================================================
# run_evaluation() integration
# ============================================================================

class TestRunEvalIntegration:
    """Test run_evaluation() with small in-memory fixtures.

    No file I/O, no OCR, no network — pure metric computation.
    """

    def _pred(self, id_: str, fields: dict, raw_fields: dict | None = None) -> dict:
        rec = {"id": id_, "fields": fields}
        if raw_fields is not None:
            rec["raw_fields"] = raw_fields
        return rec

    def _gt(self, id_: str, fields: dict) -> dict:
        return {"id": id_, "fields": fields}

    def test_perfect_predictions(self):
        from scripts.run_eval import run_evaluation
        preds = [self._pred("1", {"name": "RAVI KUMAR", "dob": "15/08/1985"})]
        gts   = [self._gt("1",  {"name": "RAVI KUMAR", "dob": "15/08/1985"})]
        report = run_evaluation(preds, gts, "aadhaar", "ocr")
        assert report.sample_count == 1
        assert report.mean_cer == 0.0
        assert report.per_field_accuracy["name"].exact_match_rate == 1.0
        assert report.per_field_accuracy["dob"].exact_match_rate == 1.0

    def test_partial_errors(self):
        from scripts.run_eval import run_evaluation
        preds = [self._pred("1", {"name": "RAVI KUMA", "dob": "15/08/1985"})]
        gts   = [self._gt("1",  {"name": "RAVI KUMAR", "dob": "15/08/1985"})]
        report = run_evaluation(preds, gts, "aadhaar", "ocr")
        assert report.per_field_accuracy["name"].exact_match_rate == 0.0
        assert report.per_field_accuracy["dob"].exact_match_rate  == 1.0
        assert report.mean_cer > 0.0

    def test_post_correction_delta_computed_when_raw_present(self):
        from scripts.run_eval import run_evaluation
        preds = [self._pred(
            "1",
            {"name": "RAVI KUMAR"},
            raw_fields={"name": "RAVI KUMA R"},
        )]
        gts = [self._gt("1", {"name": "RAVI KUMAR"})]
        report = run_evaluation(preds, gts, "aadhaar", "ocr")
        assert "name" in report.post_correction_delta
        delta = report.post_correction_delta["name"]
        assert delta.post_exact_rate == 1.0
        assert delta.pre_exact_rate  == 0.0
        assert delta.exact_delta     == pytest.approx(1.0)

    def test_no_delta_when_raw_fields_absent(self):
        from scripts.run_eval import run_evaluation
        preds = [self._pred("1", {"name": "RAVI KUMAR"})]  # no raw_fields
        gts   = [self._gt("1",  {"name": "RAVI KUMAR"})]
        report = run_evaluation(preds, gts, "pan", "ocr")
        assert report.post_correction_delta == {}

    def test_top_confusions_populated(self):
        from scripts.run_eval import run_evaluation
        preds = [
            self._pred("1", {"aadhaar_number": "1234 5678 9O12"}),
            self._pred("2", {"aadhaar_number": "1234 5678 9O12"}),
        ]
        gts = [
            self._gt("1", {"aadhaar_number": "1234 5678 9012"}),
            self._gt("2", {"aadhaar_number": "1234 5678 9012"}),
        ]
        report = run_evaluation(preds, gts, "aadhaar", "ocr")
        # O → 0 should be the top confusion
        assert len(report.top_confusions) > 0
        top = report.top_confusions[0]
        assert top["pred"] == "O"
        assert top["gt"]   == "0"
        assert top["count"] == 2

    def test_unmatched_pred_ids_warned(self, capsys):
        from scripts.run_eval import run_evaluation
        preds = [
            self._pred("1", {"name": "RAVI"}),
            self._pred("99", {"name": "GHOST"}),  # no GT counterpart
        ]
        gts = [self._gt("1", {"name": "RAVI"})]
        report = run_evaluation(preds, gts, "aadhaar", "ocr")
        captured = capsys.readouterr()
        assert "99" in captured.err or "prediction" in captured.err.lower()

    def test_no_matching_ids_exits(self):
        from scripts.run_eval import run_evaluation
        preds = [self._pred("999", {"name": "X"})]
        gts   = [self._gt("1",   {"name": "Y"})]
        with pytest.raises(SystemExit):
            run_evaluation(preds, gts, "aadhaar", "ocr")

    def test_doc_type_and_method_stored(self):
        from scripts.run_eval import run_evaluation
        preds = [self._pred("1", {"name": "RAVI"})]
        gts   = [self._gt("1",  {"name": "RAVI"})]
        report = run_evaluation(preds, gts, "pan", "text_layer")
        assert report.doc_type == "pan"
        assert report.extraction_method == "text_layer"


# ============================================================================
# CLI smoke test
# ============================================================================

class TestCLI:
    """Smoke tests for the argparse CLI and file output."""

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    def test_cli_creates_json_and_md_output(self, tmp_path: Path):
        from scripts.run_eval import main

        preds = [
            {"id": "1", "fields": {"name": "RAVI KUMAR"}, "raw_fields": {"name": "RAVI KUMA R"}},
            {"id": "2", "fields": {"name": "PRIYA SHARMA"}, "raw_fields": {"name": "PRIYA SHARMA"}},
        ]
        gts = [
            {"id": "1", "fields": {"name": "RAVI KUMAR"}},
            {"id": "2", "fields": {"name": "PRIYA SHARMA"}},
        ]
        pred_path = tmp_path / "pred.jsonl"
        gt_path   = tmp_path / "gt.jsonl"
        out_dir   = tmp_path / "out"
        self._write_jsonl(pred_path, preds)
        self._write_jsonl(gt_path, gts)

        main([
            "--predictions",       str(pred_path),
            "--ground_truth",      str(gt_path),
            "--doc_type",          "aadhaar",
            "--extraction_method", "ocr",
            "--output_dir",        str(out_dir),
            "--no_color",
        ])

        json_files = list(out_dir.glob("*_eval_report.json"))
        md_files   = list(out_dir.glob("*_eval_report.md"))
        assert len(json_files) == 1
        assert len(md_files)   == 1

        # Validate JSON can be parsed back as EvalReport
        from hrm_ocr.eval import EvalReport
        report = EvalReport.model_validate_json(json_files[0].read_text())
        assert report.sample_count == 2
        assert report.doc_type == "aadhaar"

    def test_cli_missing_pred_file_exits(self, tmp_path: Path):
        from scripts.run_eval import main
        with pytest.raises(SystemExit):
            main([
                "--predictions",       str(tmp_path / "nonexistent.jsonl"),
                "--ground_truth",      str(tmp_path / "nonexistent.jsonl"),
                "--doc_type",          "aadhaar",
                "--extraction_method", "ocr",
            ])

    def test_cli_invalid_extraction_method_exits(self, tmp_path: Path):
        from scripts.run_eval import main
        pred_path = tmp_path / "p.jsonl"
        gt_path   = tmp_path / "g.jsonl"
        pred_path.write_text("")
        gt_path.write_text("")
        with pytest.raises(SystemExit):
            main([
                "--predictions",       str(pred_path),
                "--ground_truth",      str(gt_path),
                "--doc_type",          "aadhaar",
                "--extraction_method", "bad_method",
            ])

    def test_cli_output_json_is_valid_eval_report(self, tmp_path: Path):
        from scripts.run_eval import main
        from hrm_ocr.eval import EvalReport

        records = [{"id": str(i), "fields": {"pan_number": "ABCDE1234F"}} for i in range(5)]
        gts     = [{"id": str(i), "fields": {"pan_number": "ABCDE1234F"}} for i in range(5)]
        pred_path = tmp_path / "p.jsonl"
        gt_path   = tmp_path / "g.jsonl"
        self._write_jsonl(pred_path, records)
        self._write_jsonl(gt_path, gts)

        main([
            "--predictions",       str(pred_path),
            "--ground_truth",      str(gt_path),
            "--doc_type",          "pan",
            "--extraction_method", "text_layer",
            "--output_dir",        str(tmp_path / "out"),
            "--no_color",
        ])

        json_file = next((tmp_path / "out").glob("*.json"))
        report = EvalReport.model_validate_json(json_file.read_text())
        assert report.mean_cer == 0.0
        assert report.extraction_method == "text_layer"
