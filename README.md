# HRM OCR API

Production-grade HRM OCR API — lightweight, accurate, simple. Extracts structured fields from Aadhaar and PAN cards via rule-based pipeline and PaddleOCR.

## Design Philosophy

* **Lightweight**: Docker image < 300 MB, full card OCR < 150 ms on CPU.
* **Accurate**: 100 % on structured fields via rule-based post-correction.
* **Simple**: No ML where rules suffice.

## System Requirements

To run this project, you need the following system packages installed:
* `poppler-utils` (Required for PDF rendering via `pdf2image`)

**Ubuntu / Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS (Homebrew):**
```bash
brew install poppler
```

## Running the API

```bash
poetry install
poetry run hrm-ocr
```

## Evaluation

```bash
poetry run python scripts/run_eval.py --predictions data/annotations/pred_aadhaar.jsonl --ground_truth data/annotations/gt_aadhaar.jsonl --doc_type aadhaar --extraction_method ocr
```

## Active Learning Feedback Loop (Cron Jobs)

To automate the active learning process with Label Studio, configure the following `crontab` entries on your deployment server:

```bash
# Push uncorrected anomalies to Label Studio every hour
0 * * * * cd /path/to/hrm-ocr-api && poetry run python src/hrm_ocr/feedback/push_to_ls.py >> logs/cron_push.log 2>&1

# Pull human corrections from Label Studio every 30 minutes
*/30 * * * * cd /path/to/hrm-ocr-api && poetry run python src/hrm_ocr/feedback/pull_corrections.py >> logs/cron_pull.log 2>&1

# Nightly automated rule generation analysis at 2:00 AM
0 2 * * * cd /path/to/hrm-ocr-api && poetry run python src/hrm_ocr/feedback/retrain_trigger.py >> logs/cron_retrain.log 2>&1
```
