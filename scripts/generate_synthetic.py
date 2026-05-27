#!/usr/bin/env python3
"""
scripts/generate_synthetic.py
=============================
CLI to generate synthetic ID cards.

Usage:
  python scripts/generate_synthetic.py \\
    --doc-type aadhaar \\
    --version aadhaar_v3 \\
    --count 100 \\
    --output-dir data/synthetic/aadhaar
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hrm_ocr.data.aadhaar_gen import generate_aadhaar
from hrm_ocr.data.pan_gen import generate_pan


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ID cards.")
    parser.add_argument("--doc-type", required=True, choices=["aadhaar", "pan"],
                        help="Document type to generate.")
    parser.add_argument("--version", required=True,
                        help="Template version (e.g. aadhaar_v3, pan_v1).")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of samples to generate.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"),
                        help="Output directory.")
                        
    args = parser.parse_args()
    
    try:
        if args.doc_type == "aadhaar":
            generate_aadhaar(args.version, args.count, args.output_dir)
        elif args.doc_type == "pan":
            generate_pan(args.version, args.count, args.output_dir)
            
        print(f"Successfully generated {args.count} cards for {args.version}.")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
