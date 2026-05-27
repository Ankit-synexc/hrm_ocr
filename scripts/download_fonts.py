#!/usr/bin/env python3
"""
scripts/download_fonts.py
==========================
Download freely-licensed fonts used for synthetic card rendering.

Fonts downloaded:
  - Noto Sans (Regular + Bold) — standard card body text
  - Baloo 2 — decorative header text (Aadhaar cards)
  - IBM Plex Mono — monospace (Aadhaar 12-digit number)

All fonts from Google Fonts, licensed under OFL (Open Font Licence).
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

FONTS = {
    "NotoSans-Regular.ttf": (
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSans/NotoSans-Regular.ttf"
    ),
    "NotoSans-Bold.ttf": (
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
        "NotoSans/NotoSans-Bold.ttf"
    ),
    "Baloo2-Regular.ttf": (
        "https://github.com/EkType/Baloo2/raw/master/fonts/ttf/"
        "Baloo2-Regular.ttf"
    ),
    "IBMPlexMono-Regular.ttf": (
        "https://github.com/IBM/plex/raw/master/IBM-Plex-Mono/fonts/complete/ttf/"
        "IBMPlexMono-Regular.ttf"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download fonts for synthetic data generation")
    parser.add_argument("--output-dir", default="data/reference/fonts",
                        help="Directory to save font files")
    args = parser.parse_args()

    fonts_dir = Path(args.output_dir)
    fonts_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in FONTS.items():
        dest = fonts_dir / filename
        if dest.exists():
            print(f"[skip] {filename} already downloaded.")
            continue
        print(f"Downloading {filename} …")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  ✓ {dest}")
        except Exception as exc:
            print(f"  ✗ Failed to download {filename}: {exc}")

    print("\n✓ Font download complete.")


if __name__ == "__main__":
    main()
