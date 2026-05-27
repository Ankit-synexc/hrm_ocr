#!/usr/bin/env python3
"""
scripts/download_reference_data.py
==================================
Script to download and build local SQLite databases for offline reference validation.
Builds `ifsc.db` and `pincodes.db` in `data/reference/`.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

# In a production environment, this would fetch the master CSV lists from RBI/India Post APIs.
# For demonstration and bootstrap purposes, we generate realistic seed databases.

logger = logging.getLogger(__name__)


def build_ifsc_db(db_path: Path) -> None:
    logger.info("Building IFSC database at %s", db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ifsc_codes (
            code TEXT PRIMARY KEY
        )
    """)
    
    # Mock some valid IFSC codes for testing
    mock_codes = [
        "HDFC0000001",
        "SBIN0001234",
        "ICIC0000001",
        "PUNB0123456"
    ]
    
    cursor.executemany("INSERT OR IGNORE INTO ifsc_codes (code) VALUES (?)", [(c,) for c in mock_codes])
    conn.commit()
    conn.close()
    logger.info("Successfully built IFSC database.")


def build_pincode_db(db_path: Path) -> None:
    logger.info("Building Pincode database at %s", db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pincodes (
            code TEXT PRIMARY KEY
        )
    """)
    
    # Mock some valid pincodes for testing
    mock_codes = [
        "110001", # Delhi
        "400001", # Mumbai
        "560001", # Bangalore
        "600001"  # Chennai
    ]
    
    cursor.executemany("INSERT OR IGNORE INTO pincodes (code) VALUES (?)", [(c,) for c in mock_codes])
    conn.commit()
    conn.close()
    logger.info("Successfully built Pincode database.")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ref_dir = repo_root / "data" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    
    ifsc_db_path = ref_dir / "ifsc.db"
    pincode_db_path = ref_dir / "pincodes.db"
    
    try:
        build_ifsc_db(ifsc_db_path)
        build_pincode_db(pincode_db_path)
        logger.info("Reference data download and build complete.")
    except Exception as e:
        logger.error("Failed to build reference databases: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
