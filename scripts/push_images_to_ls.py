#!/usr/bin/env python3
"""
scripts/push_images_to_ls.py
============================
CLI to set up a new annotation sprint in Label Studio.

Creates a new project with the HRM OCR schema and imports images from
the raw data directory (via local storage sync).

Usage:
  python scripts/push_images_to_ls.py \\
      --doc-type aadhaar \\
      --image-dir data/raw/aadhaar_sprint_1 \\
      --url http://localhost:8080 \\
      --token <your_ls_token>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is on path when running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hrm_ocr.data.label_studio import (
    LabelStudioClient,
    create_project,
    import_images,
    export_annotations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Push images to Label Studio for annotation.")
    parser.add_argument("--action", choices=["setup", "export"], default="setup",
                        help="Action to perform (setup a new sprint, or export completed ones).")
    parser.add_argument("--doc-type", required=True,
                        help="Document type (e.g., aadhaar, pan). Used for project naming.")
    parser.add_argument("--image-dir", type=Path,
                        help="Path to directory containing raw images (relative to repo root). Required for 'setup'.")
    parser.add_argument("--project-id", type=int,
                        help="Label Studio Project ID. Required for 'export'.")
    parser.add_argument("--output", type=Path, default=Path("data/annotations/export.json"),
                        help="Output JSON path for 'export' action.")
    parser.add_argument("--url", default="http://localhost:8080",
                        help="Label Studio API URL.")
    parser.add_argument("--token", required=True,
                        help="Label Studio API Token.")
    
    args = parser.parse_args()
    
    client = LabelStudioClient(base_url=args.url, token=args.token)
    
    if args.action == "setup":
        if not args.image_dir:
            print("ERROR: --image-dir is required for 'setup' action.", file=sys.stderr)
            sys.exit(1)
            
        config_xml_path = Path(__file__).resolve().parents[1] / "configs" / "label_studio_config.xml"
        if not config_xml_path.exists():
            print(f"ERROR: Config XML not found at {config_xml_path}", file=sys.stderr)
            sys.exit(1)
            
        config_xml = config_xml_path.read_text(encoding="utf-8")
        project_name = f"HRM OCR - {args.doc_type.upper()} Sprint"
        
        try:
            project_id = create_project(client, project_name, config_xml)
            import_images(project_id, args.image_dir, client)
            print(f"\nSuccessfully created project '{project_name}' (ID: {project_id}).")
            print(f"Images from {args.image_dir} are syncing.")
            print(f"Target: 200 annotated samples for {args.doc_type}.")
        except Exception as exc:
            print(f"ERROR: Setup failed: {exc}", file=sys.stderr)
            sys.exit(1)
            
    elif args.action == "export":
        if not args.project_id:
            print("ERROR: --project-id is required for 'export' action.", file=sys.stderr)
            sys.exit(1)
            
        try:
            export_annotations(args.project_id, client, args.output)
            print(f"\nSuccessfully exported annotations to {args.output}")
        except Exception as exc:
            print(f"ERROR: Export failed: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
