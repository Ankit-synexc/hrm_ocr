"""
hrm_ocr.data.label_studio
==========================
Label Studio integration for ground-truth annotation.

This module provides functions to create projects, upload raw images,
and export completed annotations to a flattened JSON format ready for
the evaluation harness.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LabelStudioClient:
    """Minimal REST client for Label Studio API v2."""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        }


def create_project(client: LabelStudioClient, name: str, config_xml: str) -> int:
    """Create a new Label Studio project and return its ID."""
    logger.info("Creating Label Studio project: '%s'", name)
    
    payload = {
        "title": name,
        "label_config": config_xml,
    }
    
    resp = httpx.post(
        f"{client.base_url}/api/projects",
        headers=client.headers,
        json=payload,
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()
    project_id = data["id"]
    logger.info("Project created successfully (ID: %d)", project_id)
    return project_id


def import_images(project_id: int, image_dir: Path, client: LabelStudioClient) -> None:
    """Import images from a local directory into a Label Studio project.
    
    Assumes Label Studio is running via Docker and has mounted the parent of
    image_dir to /label-studio/data/local-files, as configured in docker-compose.yml.
    """
    logger.info("Importing images from %s into project %d", image_dir, project_id)
    
    if not image_dir.exists():
        logger.error("Directory not found: %s", image_dir)
        return

    # Use the Local Storage import API so we don't upload bytes,
    # but rather sync from the mounted volume.
    payload = {
        "project": project_id,
        "type": "localfiles",
        "title": f"Sync from {image_dir.name}",
        "path": f"/label-studio/data/local-files/{image_dir.name}",
        "use_blob_urls": True,
    }

    # 1. Create the storage connection
    resp = httpx.post(
        f"{client.base_url}/api/storages/localfiles",
        headers=client.headers,
        json=payload,
        timeout=10.0,
    )
    resp.raise_for_status()
    storage_id = resp.json()["id"]
    logger.info("Created local storage connection (ID: %d)", storage_id)

    # 2. Trigger the sync
    sync_resp = httpx.post(
        f"{client.base_url}/api/storages/localfiles/{storage_id}/sync",
        headers=client.headers,
        timeout=30.0,
    )
    sync_resp.raise_for_status()
    logger.info("Triggered sync for storage %d", storage_id)


def export_annotations(project_id: int, client: LabelStudioClient, output_path: Path) -> None:
    """Export completed annotations to a flattened JSON format.
    
    Exports tasks from Label Studio and flattens them into records containing:
    {
        "image_path": str,
        "field_name": str,
        "bbox": list[float],
        "text_value": str,
        "extraction_method": str
    }
    """
    logger.info("Exporting annotations from project %d to %s", project_id, output_path)
    
    # 1. Generate an export file (Label Studio creates an export task)
    resp = httpx.get(
        f"{client.base_url}/api/projects/{project_id}/export",
        headers=client.headers,
        params={"exportType": "JSON"},
        timeout=30.0,
    )
    resp.raise_for_status()
    tasks = resp.json()
    
    records: list[dict[str, Any]] = []
    
    for task in tasks:
        # Resolve image path (handle Label Studio's internal path format)
        image_url = task.get("data", {}).get("image", "")
        # E.g., /data/local-files/?d=aadhaar/card_01.jpg -> aadhaar/card_01.jpg
        image_path = image_url.split("?d=")[-1] if "?d=" in image_url else image_url
        
        # Only process completed annotations
        annotations = task.get("annotations", [])
        if not annotations:
            continue
            
        # Use the most recent annotation
        annotation = annotations[0]
        results = annotation.get("result", [])
        
        # Group results by ID (bounding box and transcription share an ID)
        grouped_results: dict[str, dict[str, Any]] = {}
        
        for r in results:
            rid = r.get("id")
            if not rid:
                continue
                
            if rid not in grouped_results:
                grouped_results[rid] = {
                    "bbox": [],
                    "field_name": "",
                    "text_value": ""
                }
                
            val = r.get("value", {})
            rtype = r.get("type")
            
            if rtype == "rectanglelabels":
                labels = val.get("rectanglelabels", [])
                if labels:
                    grouped_results[rid]["field_name"] = labels[0]
                
                # Convert percentages to relative [x, y, w, h] (0-1)
                x = val.get("x", 0) / 100.0
                y = val.get("y", 0) / 100.0
                w = val.get("width", 0) / 100.0
                h = val.get("height", 0) / 100.0
                grouped_results[rid]["bbox"] = [x, y, w, h]
                
            elif rtype == "textarea":
                texts = val.get("text", [])
                if texts:
                    grouped_results[rid]["text_value"] = texts[0]

        # Convert to flat records
        for group in grouped_results.values():
            if not group["field_name"]:
                continue
                
            records.append({
                "image_path": image_path,
                "field_name": group["field_name"],
                "bbox": group["bbox"],
                "text_value": group["text_value"],
                "extraction_method": "ocr"  # Default assumption for bounding boxes
            })
            
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
        
    logger.info("Exported %d annotation records", len(records))
