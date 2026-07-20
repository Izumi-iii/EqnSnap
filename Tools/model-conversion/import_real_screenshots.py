#!/usr/bin/env python3
"""Import real formula screenshots as EqnSnap Layer 3 evaluation data."""

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path.home() / "Documents" / "final_test_images"
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "datasets" / "evaluation" / "layer3-real-screenshots"
)
CSV_FIELDS = (
    "id",
    "source_filename",
    "image",
    "expected_latex",
    "scope",
    "notes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import real PNG formula screenshots and create a label template."
    )
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(index: int) -> str:
    return f"real-{index:04d}"


def read_existing_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        return {
            row["source_filename"]: row
            for row in csv.DictReader(file)
            if row.get("source_filename")
        }


def scope_value(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "in":
        return True
    if normalized == "out":
        return False
    return None


def write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    source_dir = args.source.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Screenshot directory not found: {source_dir}")

    source_images = sorted(source_dir.glob("*.png"), key=natural_key)
    if not source_images:
        raise RuntimeError(f"No PNG screenshots found in {source_dir}")

    labels_path = output_dir / "labels.csv"
    existing_labels = read_existing_labels(labels_path)
    images_dir = output_dir / "images"
    shutil.rmtree(images_dir, ignore_errors=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    label_rows = []
    records = []
    for index, source_path in enumerate(source_images, start=1):
        sample_id = stable_id(index)
        filename = f"{sample_id}.png"
        destination = images_dir / filename
        shutil.copy2(source_path, destination)
        if sha256(source_path) != sha256(destination):
            raise RuntimeError(f"Copied image does not match source: {source_path}")

        with Image.open(destination) as image:
            image.verify()
        with Image.open(destination) as image:
            image_size = list(image.size)
            image_mode = image.mode

        previous = existing_labels.get(source_path.name, {})
        label_row = {
            "id": sample_id,
            "source_filename": source_path.name,
            "image": str(Path("images") / filename),
            "expected_latex": previous.get("expected_latex", ""),
            "scope": previous.get("scope", ""),
            "notes": previous.get("notes", ""),
        }
        label_rows.append(label_row)
        expected_latex = label_row["expected_latex"].strip()
        records.append(
            {
                "id": sample_id,
                "layer": 3,
                "source_type": "real_screenshot",
                "source_filename": source_path.name,
                "source_path": str(source_path),
                "image": label_row["image"],
                "sha256": sha256(destination),
                "size": image_size,
                "mode": image_mode,
                "expected_latex": expected_latex or None,
                "label_status": "verified" if expected_latex else "pending",
                "in_v0_1_scope": scope_value(label_row["scope"]),
                "notes": label_row["notes"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_labels(labels_path, label_rows)
    manifest_path = output_dir / "manifest.jsonl"
    write_manifest(manifest_path, records)
    summary = {
        "schema_version": 1,
        "source_directory": str(source_dir),
        "records": len(records),
        "labeled": sum(record["label_status"] == "verified" for record in records),
        "pending_labels": sum(
            record["label_status"] == "pending" for record in records
        ),
        "scope_in": sum(record["in_v0_1_scope"] is True for record in records),
        "scope_out": sum(record["in_v0_1_scope"] is False for record in records),
        "scope_unreviewed": sum(
            record["in_v0_1_scope"] is None for record in records
        ),
        "labels": "labels.csv",
        "manifest": "manifest.jsonl",
        "rule": (
            "Model predictions must not be written into expected_latex unless "
            "a person has verified them against the image."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== Layer 3 real screenshots imported ===")
    print(f"Images: {len(records)}")
    print(f"Pending labels: {summary['pending_labels']}")
    print(f"Label template: {labels_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
