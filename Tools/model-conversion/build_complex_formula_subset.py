#!/usr/bin/env python3
"""Build a reproducible complex and multi-line subset from Layer 1."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
from typing import Any

from build_simple_single_line_subset import formula_profile


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR
    / "datasets"
    / "evaluation"
    / "layer1-im2latex-test"
    / "manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "datasets" / "evaluation" / "layer1b-complex-formulas"
)
MULTILINE_REASONS = {
    "multi_line_environment",
    "explicit_line_break",
    "stacked_expression",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select complex and multi-line formulas using only manifest "
            "metadata and LaTeX structure, never model predictions."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of records to select; use 0 to keep every eligible record.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def selection_reasons(
    record: dict[str, Any], profile: dict[str, int | None]
) -> list[str]:
    broad_reasons = set(record.get("out_of_scope_reasons", []))
    reasons = []

    if broad_reasons & MULTILINE_REASONS:
        reasons.append("multi_line")
    if "latex_too_long" in broad_reasons:
        reasons.append("broad_scope_latex_too_long")
    if "formula_too_tall" in broad_reasons:
        reasons.append("broad_scope_formula_too_tall")

    # These thresholds deliberately select a stress set, not the product scope.
    if profile["latex_chars"] >= 160:
        reasons.append("latex_at_least_160_chars")
    if profile["commands"] >= 20:
        reasons.append("at_least_20_commands")
    if profile["scripts"] >= 15:
        reasons.append("at_least_15_scripts")
    if profile["fractions"] >= 4:
        reasons.append("at_least_4_fractions")
    if profile["large_operators"] >= 3:
        reasons.append("at_least_3_large_operators")
    if profile["formula_height"] >= 100:
        reasons.append("formula_at_least_100px_tall")

    return reasons


def category(record: dict[str, Any], reasons: list[str]) -> str:
    if "multi_line" in reasons:
        return "multi_line"
    if record.get("in_v0_1_scope") is not True:
        return "long_or_tall_out_of_scope"
    return "complex_single_line"


def main() -> None:
    args = parse_args()
    if args.sample_size < 0:
        raise ValueError("--sample-size must be 0 or greater")

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Layer 1 manifest not found: {input_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for record in read_jsonl(input_path):
        profile = formula_profile(record)
        reasons = selection_reasons(record, profile)
        if reasons:
            candidates.append((record, profile, reasons))

    if args.sample_size == 0:
        selected = candidates
    else:
        if args.sample_size > len(candidates):
            raise ValueError(
                f"--sample-size {args.sample_size} exceeds "
                f"{len(candidates)} eligible records"
            )
        selected = random.Random(args.seed).sample(candidates, args.sample_size)

    selected.sort(key=lambda item: (item[0].get("test_index", 0), item[0]["id"]))
    output_records = []
    category_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for record, profile, reasons in selected:
        source_image = (input_path.parent / record["image"]).resolve()
        if not source_image.is_file():
            raise FileNotFoundError(f"Source image not found: {source_image}")
        record_category = category(record, reasons)
        category_counts[record_category] += 1
        reason_counts.update(reasons)

        output_record = dict(record)
        output_record["image"] = os.path.relpath(source_image, output_dir)
        output_record["subset"] = "complex-formulas-v1"
        output_record["complex_formula_category"] = record_category
        output_record["complex_selection_reasons"] = reasons
        output_record["complex_profile"] = profile
        output_records.append(output_record)

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, output_records)
    summary = {
        "schema_version": 1,
        "subset": "complex-formulas-v1",
        "selection_independent_of_model_predictions": True,
        "source_manifest": str(input_path),
        "eligible_records": len(candidates),
        "selected_records": len(output_records),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "category_counts": dict(sorted(category_counts.items())),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "thresholds": {
            "min_latex_chars": 160,
            "min_commands": 20,
            "min_scripts": 15,
            "min_fractions": 4,
            "min_large_operators": 3,
            "min_formula_height": 100,
            "include_existing_multiline_and_broad_scope_exclusions": True,
        },
        "manifest": str(manifest_path),
        "image_storage": "References Layer 1 images; no images are duplicated.",
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== complex formula subset ===")
    print(f"Eligible records: {len(candidates)}")
    print(f"Selected records: {len(output_records)}")
    print(f"Categories: {dict(sorted(category_counts.items()))}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
