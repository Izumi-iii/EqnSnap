#!/usr/bin/env python3
"""Build a reproducible strict simple-single-line subset from Layer 1."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR
    / "datasets"
    / "evaluation"
    / "layer1-im2latex-test"
    / "manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "datasets" / "evaluation" / "layer1a-simple-single-line"
)

COMMAND = re.compile(r"\\[A-Za-z]+|\\.")
FORBIDDEN_CONSTRUCTS = re.compile(
    r"\\(?:begin|end|text|textrm|textbf|textit|mbox|hbox|vbox|parbox|"
    r"raisebox|includegraphics)\b"
)
LARGE_OPERATORS = (r"\sum", r"\prod", r"\int", r"\oint", r"\lim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a fixed strict subset of clear, simple, single-line formulas "
            "without using model predictions."
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


def brace_depth(latex: str) -> int | None:
    depth = 0
    maximum = 0
    for character in latex:
        if character == "{":
            depth += 1
            maximum = max(maximum, depth)
        elif character == "}":
            depth -= 1
            if depth < 0:
                return None
    return maximum if depth == 0 else None


def formula_profile(record: dict[str, Any]) -> dict[str, int | None]:
    latex = str(record.get("normalized_latex", ""))
    formula_size = record.get("formula_size") or [0, 0]
    return {
        "latex_chars": len(latex),
        "formula_width": int(formula_size[0]),
        "formula_height": int(formula_size[1]),
        "commands": len(COMMAND.findall(latex)),
        "brace_depth": brace_depth(latex),
        "scripts": latex.count("^") + latex.count("_"),
        "fractions": sum(
            latex.count(command) for command in (r"\frac", r"\over", r"\atop")
        ),
        "roots": latex.count(r"\sqrt"),
        "large_operators": sum(latex.count(command) for command in LARGE_OPERATORS),
    }


def exclusion_reasons(record: dict[str, Any]) -> tuple[list[str], dict[str, int | None]]:
    latex = str(record.get("normalized_latex", ""))
    profile = formula_profile(record)
    reasons = []

    if record.get("in_v0_1_scope") is not True:
        reasons.append("outside_broad_v0_1_scope")
    if not latex:
        reasons.append("missing_latex")
    if len(latex) > 60:
        reasons.append("latex_over_60_chars")
    if "%" in latex:
        reasons.append("latex_comment_or_export_artifact")
    if r"\\" in latex or "\n" in latex or "\r" in latex:
        reasons.append("explicit_or_physical_line_break")
    if FORBIDDEN_CONSTRUCTS.search(latex):
        reasons.append("text_or_layout_construct")
    if profile["formula_width"] > 550:
        reasons.append("formula_over_550px_wide")
    if profile["formula_height"] > 90:
        reasons.append("formula_over_90px_tall")
    if profile["commands"] > 8:
        reasons.append("over_8_commands")
    if profile["brace_depth"] is None:
        reasons.append("unbalanced_braces")
    elif profile["brace_depth"] > 2:
        reasons.append("brace_depth_over_2")
    if profile["scripts"] > 5:
        reasons.append("over_5_scripts")
    if profile["fractions"] > 1:
        reasons.append("over_1_fraction")
    if profile["roots"] > 1:
        reasons.append("over_1_root")
    if profile["large_operators"] > 1:
        reasons.append("over_1_large_operator")

    return reasons, profile


def main() -> None:
    args = parse_args()
    if args.sample_size < 0:
        raise ValueError("--sample-size must be 0 or greater")

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Layer 1 manifest not found: {input_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(input_path)
    candidates = []
    reason_counts: Counter[str] = Counter()
    for record in records:
        reasons, profile = exclusion_reasons(record)
        reason_counts.update(reasons)
        if not reasons:
            candidates.append((record, profile))

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
    for record, profile in selected:
        source_image = (input_path.parent / record["image"]).resolve()
        if not source_image.is_file():
            raise FileNotFoundError(f"Source image not found: {source_image}")
        output_record = dict(record)
        output_record["image"] = os.path.relpath(source_image, output_dir)
        output_record["subset"] = "strict-simple-single-line-v1"
        output_record["strict_simple_single_line"] = True
        output_record["simple_profile"] = profile
        output_records.append(output_record)

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, output_records)
    summary = {
        "schema_version": 1,
        "subset": "strict-simple-single-line-v1",
        "selection_independent_of_model_predictions": True,
        "source_manifest": str(input_path),
        "source_records": len(records),
        "eligible_records": len(candidates),
        "selected_records": len(output_records),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "thresholds": {
            "max_latex_chars": 60,
            "max_formula_width": 550,
            "max_formula_height": 90,
            "max_commands": 8,
            "max_brace_depth": 2,
            "max_scripts": 5,
            "max_fractions": 1,
            "max_roots": 1,
            "max_large_operators": 1,
            "forbid_text_and_layout_constructs": True,
            "require_broad_v0_1_scope": True,
        },
        "excluded_reason_counts": dict(sorted(reason_counts.items())),
        "manifest": str(manifest_path),
        "image_storage": "References Layer 1 images; no images are duplicated.",
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== strict simple-single-line subset ===")
    print(f"Source records: {len(records)}")
    print(f"Eligible records: {len(candidates)}")
    print(f"Selected records: {len(output_records)}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
