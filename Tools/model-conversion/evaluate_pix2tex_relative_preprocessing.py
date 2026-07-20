#!/usr/bin/env python3
"""Evaluate the selected pix2tex relative-scale preprocessing contract."""

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from typing import Any
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")

import numpy as np
import torch
from PIL import Image
from pix2tex.cli import LatexOCR, token2str
from pix2tex.utils import post_process

from compare_pix2tex_preprocessing import prepare_relative_scale
from compare_pix2tex_strategies import score_prediction
from evaluate_pix2tex import (
    configure_device,
    expected_label,
    image_to_rgb,
    load_manifest,
    sample_seed,
    synchronize,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate relative-scale pix2tex preprocessing on an EqnSnap manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--source-subset-manifest",
        type=Path,
        default=None,
        help="Keep only records whose derived_from ID exists in this manifest.",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=None,
        help="Optional existing report with matching sample IDs for deltas.",
    )
    parser.add_argument("--relative-scale", type=float, default=0.75)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_baseline(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Baseline report not found: {resolved}")
    report = json.loads(resolved.read_text(encoding="utf-8"))
    return {
        record["id"]: record
        for record in report["samples"]
        if record.get("error") is None
    }


def filter_source_subset(
    records: list[dict[str, Any]], source_manifest: Path | None
) -> tuple[list[dict[str, Any]], int | None]:
    if source_manifest is None:
        return records, None
    resolved = source_manifest.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Source subset manifest not found: {resolved}")
    source_ids = {record["id"] for record in load_manifest(resolved)}
    filtered = [record for record in records if record.get("derived_from") in source_ids]
    return filtered, len(source_ids)


def evaluate_record(
    record: dict[str, Any],
    manifest_dir: Path,
    model: LatexOCR,
    device: torch.device,
    relative_scale: float,
    temperature: float,
    seed: int,
    baseline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected, label_source = expected_label(record)
    if expected is None:
        raise RuntimeError(f"Sample has no expected label: {record['id']}")
    image_path = (manifest_dir / record["image"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    deterministic_seed = sample_seed(seed, record["id"])
    torch.manual_seed(deterministic_seed)
    with Image.open(image_path) as image:
        tensor, preprocessing = prepare_relative_scale(
            image_to_rgb(image),
            relative_scale,
            model.args.max_dimensions,
            model.args.min_dimensions,
        )
    tensor = tensor.to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        decoded = model.model.generate(tensor, temperature=temperature)
    synchronize(device)
    inference_seconds = time.perf_counter() - started
    prediction = post_process(token2str(decoded, model.tokenizer)[0])
    score = score_prediction(record, prediction, model)

    baseline_record = baseline.get(record["id"])
    baseline_delta = None
    matches_baseline = None
    if baseline_record is not None:
        baseline_delta = (
            score["character_edit_similarity"]
            - baseline_record["character_edit_similarity"]
        )
        matches_baseline = prediction == baseline_record["predicted_latex"]

    return {
        "id": record["id"],
        "layer": record.get("layer"),
        "variant": record.get("transform", {}).get("variant"),
        "derived_from": record.get("derived_from"),
        "source_filename": record.get("source_filename"),
        "in_v0_1_scope": record.get("in_v0_1_scope"),
        "label_source": label_source,
        "image": str(image_path),
        "seed": deterministic_seed,
        **preprocessing,
        **score,
        "character_similarity_delta_from_baseline": baseline_delta,
        "matches_baseline_prediction": matches_baseline,
        "inference_seconds": inference_seconds,
        "notes": record.get("notes", ""),
        "error": None,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("error") is None]
    if not successful:
        return {"records": len(records), "successful": 0, "failed": len(records)}
    latencies = np.array(
        [record["inference_seconds"] for record in successful], dtype=np.float64
    )
    deltas = [
        record["character_similarity_delta_from_baseline"]
        for record in successful
        if record["character_similarity_delta_from_baseline"] is not None
    ]
    metrics = {
        "records": len(records),
        "successful": len(successful),
        "failed": len(records) - len(successful),
        "normalized_exact_matches": sum(
            record["normalized_exact_match"] for record in successful
        ),
        "mean_character_edit_similarity": float(
            np.mean([record["character_edit_similarity"] for record in successful])
        ),
        "mean_token_edit_similarity": float(
            np.mean([record["token_edit_similarity"] for record in successful])
        ),
        "character_similarity_ge_0_9": sum(
            record["character_edit_similarity"] >= 0.9 for record in successful
        ),
        "character_similarity_ge_0_8": sum(
            record["character_edit_similarity"] >= 0.8 for record in successful
        ),
        "character_similarity_lt_0_2": sum(
            record["character_edit_similarity"] < 0.2 for record in successful
        ),
        "length_expansion_2x": sum(record["runaway_2x"] for record in successful),
        "latency_seconds": {
            "mean": float(latencies.mean()),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(latencies.max()),
        },
    }
    if deltas:
        metrics["baseline_comparison"] = {
            "records": len(deltas),
            "mean_character_similarity_delta": float(np.mean(deltas)),
            "improved": sum(delta > 0.05 for delta in deltas),
            "degraded": sum(delta < -0.05 for delta in deltas),
        }
    return metrics


def shape_counts(records: list[dict[str, Any]]) -> list[dict[str, int]]:
    counts: dict[tuple[int, int], int] = {}
    for record in records:
        if record.get("error") is not None:
            continue
        shape = (record["canvas_height"], record["canvas_width"])
        counts[shape] = counts.get(shape, 0) + 1
    return [
        {"height": height, "width": width, "records": count}
        for (height, width), count in sorted(counts.items())
    ]


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "id",
        "layer",
        "variant",
        "derived_from",
        "source_filename",
        "in_v0_1_scope",
        "canvas_height",
        "canvas_width",
        "actual_foreground_height",
        "actual_foreground_width",
        "normalized_exact_match",
        "character_edit_similarity",
        "token_edit_similarity",
        "prediction_length_ratio",
        "runaway_2x",
        "character_similarity_delta_from_baseline",
        "matches_baseline_prediction",
        "inference_seconds",
        "expected_latex",
        "predicted_latex",
        "image",
        "notes",
        "error",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_partial(
    path: Path, manifest_path: Path, completed: list[dict[str, Any]], total: int
) -> None:
    path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "completed": len(completed),
                "total": total,
                "metrics": aggregate(completed),
                "samples": completed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.relative_scale <= 0:
        raise ValueError("--relative-scale must be greater than 0")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")

    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    all_manifest_records = load_manifest(manifest_path)
    records, source_subset_size = filter_source_subset(
        all_manifest_records, args.source_subset_manifest
    )
    records = [record for record in records if expected_label(record)[0] is not None]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No labeled records selected")

    baseline = load_baseline(args.baseline_report)
    device = torch.device(args.device)
    print(f"Loading pix2tex on {device}...")
    model = LatexOCR()
    configure_device(model, device)
    model.args.max_seq_len = args.max_tokens

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.json")
    results = []
    for index, record in enumerate(records, start=1):
        try:
            result = evaluate_record(
                record=record,
                manifest_dir=manifest_path.parent,
                model=model,
                device=device,
                relative_scale=args.relative_scale,
                temperature=args.temperature,
                seed=args.seed,
                baseline=baseline,
            )
        except Exception as error:
            expected, label_source = expected_label(record)
            result = {
                "id": record["id"],
                "layer": record.get("layer"),
                "variant": record.get("transform", {}).get("variant"),
                "derived_from": record.get("derived_from"),
                "source_filename": record.get("source_filename"),
                "in_v0_1_scope": record.get("in_v0_1_scope"),
                "label_source": label_source,
                "expected_latex": expected,
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        if index == 1 or index % 25 == 0 or index == len(records):
            if result.get("error") is None:
                print(
                    f"[{index}/{len(records)}] {record['id']} "
                    f"{result['inference_seconds']:.3f}s "
                    f"char={result['character_edit_similarity']:.3f} "
                    f"shape={result['canvas_height']}x{result['canvas_width']}"
                )
            else:
                print(f"[{index}/{len(records)}] {record['id']} failed")
        if index % args.checkpoint_every == 0:
            write_partial(partial_path, manifest_path, results, len(records))

    variants = sorted(
        {record["variant"] for record in results if record.get("variant") is not None}
    )
    in_scope = [record for record in results if record.get("in_v0_1_scope") is True]
    out_of_scope = [
        record for record in results if record.get("in_v0_1_scope") is False
    ]
    report = {
        "report_schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pix2tex": package_version("pix2tex"),
            "torch": torch.__version__,
            "device": str(device),
            "seed": args.seed,
            "temperature": args.temperature,
        },
        "input": {
            "manifest": str(manifest_path),
            "manifest_records": len(all_manifest_records),
            "source_subset_manifest": (
                None
                if args.source_subset_manifest is None
                else str(args.source_subset_manifest.expanduser().resolve())
            ),
            "source_subset_records": source_subset_size,
            "evaluated_records": len(records),
            "baseline_report": (
                None
                if args.baseline_report is None
                else str(args.baseline_report.expanduser().resolve())
            ),
        },
        "preprocessing_contract": {
            "relative_scale": args.relative_scale,
            "dimension_multiple": 32,
            "placement": "top_left",
            "background": "white",
            "interpolation": "Lanczos",
            "max_dimensions_width_height": [672, 192],
            "min_dimensions_width_height": [32, 32],
            "decoder_max_tokens": args.max_tokens,
        },
        "metrics": {
            "all": aggregate(results),
            "v0_1_in_scope": aggregate(in_scope),
            "out_of_scope": aggregate(out_of_scope),
            "by_variant": {
                variant: aggregate(
                    [record for record in results if record.get("variant") == variant]
                )
                for variant in variants
            },
        },
        "observed_shapes_hw": shape_counts(results),
        "samples": results,
        "notes": [
            "String similarity is diagnostic, not mathematical-semantic accuracy.",
            "Layer 3 still requires human review of changed predictions.",
        ],
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = args.csv.expanduser().resolve()
    write_csv(csv_path, results)
    partial_path.unlink(missing_ok=True)

    metrics = report["metrics"]["v0_1_in_scope"]
    print("\n=== relative preprocessing evaluation ===")
    print(f"Evaluated: {len(records)}")
    print(f"v0.1 in-scope: {metrics['records']}")
    print(
        f"mean char/token: "
        f"{metrics.get('mean_character_edit_similarity', 0):.3f} / "
        f"{metrics.get('mean_token_edit_similarity', 0):.3f}"
    )
    print(
        f"char >= 0.8: {metrics.get('character_similarity_ge_0_8', 0)}/"
        f"{metrics['records']}"
    )
    print(
        f"length >= 2x: {metrics.get('length_expansion_2x', 0)}/"
        f"{metrics['records']}"
    )
    print(f"JSON: {output_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
