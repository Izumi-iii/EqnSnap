#!/usr/bin/env python3
"""Find a fixed Core ML foreground height for pix2tex preprocessing."""

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
from pix2tex.cli import LatexOCR, minmax_size, pad, token2str
from pix2tex.dataset.transforms import test_transform
from pix2tex.utils import post_process

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
DEFAULT_MANIFEST = (
    SCRIPT_DIR
    / "datasets"
    / "evaluation"
    / "layer1a-simple-single-line"
    / "manifest.jsonl"
)
DEFAULT_DIAGNOSTIC_REPORT = (
    SCRIPT_DIR / "test-output" / "pix2tex-strategy-comparison.json"
)
DEFAULT_BASELINE_REPORT = (
    SCRIPT_DIR / "test-output" / "layer1-pix2tex-evaluation.json"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "test-output" / "pix2tex-preprocessing-comparison.json"
)
DEFAULT_CSV = (
    SCRIPT_DIR / "test-output" / "pix2tex-preprocessing-comparison.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed foreground heights on a 448x64 canvas, select the "
            "best height, then validate it on the full strict subset."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--diagnostic-report", type=Path, default=DEFAULT_DIAGNOSTIC_REPORT
    )
    parser.add_argument(
        "--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--target-heights",
        type=int,
        nargs="+",
        default=(16, 18, 20, 22, 24, 26, 28),
    )
    parser.add_argument("--canvas-width", type=int, default=448)
    parser.add_argument("--canvas-height", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-diagnostic",
        type=int,
        default=None,
        help="Limit diagnostic samples for a smoke test.",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Only scan diagnostic samples; do not run the full subset.",
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def crop_foreground(image: Image.Image) -> Image.Image:
    normalized = pad(image)
    pixels = np.array(normalized, dtype=np.uint8)
    rows, columns = np.nonzero(pixels < 250)
    if len(rows) == 0:
        raise RuntimeError("Image contains no visible formula foreground")
    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max()) + 1
    bottom = int(rows.max()) + 1
    return normalized.crop((left, top, right, bottom))


def prepare_coreml_canvas(
    image: Image.Image,
    target_height: int,
    canvas_width: int,
    canvas_height: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    foreground = crop_foreground(image)
    source_width, source_height = foreground.size
    target_scale = target_height / source_height
    width_scale = canvas_width / source_width
    scale = min(target_scale, width_scale, canvas_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = foreground.resize(
        (resized_width, resized_height), Image.Resampling.LANCZOS
    )
    canvas = Image.new("L", (canvas_width, canvas_height), 255)
    canvas.paste(resized, (0, 0))
    transformed = test_transform(
        image=np.array(canvas.convert("RGB"))
    )["image"][:1].unsqueeze(0).contiguous()
    expected_shape = (1, 1, canvas_height, canvas_width)
    if tuple(transformed.shape) != expected_shape:
        raise RuntimeError(
            f"Expected tensor shape {expected_shape}, got {tuple(transformed.shape)}"
        )
    return transformed, {
        "source_foreground_width": source_width,
        "source_foreground_height": source_height,
        "requested_foreground_height": target_height,
        "actual_foreground_width": resized_width,
        "actual_foreground_height": resized_height,
        "scale": scale,
        "width_limited": width_scale < target_scale,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "placement": "top_left",
        "tensor_shape_nchw": list(transformed.shape),
    }


def prepare_relative_scale(
    image: Image.Image,
    relative_scale: float,
    max_dimensions: list[int],
    min_dimensions: list[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    normalized = pad(image)
    scaled_size = tuple(
        max(1, round(dimension * relative_scale))
        for dimension in normalized.size
    )
    scaled = normalized.resize(scaled_size, Image.Resampling.LANCZOS)
    final_image = pad(
        minmax_size(
            pad(scaled),
            max_dimensions=max_dimensions,
            min_dimensions=min_dimensions,
        )
    )
    transformed = test_transform(
        image=np.array(final_image.convert("RGB"))
    )["image"][:1].unsqueeze(0).contiguous()
    pixels = np.array(final_image, dtype=np.uint8)
    rows, columns = np.nonzero(pixels < 250)
    if len(rows) == 0:
        raise RuntimeError("Scaled image contains no visible formula foreground")
    return transformed, {
        "source_foreground_width": normalized.size[0],
        "source_foreground_height": normalized.size[1],
        "requested_foreground_height": None,
        "actual_foreground_width": int(columns.max() - columns.min() + 1),
        "actual_foreground_height": int(rows.max() - rows.min() + 1),
        "scale": relative_scale,
        "width_limited": False,
        "canvas_width": final_image.size[0],
        "canvas_height": final_image.size[1],
        "placement": "top_left",
        "tensor_shape_nchw": list(transformed.shape),
    }


def baseline_by_id(
    report: dict[str, Any], manifest_ids: set[str]
) -> dict[str, dict[str, Any]]:
    result = {
        record["id"]: record
        for record in report["samples"]
        if record["id"] in manifest_ids and record.get("error") is None
    }
    missing = manifest_ids - set(result)
    if missing:
        raise RuntimeError(f"Baseline report is missing {len(missing)} samples")
    return result


def diagnostic_records(
    manifest_by_id: dict[str, dict[str, Any]], report: dict[str, Any]
) -> list[dict[str, Any]]:
    selected = []
    for result in report["samples"]:
        if result.get("config") != "baseline":
            continue
        sample_id = result["id"]
        if sample_id not in manifest_by_id:
            continue
        record = dict(manifest_by_id[sample_id])
        record["diagnostic_group"] = result["diagnostic_group"]
        selected.append(record)
    if not selected:
        raise RuntimeError("Diagnostic report contains no baseline samples")
    return selected


def run_height(
    records: list[dict[str, Any]],
    phase: str,
    target_height: int,
    manifest_dir: Path,
    baseline: dict[str, dict[str, Any]],
    model: LatexOCR,
    device: torch.device,
    canvas_width: int,
    canvas_height: int,
    temperature: float,
    seed: int,
) -> list[dict[str, Any]]:
    results = []
    for index, record in enumerate(records, start=1):
        image_path = (manifest_dir / record["image"]).resolve()
        deterministic_seed = sample_seed(seed, record["id"])
        torch.manual_seed(deterministic_seed)
        with Image.open(image_path) as image:
            tensor, preprocessing = prepare_coreml_canvas(
                image_to_rgb(image),
                target_height,
                canvas_width,
                canvas_height,
            )
        tensor = tensor.to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            decoded = model.model.generate(tensor, temperature=temperature)
        synchronize(device)
        inference_seconds = time.perf_counter() - started
        prediction = post_process(token2str(decoded, model.tokenizer)[0])
        score = score_prediction(record, prediction, model)
        baseline_result = baseline[record["id"]]
        results.append(
            {
                "phase": phase,
                "config": f"foreground_{target_height}",
                "id": record["id"],
                "diagnostic_group": record.get(
                    "diagnostic_group", "all_subset"
                ),
                "image": str(image_path),
                "seed": deterministic_seed,
                **preprocessing,
                **score,
                "character_similarity_delta": score[
                    "character_edit_similarity"
                ]
                - baseline_result["character_edit_similarity"],
                "matches_baseline_prediction": prediction
                == baseline_result["predicted_latex"],
                "inference_seconds": inference_seconds,
                "error": None,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(records):
            print(
                f"[{phase} h={target_height} {index}/{len(records)}] "
                f"{record['id']} {inference_seconds:.3f}s "
                f"char={score['character_edit_similarity']:.3f} "
                f"actual_h={preprocessing['actual_foreground_height']}"
            )
    return results


def run_relative_scale(
    records: list[dict[str, Any]],
    relative_scale: float,
    manifest_dir: Path,
    baseline: dict[str, dict[str, Any]],
    model: LatexOCR,
    device: torch.device,
    temperature: float,
    seed: int,
) -> list[dict[str, Any]]:
    results = []
    for index, record in enumerate(records, start=1):
        image_path = (manifest_dir / record["image"]).resolve()
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
        baseline_result = baseline[record["id"]]
        results.append(
            {
                "phase": "full_relative_scale",
                "config": f"relative_scale_{relative_scale}",
                "id": record["id"],
                "diagnostic_group": "all_subset",
                "image": str(image_path),
                "seed": deterministic_seed,
                **preprocessing,
                **score,
                "character_similarity_delta": score[
                    "character_edit_similarity"
                ]
                - baseline_result["character_edit_similarity"],
                "matches_baseline_prediction": prediction
                == baseline_result["predicted_latex"],
                "inference_seconds": inference_seconds,
                "error": None,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(records):
            print(
                f"[relative scale={relative_scale} {index}/{len(records)}] "
                f"{record['id']} {inference_seconds:.3f}s "
                f"char={score['character_edit_similarity']:.3f} "
                f"shape={tuple(tensor.shape[-2:])}"
            )
    return results


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"records": 0}
    latencies = np.array(
        [record["inference_seconds"] for record in records], dtype=np.float64
    )
    return {
        "records": len(records),
        "normalized_exact_matches": sum(
            record["normalized_exact_match"] for record in records
        ),
        "mean_character_edit_similarity": float(
            np.mean([record["character_edit_similarity"] for record in records])
        ),
        "mean_token_edit_similarity": float(
            np.mean([record["token_edit_similarity"] for record in records])
        ),
        "character_similarity_ge_0_9": sum(
            record["character_edit_similarity"] >= 0.9 for record in records
        ),
        "character_similarity_ge_0_8": sum(
            record["character_edit_similarity"] >= 0.8 for record in records
        ),
        "character_similarity_lt_0_2": sum(
            record["character_edit_similarity"] < 0.2 for record in records
        ),
        "length_expansion_2x": sum(
            record["runaway_2x"] for record in records
        ),
        "improved_over_baseline": sum(
            record["character_similarity_delta"] > 0.05 for record in records
        ),
        "degraded_from_baseline": sum(
            record["character_similarity_delta"] < -0.05 for record in records
        ),
        "width_limited": sum(record["width_limited"] for record in records),
        "mean_actual_foreground_height": float(
            np.mean([record["actual_foreground_height"] for record in records])
        ),
        "latency_seconds": {
            "mean": float(latencies.mean()),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(latencies.max()),
        },
    }


def grouped_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({record["diagnostic_group"] for record in records})
    return {
        "all": aggregate(records),
        "by_diagnostic_group": {
            group: aggregate(
                [record for record in records if record["diagnostic_group"] == group]
            )
            for group in groups
        },
    }


def choose_best(metrics: dict[int, dict[str, Any]]) -> int:
    return max(
        metrics,
        key=lambda height: (
            -metrics[height]["all"]["length_expansion_2x"],
            metrics[height]["all"]["mean_character_edit_similarity"],
            metrics[height]["all"]["character_similarity_ge_0_8"],
            -metrics[height]["all"]["latency_seconds"]["p95"],
        ),
    )


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "phase",
        "config",
        "diagnostic_group",
        "id",
        "requested_foreground_height",
        "actual_foreground_width",
        "actual_foreground_height",
        "source_foreground_width",
        "source_foreground_height",
        "scale",
        "width_limited",
        "canvas_width",
        "canvas_height",
        "tensor_shape_nchw",
        "normalized_exact_match",
        "character_edit_similarity",
        "token_edit_similarity",
        "prediction_length_ratio",
        "runaway_2x",
        "character_similarity_delta",
        "inference_seconds",
        "expected_latex",
        "predicted_latex",
        "image",
        "error",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.canvas_width < 1 or args.canvas_height < 1:
        raise ValueError("Canvas dimensions must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    heights = sorted(set(args.target_heights))
    if not heights or any(height < 1 or height > args.canvas_height for height in heights):
        raise ValueError("Target heights must be within the canvas height")
    if args.limit_diagnostic is not None and args.limit_diagnostic < 1:
        raise ValueError("--limit-diagnostic must be at least 1")

    manifest_path = args.manifest.expanduser().resolve()
    diagnostic_path = args.diagnostic_report.expanduser().resolve()
    baseline_path = args.baseline_report.expanduser().resolve()
    for path in (manifest_path, diagnostic_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    manifest = load_manifest(manifest_path)
    manifest_by_id = {record["id"]: record for record in manifest}
    diagnostic_report = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = baseline_by_id(baseline_report, set(manifest_by_id))
    diagnostic = diagnostic_records(manifest_by_id, diagnostic_report)
    if args.limit_diagnostic is not None:
        diagnostic = diagnostic[: args.limit_diagnostic]

    device = torch.device(args.device)
    print(f"Loading pix2tex on {device}...")
    model = LatexOCR()
    configure_device(model, device)
    model.args.max_seq_len = args.max_tokens

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.json")
    all_results = []
    scan_metrics: dict[int, dict[str, Any]] = {}
    for target_height in heights:
        print(f"\n=== diagnostic target foreground height {target_height} ===")
        height_results = run_height(
            records=diagnostic,
            phase="diagnostic_scan",
            target_height=target_height,
            manifest_dir=manifest_path.parent,
            baseline=baseline,
            model=model,
            device=device,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            temperature=args.temperature,
            seed=args.seed,
        )
        all_results.extend(height_results)
        scan_metrics[target_height] = grouped_metrics(height_results)
        partial_path.write_text(
            json.dumps(
                {
                    "completed_target_heights": sorted(scan_metrics),
                    "scan_metrics": scan_metrics,
                    "samples": all_results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    best_height = choose_best(scan_metrics)
    print(f"\nSelected target foreground height: {best_height}")

    full_metrics = None
    relative_scale_metrics = None
    observed_shapes = []
    if not args.skip_full:
        print(f"\n=== full subset target foreground height {best_height} ===")
        full_records = []
        for record in manifest:
            full_record = dict(record)
            full_record["diagnostic_group"] = "all_subset"
            full_records.append(full_record)
        full_results = run_height(
            records=full_records,
            phase="full_validation",
            target_height=best_height,
            manifest_dir=manifest_path.parent,
            baseline=baseline,
            model=model,
            device=device,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            temperature=args.temperature,
            seed=args.seed,
        )
        all_results.extend(full_results)
        full_metrics = grouped_metrics(full_results)

        print("\n=== full subset relative foreground scale 0.75 ===")
        relative_results = run_relative_scale(
            records=full_records,
            relative_scale=0.75,
            manifest_dir=manifest_path.parent,
            baseline=baseline,
            model=model,
            device=device,
            temperature=args.temperature,
            seed=args.seed,
        )
        all_results.extend(relative_results)
        relative_scale_metrics = grouped_metrics(relative_results)
        shape_counts: dict[tuple[int, int], int] = {}
        for record in relative_results:
            shape = (record["canvas_height"], record["canvas_width"])
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        observed_shapes = [
            {"height": height, "width": width, "records": count}
            for (height, width), count in sorted(shape_counts.items())
        ]

    report = {
        "report_schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pix2tex": package_version("pix2tex"),
            "torch": torch.__version__,
            "device": str(device),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "inputs": {
            "manifest": str(manifest_path),
            "diagnostic_report": str(diagnostic_path),
            "baseline_report": str(baseline_path),
            "diagnostic_samples": len(diagnostic),
            "full_samples": 0 if args.skip_full else len(manifest),
        },
        "rejected_single_shape_contract": {
            "canvas_shape_nchw": [1, 1, args.canvas_height, args.canvas_width],
            "foreground_detection": "pix2tex pad normalization, pixels < 250",
            "aspect_ratio_preserved": True,
            "placement": "top_left",
            "background": "white",
            "interpolation": "Lanczos",
            "width_overflow": "scale down to fit canvas width",
            "normalization": "pix2tex test_transform first channel",
            "decoder_max_tokens": args.max_tokens,
        },
        "selected_coreml_preprocessing_contract": {
            "mode": "relative_scale_then_enumerated_shape_padding",
            "relative_foreground_scale": 0.75,
            "aspect_ratio_preserved": True,
            "placement": "top_left",
            "background": "white",
            "interpolation": "Lanczos",
            "dimension_multiple": 32,
            "max_dimensions_width_height": [672, 192],
            "min_dimensions_width_height": [32, 32],
            "normalization": "pix2tex test_transform first channel",
            "decoder_max_tokens": args.max_tokens,
            "observed_shapes_hw": observed_shapes,
        },
        "candidate_target_foreground_heights": heights,
        "selection_rule": (
            "fewest 2x length expansions, then highest mean character similarity, "
            "then most samples >= 0.8, then lower P95 latency"
        ),
        "best_rejected_fixed_target_foreground_height": best_height,
        "selected_target_foreground_height": None,
        "selected_preprocessing_mode": (
            "relative foreground scale 0.75 with dimensions padded to multiples of 32"
        ),
        "scan_metrics": scan_metrics,
        "fixed_height_full_validation_metrics": full_metrics,
        "relative_scale_full_validation_metrics": relative_scale_metrics,
        "samples": all_results,
        "notes": [
            "String similarity is diagnostic, not mathematical-semantic accuracy.",
            "A single 448x64 canvas was rejected because all tested foreground heights regressed badly.",
            "The selected relative-scale rule must still be checked on Layer 2 and Layer 3 screenshots.",
        ],
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = args.csv.expanduser().resolve()
    write_csv(csv_path, all_results)
    partial_path.unlink(missing_ok=True)

    print("\n=== preprocessing comparison complete ===")
    for target_height in heights:
        metrics = scan_metrics[target_height]["all"]
        print(
            f"h={target_height}: char "
            f"{metrics['mean_character_edit_similarity']:.3f}, "
            f">=0.8 {metrics['character_similarity_ge_0_8']}/{metrics['records']}, "
            f"length>=2x {metrics['length_expansion_2x']}/{metrics['records']}, "
            f"width-limited {metrics['width_limited']}/{metrics['records']}"
        )
    if full_metrics is not None:
        metrics = full_metrics["all"]
        print(
            f"full h={best_height}: char "
            f"{metrics['mean_character_edit_similarity']:.3f}, "
            f">=0.8 {metrics['character_similarity_ge_0_8']}/{metrics['records']}, "
            f"length>=2x {metrics['length_expansion_2x']}/{metrics['records']}, "
            f"P95 {metrics['latency_seconds']['p95']:.3f}s"
        )
    if relative_scale_metrics is not None:
        metrics = relative_scale_metrics["all"]
        print(
            f"full relative scale=0.75: char "
            f"{metrics['mean_character_edit_similarity']:.3f}, "
            f">=0.8 {metrics['character_similarity_ge_0_8']}/{metrics['records']}, "
            f"length>=2x {metrics['length_expansion_2x']}/{metrics['records']}, "
            f"P95 {metrics['latency_seconds']['p95']:.3f}s, "
            f"shapes {len(observed_shapes)}"
        )
    print(f"JSON: {output_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
