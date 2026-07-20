#!/usr/bin/env python3
"""Choose a fixed foreground height with variable 32-multiple input shapes."""

import argparse
import csv
import importlib.metadata
import json
import math
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
from pix2tex.dataset.transforms import test_transform
from pix2tex.utils import post_process

from compare_pix2tex_preprocessing import crop_foreground
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
EVALUATION_DIR = SCRIPT_DIR / "datasets" / "evaluation"
DEFAULT_LAYER1 = EVALUATION_DIR / "layer1a-simple-single-line" / "manifest.jsonl"
DEFAULT_LAYER2 = EVALUATION_DIR / "layer2-screenshot-styles" / "manifest.jsonl"
DEFAULT_LAYER3 = EVALUATION_DIR / "layer3-real-screenshots" / "manifest.jsonl"
DEFAULT_DIAGNOSTIC = SCRIPT_DIR / "test-output" / "pix2tex-strategy-comparison.json"
DEFAULT_LAYER1_BASELINE = SCRIPT_DIR / "test-output" / "layer1-pix2tex-evaluation.json"
DEFAULT_LAYER3_BASELINE = SCRIPT_DIR / "test-output" / "layer3-pix2tex-evaluation.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "test-output" / "pix2tex-target-height-comparison.json"
DEFAULT_CSV = SCRIPT_DIR / "test-output" / "pix2tex-target-height-comparison.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan fixed foreground heights while padding only to the nearest "
            "32-multiple tensor shape across Layer 1, Layer 2, and Layer 3."
        )
    )
    parser.add_argument("--layer1-manifest", type=Path, default=DEFAULT_LAYER1)
    parser.add_argument("--layer2-manifest", type=Path, default=DEFAULT_LAYER2)
    parser.add_argument("--layer3-manifest", type=Path, default=DEFAULT_LAYER3)
    parser.add_argument("--diagnostic-report", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument(
        "--layer1-baseline", type=Path, default=DEFAULT_LAYER1_BASELINE
    )
    parser.add_argument(
        "--layer3-baseline", type=Path, default=DEFAULT_LAYER3_BASELINE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--target-heights", type=int, nargs="+", default=(24, 28, 32, 36, 40)
    )
    parser.add_argument("--max-width", type=int, default=672)
    parser.add_argument("--max-height", type=int, default=192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--limit-diagnostic-per-group",
        type=int,
        default=None,
        help="Limit each diagnostic group for a smoke test.",
    )
    parser.add_argument("--skip-full", action="store_true")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def next_multiple_32(value: int) -> int:
    return max(32, math.ceil(value / 32) * 32)


def prepare_target_height(
    image: Image.Image,
    target_height: int,
    max_width: int,
    max_height: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    foreground = crop_foreground(image)
    source_width, source_height = foreground.size
    target_scale = target_height / source_height
    fit_scale = min(max_width / source_width, max_height / source_height)
    scale = min(target_scale, fit_scale)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = foreground.resize(
        (resized_width, resized_height), Image.Resampling.LANCZOS
    )
    canvas_width = next_multiple_32(resized_width)
    canvas_height = next_multiple_32(resized_height)
    if canvas_width > max_width or canvas_height > max_height:
        raise RuntimeError(
            f"Prepared shape {canvas_height}x{canvas_width} exceeds limits"
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
        "requested_foreground_height": target_height,
        "source_foreground_width": source_width,
        "source_foreground_height": source_height,
        "actual_foreground_width": resized_width,
        "actual_foreground_height": resized_height,
        "scale": scale,
        "fit_limited": fit_scale < target_scale,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "tensor_shape_nchw": list(transformed.shape),
        "placement": "top_left",
    }


def load_report_by_id(path: Path) -> dict[str, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    report = json.loads(resolved.read_text(encoding="utf-8"))
    return {
        record["id"]: record
        for record in report["samples"]
        if record.get("error") is None
    }


def make_sample(
    record: dict[str, Any], manifest_dir: Path, evaluation_group: str
) -> dict[str, Any]:
    return {
        "record": record,
        "manifest_dir": manifest_dir,
        "evaluation_group": evaluation_group,
    }


def build_samples(
    layer1_path: Path,
    layer2_path: Path,
    layer3_path: Path,
    diagnostic_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer1 = load_manifest(layer1_path)
    layer1_ids = {record["id"] for record in layer1}
    layer2 = [
        record
        for record in load_manifest(layer2_path)
        if record.get("derived_from") in layer1_ids
    ]
    layer3 = [
        record
        for record in load_manifest(layer3_path)
        if expected_label(record)[0] is not None
    ]

    diagnostic_report = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    diagnostic_ids = {
        record["id"]
        for record in diagnostic_report["samples"]
        if record.get("config") == "baseline"
    }
    diagnostic = []
    full = []
    for record in layer1:
        sample = make_sample(record, layer1_path.parent, "layer1")
        full.append(sample)
        if record["id"] in diagnostic_ids:
            diagnostic.append(sample)
    for record in layer2:
        variant = record["transform"]["variant"]
        sample = make_sample(record, layer2_path.parent, f"layer2_{variant}")
        full.append(sample)
        if record["derived_from"] in diagnostic_ids:
            diagnostic.append(sample)
    for record in layer3:
        group = (
            "layer3_in_scope"
            if record.get("in_v0_1_scope") is True
            else "layer3_out_of_scope"
        )
        sample = make_sample(record, layer3_path.parent, group)
        full.append(sample)
        diagnostic.append(sample)
    return diagnostic, full


def limit_per_group(
    samples: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None:
        return samples
    counts: dict[str, int] = {}
    selected = []
    for sample in samples:
        group = sample["evaluation_group"]
        count = counts.get(group, 0)
        if count >= limit:
            continue
        counts[group] = count + 1
        selected.append(sample)
    return selected


def evaluate_sample(
    sample: dict[str, Any],
    target_height: int,
    model: LatexOCR,
    device: torch.device,
    max_width: int,
    max_height: int,
    temperature: float,
    seed: int,
    baseline: dict[str, dict[str, Any]],
    phase: str,
) -> dict[str, Any]:
    record = sample["record"]
    image_path = (sample["manifest_dir"] / record["image"]).resolve()
    deterministic_seed = sample_seed(seed, record["id"])
    torch.manual_seed(deterministic_seed)
    with Image.open(image_path) as image:
        tensor, preprocessing = prepare_target_height(
            image_to_rgb(image), target_height, max_width, max_height
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
    delta = None
    if baseline_record is not None:
        delta = (
            score["character_edit_similarity"]
            - baseline_record["character_edit_similarity"]
        )
    return {
        "phase": phase,
        "config": f"target_height_{target_height}",
        "evaluation_group": sample["evaluation_group"],
        "id": record["id"],
        "layer": record.get("layer"),
        "variant": record.get("transform", {}).get("variant"),
        "derived_from": record.get("derived_from"),
        "source_filename": record.get("source_filename"),
        "in_v0_1_scope": record.get("in_v0_1_scope"),
        "image": str(image_path),
        "seed": deterministic_seed,
        **preprocessing,
        **score,
        "character_similarity_delta_from_baseline": delta,
        "inference_seconds": inference_seconds,
        "error": None,
    }


def run_height(
    samples: list[dict[str, Any]],
    phase: str,
    target_height: int,
    model: LatexOCR,
    device: torch.device,
    max_width: int,
    max_height: int,
    temperature: float,
    seed: int,
    baseline: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for index, sample in enumerate(samples, start=1):
        result = evaluate_sample(
            sample=sample,
            target_height=target_height,
            model=model,
            device=device,
            max_width=max_width,
            max_height=max_height,
            temperature=temperature,
            seed=seed,
            baseline=baseline,
            phase=phase,
        )
        results.append(result)
        if index == 1 or index % 50 == 0 or index == len(samples):
            print(
                f"[{phase} h={target_height} {index}/{len(samples)}] "
                f"{result['id']} {result['inference_seconds']:.3f}s "
                f"char={result['character_edit_similarity']:.3f} "
                f"shape={result['canvas_height']}x{result['canvas_width']}"
            )
    return results


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"records": 0}
    latencies = np.array(
        [record["inference_seconds"] for record in records], dtype=np.float64
    )
    deltas = [
        record["character_similarity_delta_from_baseline"]
        for record in records
        if record["character_similarity_delta_from_baseline"] is not None
    ]
    result = {
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
        "length_expansion_2x": sum(record["runaway_2x"] for record in records),
        "fit_limited": sum(record["fit_limited"] for record in records),
        "latency_seconds": {
            "mean": float(latencies.mean()),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(latencies.max()),
        },
    }
    if deltas:
        result["baseline_comparison"] = {
            "records": len(deltas),
            "mean_character_similarity_delta": float(np.mean(deltas)),
            "improved": sum(delta > 0.05 for delta in deltas),
            "degraded": sum(delta < -0.05 for delta in deltas),
        }
    return result


def grouped_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({record["evaluation_group"] for record in records})
    return {
        "all": aggregate(records),
        "by_group": {
            group: aggregate(
                [record for record in records if record["evaluation_group"] == group]
            )
            for group in groups
        },
    }


def choose_best(metrics: dict[int, dict[str, Any]]) -> int:
    selection_groups = (
        "layer1",
        "layer2_light",
        "layer2_dark",
        "layer2_degraded",
        "layer3_in_scope",
    )

    def key(height: int) -> tuple[Any, ...]:
        group_metrics = metrics[height]["by_group"]
        supported = [group_metrics[group] for group in selection_groups]
        expansions = sum(group["length_expansion_2x"] for group in supported)
        similarities = [group["mean_character_edit_similarity"] for group in supported]
        macro_mean = float(np.mean(similarities))
        return (
            -expansions,
            min(similarities),
            macro_mean,
            -metrics[height]["all"]["latency_seconds"]["p95"],
        )

    return max(metrics, key=key)


def shape_counts(records: list[dict[str, Any]]) -> list[dict[str, int]]:
    counts: dict[tuple[int, int], int] = {}
    for record in records:
        shape = (record["canvas_height"], record["canvas_width"])
        counts[shape] = counts.get(shape, 0) + 1
    return [
        {"height": height, "width": width, "records": count}
        for (height, width), count in sorted(counts.items())
    ]


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "phase",
        "config",
        "evaluation_group",
        "id",
        "layer",
        "variant",
        "derived_from",
        "source_filename",
        "requested_foreground_height",
        "actual_foreground_height",
        "actual_foreground_width",
        "canvas_height",
        "canvas_width",
        "fit_limited",
        "normalized_exact_match",
        "character_edit_similarity",
        "token_edit_similarity",
        "prediction_length_ratio",
        "runaway_2x",
        "character_similarity_delta_from_baseline",
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
    heights = sorted(set(args.target_heights))
    if not heights or any(height < 1 or height > args.max_height for height in heights):
        raise ValueError("Target heights must be within the maximum height")
    if args.max_width < 32 or args.max_height < 32:
        raise ValueError("Maximum dimensions must be at least 32")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")
    if (
        args.limit_diagnostic_per_group is not None
        and args.limit_diagnostic_per_group < 1
    ):
        raise ValueError("--limit-diagnostic-per-group must be at least 1")

    paths = [
        args.layer1_manifest.expanduser().resolve(),
        args.layer2_manifest.expanduser().resolve(),
        args.layer3_manifest.expanduser().resolve(),
        args.diagnostic_report.expanduser().resolve(),
        args.layer1_baseline.expanduser().resolve(),
        args.layer3_baseline.expanduser().resolve(),
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    diagnostic, full = build_samples(paths[0], paths[1], paths[2], paths[3])
    diagnostic = limit_per_group(diagnostic, args.limit_diagnostic_per_group)
    baseline = {
        **load_report_by_id(paths[4]),
        **load_report_by_id(paths[5]),
    }
    print(f"Diagnostic samples: {len(diagnostic)}; full samples: {len(full)}")

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
    for height in heights:
        print(f"\n=== diagnostic target height {height} ===")
        results = run_height(
            samples=diagnostic,
            phase="diagnostic_scan",
            target_height=height,
            model=model,
            device=device,
            max_width=args.max_width,
            max_height=args.max_height,
            temperature=args.temperature,
            seed=args.seed,
            baseline=baseline,
        )
        all_results.extend(results)
        scan_metrics[height] = grouped_metrics(results)
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

    selected_height = choose_best(scan_metrics)
    print(f"\nSelected target foreground height: {selected_height}")
    full_metrics = None
    selected_shapes = []
    if not args.skip_full:
        print(f"\n=== full validation target height {selected_height} ===")
        full_results = run_height(
            samples=full,
            phase="full_validation",
            target_height=selected_height,
            model=model,
            device=device,
            max_width=args.max_width,
            max_height=args.max_height,
            temperature=args.temperature,
            seed=args.seed,
            baseline=baseline,
        )
        all_results.extend(full_results)
        full_metrics = grouped_metrics(full_results)
        selected_shapes = shape_counts(full_results)

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
            "layer1_manifest": str(paths[0]),
            "layer2_manifest": str(paths[1]),
            "layer3_manifest": str(paths[2]),
            "diagnostic_report": str(paths[3]),
            "diagnostic_samples": len(diagnostic),
            "full_samples": 0 if args.skip_full else len(full),
        },
        "preprocessing_contract": {
            "foreground_crop": "pix2tex normalization, exact pixels < 250 bbox",
            "target_foreground_height": selected_height,
            "aspect_ratio_preserved": True,
            "interpolation": "Lanczos",
            "placement": "top_left",
            "background": "white",
            "dimension_rounding": "ceil each dimension to nearest multiple of 32",
            "max_dimensions_width_height": [args.max_width, args.max_height],
            "normalization": "pix2tex test_transform first channel",
            "decoder_max_tokens": args.max_tokens,
            "observed_shapes_hw": selected_shapes,
        },
        "candidate_target_foreground_heights": heights,
        "selection_rule": (
            "fewest total 2x length expansions across supported groups, then "
            "highest worst-group character similarity, then highest macro mean, "
            "then lower P95 latency"
        ),
        "selected_target_foreground_height": selected_height,
        "scan_metrics": scan_metrics,
        "full_validation_metrics": full_metrics,
        "samples": all_results,
        "notes": [
            "Selection groups are Layer 1, Layer 2 light/dark/degraded, and Layer 3 in-scope.",
            "Layer 3 out-of-scope samples are evaluated but do not choose the height.",
            "String similarity is diagnostic, not mathematical-semantic accuracy.",
        ],
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = args.csv.expanduser().resolve()
    write_csv(csv_path, all_results)
    partial_path.unlink(missing_ok=True)

    print("\n=== target height comparison complete ===")
    for height in heights:
        groups = scan_metrics[height]["by_group"]
        supported = (
            "layer1",
            "layer2_light",
            "layer2_dark",
            "layer2_degraded",
            "layer3_in_scope",
        )
        chars = [groups[group]["mean_character_edit_similarity"] for group in supported]
        expansions = sum(groups[group]["length_expansion_2x"] for group in supported)
        print(
            f"h={height}: worst-char {min(chars):.3f}, "
            f"macro-char {float(np.mean(chars)):.3f}, expansions {expansions}"
        )
    if full_metrics is not None:
        print(f"full h={selected_height}:")
        for group, metrics in full_metrics["by_group"].items():
            print(
                f"  {group}: char {metrics['mean_character_edit_similarity']:.3f}, "
                f">=0.8 {metrics['character_similarity_ge_0_8']}/{metrics['records']}, "
                f"length>=2x {metrics['length_expansion_2x']}/{metrics['records']}"
            )
    print(f"JSON: {output_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
