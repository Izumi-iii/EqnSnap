#!/usr/bin/env python3
"""Test an image-feature rule that selects between two preprocessing profiles."""

import argparse
import csv
import hashlib
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

import cv2
import numpy as np
import torch
from PIL import Image
from pix2tex.cli import LatexOCR

from compare_pix2tex_preprocessing import crop_foreground
from compare_pix2tex_target_height import (
    aggregate,
    build_samples,
    evaluate_sample,
    grouped_metrics,
    load_report_by_id,
    shape_counts,
)
from evaluate_pix2tex import configure_device, image_to_rgb
from stroke_profile_contract import stroke_profile_metrics


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR / "datasets" / "evaluation"
DEFAULT_LAYER1 = EVALUATION_DIR / "layer1a-simple-single-line" / "manifest.jsonl"
DEFAULT_LAYER2 = EVALUATION_DIR / "layer2-screenshot-styles" / "manifest.jsonl"
DEFAULT_LAYER3 = EVALUATION_DIR / "layer3-real-screenshots" / "manifest.jsonl"
DEFAULT_HEIGHT_REPORT = SCRIPT_DIR / "test-output" / "pix2tex-target-height-comparison.json"
DEFAULT_LAYER1_BASELINE = SCRIPT_DIR / "test-output" / "layer1-pix2tex-evaluation.json"
DEFAULT_LAYER3_BASELINE = SCRIPT_DIR / "test-output" / "layer3-pix2tex-evaluation.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "test-output" / "pix2tex-stroke-profile-comparison.json"
DEFAULT_CSV = SCRIPT_DIR / "test-output" / "pix2tex-stroke-profile-comparison.csv"

SUPPORTED_GROUPS = (
    "layer1",
    "layer2_light",
    "layer2_dark",
    "layer2_degraded",
    "layer3_in_scope",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learn an interpretable image-feature threshold that selects a small "
            "or large target-height preprocessing profile."
        )
    )
    parser.add_argument("--layer1-manifest", type=Path, default=DEFAULT_LAYER1)
    parser.add_argument("--layer2-manifest", type=Path, default=DEFAULT_LAYER2)
    parser.add_argument("--layer3-manifest", type=Path, default=DEFAULT_LAYER3)
    parser.add_argument("--height-report", type=Path, default=DEFAULT_HEIGHT_REPORT)
    parser.add_argument(
        "--layer1-baseline", type=Path, default=DEFAULT_LAYER1_BASELINE
    )
    parser.add_argument(
        "--layer3-baseline", type=Path, default=DEFAULT_LAYER3_BASELINE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--small-height", type=int, default=24)
    parser.add_argument("--large-heights", type=int, nargs="+", default=(36, 40))
    parser.add_argument("--max-width", type=int, default=672)
    parser.add_argument("--max-height", type=int, default=192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--skip-full", action="store_true")
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def image_features(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        pixels = np.array(crop_foreground(image_to_rgb(image)), dtype=np.uint8)
    foreground = pixels < 250
    foreground_count = max(1, int(foreground.sum()))
    darkness = (255 - pixels[foreground]) / 255.0
    laplacian = cv2.Laplacian(pixels, cv2.CV_32F)
    gradient_x = cv2.Sobel(pixels, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(pixels, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    stroke = stroke_profile_metrics(pixels)
    return {
        "foreground_height": float(pixels.shape[0]),
        "foreground_width": float(pixels.shape[1]),
        "aspect_ratio": float(pixels.shape[1] / pixels.shape[0]),
        "foreground_density": float(foreground.mean()),
        "soft_pixel_ratio": float(
            ((pixels > 16) & (pixels < 240)).sum() / foreground_count
        ),
        "mean_darkness": float(darkness.mean()),
        "foreground_gray_std": float(pixels[foreground].std() / 255.0),
        "laplacian_variance": float(laplacian[foreground].var() / 10000.0),
        "gradient_mean": float(gradient[foreground].mean() / 255.0),
        "stroke_width_area_perimeter": stroke["stroke_width_area_perimeter"],
        "relative_stroke_width": stroke["relative_stroke_width"],
    }


def base_key(sample: dict[str, Any]) -> str:
    record = sample["record"]
    return str(record.get("derived_from") or record["id"])


def is_holdout(sample: dict[str, Any], seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{base_key(sample)}".encode()).digest()
    return digest[0] % 5 == 0


def diagnostic_result_map(report: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    result = {}
    for record in report["samples"]:
        if record.get("phase") != "diagnostic_scan":
            continue
        height = int(record["requested_foreground_height"])
        result[(record["id"], record["evaluation_group"], height)] = record
    return result


def score_records(records: list[dict[str, Any]]) -> tuple[Any, ...]:
    grouped = grouped_metrics(records)["by_group"]
    supported = [grouped[group] for group in SUPPORTED_GROUPS]
    expansions = sum(group["length_expansion_2x"] for group in supported)
    similarities = [group["mean_character_edit_similarity"] for group in supported]
    return (-expansions, min(similarities), float(np.mean(similarities)))


def choose_result(
    item: dict[str, Any],
    result_map: dict[tuple[str, str, int], dict[str, Any]],
    small_height: int,
    large_height: int,
    feature: str,
    threshold: float,
    large_when_le: bool,
) -> dict[str, Any]:
    use_large = (
        item["features"][feature] <= threshold
        if large_when_le
        else item["features"][feature] > threshold
    )
    height = large_height if use_large else small_height
    key = (item["id"], item["evaluation_group"], height)
    selected = dict(result_map[key])
    selected["selected_profile_height"] = height
    return selected


def thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    return [(left + right) / 2 for left, right in zip(unique, unique[1:])]


def search_rule(
    train: list[dict[str, Any]],
    result_map: dict[tuple[str, str, int], dict[str, Any]],
    small_height: int,
    large_heights: list[int],
) -> dict[str, Any]:
    feature_names = sorted(train[0]["features"])
    best = None
    for large_height in large_heights:
        for feature in feature_names:
            candidates = thresholds([item["features"][feature] for item in train])
            for threshold in candidates:
                for large_when_le in (True, False):
                    chosen = [
                        choose_result(
                            item,
                            result_map,
                            small_height,
                            large_height,
                            feature,
                            threshold,
                            large_when_le,
                        )
                        for item in train
                    ]
                    score = score_records(chosen)
                    candidate = {
                        "small_height": small_height,
                        "large_height": large_height,
                        "feature": feature,
                        "threshold": threshold,
                        "large_when_le": large_when_le,
                        "score": score,
                    }
                    if best is None or score > best["score"]:
                        best = candidate
    if best is None:
        raise RuntimeError("No threshold rule could be evaluated")
    return best


def simulate_rule(
    items: list[dict[str, Any]],
    result_map: dict[tuple[str, str, int], dict[str, Any]],
    rule: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        choose_result(
            item,
            result_map,
            rule["small_height"],
            rule["large_height"],
            rule["feature"],
            rule["threshold"],
            rule["large_when_le"],
        )
        for item in items
    ]


def oracle_results(
    items: list[dict[str, Any]],
    result_map: dict[tuple[str, str, int], dict[str, Any]],
    small_height: int,
    large_height: int,
) -> list[dict[str, Any]]:
    selected = []
    for item in items:
        key = (item["id"], item["evaluation_group"])
        small = result_map[key + (small_height,)]
        large = result_map[key + (large_height,)]
        chosen = max(
            (small, large),
            key=lambda record: (
                -int(record["runaway_2x"]),
                record["character_edit_similarity"],
            ),
        )
        selected.append(chosen)
    return selected


def apply_rule(rule: dict[str, Any], features: dict[str, float]) -> int:
    use_large = (
        features[rule["feature"]] <= rule["threshold"]
        if rule["large_when_le"]
        else features[rule["feature"]] > rule["threshold"]
    )
    return rule["large_height"] if use_large else rule["small_height"]


def run_full(
    samples: list[dict[str, Any]],
    rule: dict[str, Any],
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
        record = sample["record"]
        image_path = (sample["manifest_dir"] / record["image"]).resolve()
        features = image_features(image_path)
        selected_height = apply_rule(rule, features)
        result = evaluate_sample(
            sample=sample,
            target_height=selected_height,
            model=model,
            device=device,
            max_width=max_width,
            max_height=max_height,
            temperature=temperature,
            seed=seed,
            baseline=baseline,
            phase="full_adaptive_validation",
        )
        result["profile_feature"] = rule["feature"]
        result["profile_feature_value"] = features[rule["feature"]]
        result["profile_threshold"] = rule["threshold"]
        result["selected_profile_height"] = selected_height
        results.append(result)
        if index == 1 or index % 50 == 0 or index == len(samples):
            print(
                f"[adaptive {index}/{len(samples)}] {result['id']} "
                f"h={selected_height} char={result['character_edit_similarity']:.3f} "
                f"shape={result['canvas_height']}x{result['canvas_width']}"
            )
    return results


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "phase",
        "evaluation_group",
        "id",
        "layer",
        "variant",
        "derived_from",
        "source_filename",
        "profile_feature",
        "profile_feature_value",
        "profile_threshold",
        "selected_profile_height",
        "actual_foreground_height",
        "actual_foreground_width",
        "canvas_height",
        "canvas_width",
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
    large_heights = sorted(set(args.large_heights))
    if args.small_height < 1 or any(height <= args.small_height for height in large_heights):
        raise ValueError("Large heights must be greater than the small height")
    paths = [
        args.layer1_manifest.expanduser().resolve(),
        args.layer2_manifest.expanduser().resolve(),
        args.layer3_manifest.expanduser().resolve(),
        args.height_report.expanduser().resolve(),
        args.layer1_baseline.expanduser().resolve(),
        args.layer3_baseline.expanduser().resolve(),
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    _, full_samples = build_samples(
        paths[0], paths[1], paths[2], paths[3]
    )
    height_report = json.loads(paths[3].read_text(encoding="utf-8"))
    result_map = diagnostic_result_map(height_report)
    required_heights = {args.small_height, *large_heights}
    diagnostic_samples = [
        sample
        for sample in full_samples
        if all(
            (
                sample["record"]["id"],
                sample["evaluation_group"],
                height,
            )
            in result_map
            for height in required_heights
        )
    ]
    if not diagnostic_samples:
        raise RuntimeError(
            "The height report has no diagnostic samples for all requested heights"
        )
    diagnostic_items = []
    for sample in diagnostic_samples:
        record = sample["record"]
        image_path = (sample["manifest_dir"] / record["image"]).resolve()
        diagnostic_items.append(
            {
                "id": record["id"],
                "evaluation_group": sample["evaluation_group"],
                "base_key": base_key(sample),
                "features": image_features(image_path),
                "holdout": is_holdout(sample, args.seed),
            }
        )
    train = [item for item in diagnostic_items if not item["holdout"]]
    holdout = [item for item in diagnostic_items if item["holdout"]]
    print(
        f"Diagnostic items: {len(diagnostic_items)}; "
        f"train: {len(train)}; holdout: {len(holdout)}"
    )

    rule = search_rule(
        train=train,
        result_map=result_map,
        small_height=args.small_height,
        large_heights=large_heights,
    )
    train_results = simulate_rule(train, result_map, rule)
    holdout_results = simulate_rule(holdout, result_map, rule)
    diagnostic_results = simulate_rule(diagnostic_items, result_map, rule)
    oracle = oracle_results(
        diagnostic_items,
        result_map,
        args.small_height,
        rule["large_height"],
    )
    print(
        f"Selected rule: use h={rule['large_height']} when "
        f"{rule['feature']} "
        f"{'<=' if rule['large_when_le'] else '>'} {rule['threshold']:.6f}; "
        f"otherwise h={rule['small_height']}"
    )

    full_results = []
    if not args.skip_full:
        device = torch.device(args.device)
        print(f"Loading pix2tex on {device}...")
        model = LatexOCR()
        configure_device(model, device)
        model.args.max_seq_len = args.max_tokens
        baseline = {
            **load_report_by_id(paths[4]),
            **load_report_by_id(paths[5]),
        }
        full_results = run_full(
            samples=full_samples,
            rule=rule,
            model=model,
            device=device,
            max_width=args.max_width,
            max_height=args.max_height,
            temperature=args.temperature,
            seed=args.seed,
            baseline=baseline,
        )

    report = {
        "report_schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pix2tex": package_version("pix2tex"),
            "torch": torch.__version__,
            "device": args.device,
            "seed": args.seed,
        },
        "inputs": {
            "height_report": str(paths[3]),
            "diagnostic_records": len(diagnostic_items),
            "train_records": len(train),
            "holdout_records": len(holdout),
            "full_records": 0 if args.skip_full else len(full_samples),
        },
        "feature_definitions": {
            "foreground_height": "height of the exact normalized foreground bbox",
            "foreground_width": "width of the exact normalized foreground bbox",
            "aspect_ratio": "foreground width divided by height",
            "foreground_density": "pixels below 250 divided by bbox area",
            "soft_pixel_ratio": "midtone pixels 17-239 divided by foreground pixels",
            "mean_darkness": "mean normalized darkness of foreground pixels",
            "foreground_gray_std": "foreground grayscale standard deviation",
            "laplacian_variance": "normalized Laplacian variance on foreground",
            "gradient_mean": "normalized Sobel gradient magnitude on foreground",
            "stroke_width_area_perimeter": (
                "portable Otsu foreground width estimated as 2 * area / grid perimeter"
            ),
            "relative_stroke_width": (
                "portable grid-perimeter stroke width divided by foreground height"
            ),
        },
        "selected_rule": {
            **{key: value for key, value in rule.items() if key != "score"},
            "description": (
                f"Use target height {rule['large_height']} when "
                f"{rule['feature']} "
                f"{'<=' if rule['large_when_le'] else '>'} "
                f"{rule['threshold']:.6f}; otherwise use {rule['small_height']}."
            ),
        },
        "simulated_diagnostic_metrics": {
            "train": grouped_metrics(train_results),
            "holdout": grouped_metrics(holdout_results),
            "all": grouped_metrics(diagnostic_results),
            "oracle_upper_bound": grouped_metrics(oracle),
        },
        "full_validation_metrics": (
            None if args.skip_full else grouped_metrics(full_results)
        ),
        "observed_shapes_hw": [] if args.skip_full else shape_counts(full_results),
        "full_samples": full_results,
        "notes": [
            "The rule uses image features only; dataset/layer identity is not an input.",
            "Layer 1 and its Layer 2 variants share the same train/holdout split key.",
            "String similarity is diagnostic, not mathematical-semantic accuracy.",
        ],
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = args.csv.expanduser().resolve()
    write_csv(csv_path, full_results)

    print("\n=== stroke profile comparison complete ===")
    for name, records in (
        ("train", train_results),
        ("holdout", holdout_results),
        ("diagnostic", diagnostic_results),
        ("oracle", oracle),
    ):
        metrics = grouped_metrics(records)["by_group"]
        chars = [metrics[group]["mean_character_edit_similarity"] for group in SUPPORTED_GROUPS]
        expansions = sum(metrics[group]["length_expansion_2x"] for group in SUPPORTED_GROUPS)
        print(
            f"{name}: worst-char {min(chars):.3f}, "
            f"macro-char {float(np.mean(chars)):.3f}, expansions {expansions}"
        )
    if full_results:
        for group, metrics in grouped_metrics(full_results)["by_group"].items():
            print(
                f"full {group}: char {metrics['mean_character_edit_similarity']:.3f}, "
                f"length>=2x {metrics['length_expansion_2x']}/{metrics['records']}"
            )
    print(f"JSON: {output_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
