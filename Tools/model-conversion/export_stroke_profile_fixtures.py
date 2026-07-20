#!/usr/bin/env python3
"""Export normalized grayscale fixtures for Python-Swift profile parity tests."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from compare_pix2tex_preprocessing import crop_foreground
from compare_pix2tex_target_height import build_samples
from evaluate_pix2tex import image_to_rgb
from stroke_profile_contract import stroke_profile_metrics


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
EVALUATION_DIR = SCRIPT_DIR / "datasets" / "evaluation"
DEFAULT_LAYER1 = EVALUATION_DIR / "layer1a-simple-single-line" / "manifest.jsonl"
DEFAULT_LAYER2 = EVALUATION_DIR / "layer2-screenshot-styles" / "manifest.jsonl"
DEFAULT_LAYER3 = EVALUATION_DIR / "layer3-real-screenshots" / "manifest.jsonl"
DEFAULT_HEIGHT_REPORT = SCRIPT_DIR / "test-output" / "pix2tex-target-height-comparison.json"
DEFAULT_RULE_REPORT = SCRIPT_DIR / "test-output" / "pix2tex-stroke-profile-comparison.json"
DEFAULT_OUTPUT = REPO_ROOT / "EqnSnapTests" / "Fixtures" / "StrokeProfiles"

FIXTURE_IDS = (
    "real-0024",
    "im2latex-test-04022",
    "im2latex-test-03404--degraded",
    "im2latex-test-06291",
    "im2latex-test-04673--degraded",
    "real-0006",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer1-manifest", type=Path, default=DEFAULT_LAYER1)
    parser.add_argument("--layer2-manifest", type=Path, default=DEFAULT_LAYER2)
    parser.add_argument("--layer3-manifest", type=Path, default=DEFAULT_LAYER3)
    parser.add_argument("--height-report", type=Path, default=DEFAULT_HEIGHT_REPORT)
    parser.add_argument("--rule-report", type=Path, default=DEFAULT_RULE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [
        args.layer1_manifest.expanduser().resolve(),
        args.layer2_manifest.expanduser().resolve(),
        args.layer3_manifest.expanduser().resolve(),
        args.height_report.expanduser().resolve(),
        args.rule_report.expanduser().resolve(),
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    _, samples = build_samples(paths[0], paths[1], paths[2], paths[3])
    samples_by_id = {sample["record"]["id"]: sample for sample in samples}
    rule = json.loads(paths[4].read_text(encoding="utf-8"))["selected_rule"]
    if rule["feature"] != "relative_stroke_width":
        raise RuntimeError(
            "Swift parity fixtures require a relative_stroke_width rule"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for sample_id in FIXTURE_IDS:
        sample = samples_by_id.get(sample_id)
        if sample is None:
            raise RuntimeError(f"Fixture sample not found: {sample_id}")
        record = sample["record"]
        image_path = (sample["manifest_dir"] / record["image"]).resolve()
        with Image.open(image_path) as image:
            pixels = np.array(
                crop_foreground(image_to_rgb(image)), dtype=np.uint8
            )
        metrics = stroke_profile_metrics(pixels)
        use_large = (
            metrics[rule["feature"]] <= rule["threshold"]
            if rule["large_when_le"]
            else metrics[rule["feature"]] > rule["threshold"]
        )
        selected_height = (
            rule["large_height"] if use_large else rule["small_height"]
        )
        image_name = f"{sample_id}.pgm"
        header = f"P5\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode("ascii")
        (output_dir / image_name).write_bytes(header + pixels.tobytes())
        cases.append(
            {
                "id": sample_id,
                "evaluationGroup": sample["evaluation_group"],
                "image": image_name,
                "width": int(pixels.shape[1]),
                "height": int(pixels.shape[0]),
                "otsuThreshold": metrics["otsu_threshold"],
                "foregroundArea": metrics["foreground_area"],
                "foregroundPerimeter": metrics["foreground_perimeter"],
                "estimatedStrokeWidth": metrics["stroke_width_area_perimeter"],
                "relativeStrokeWidth": metrics["relative_stroke_width"],
                "selectedTargetHeight": selected_height,
            }
        )

    fixture = {
        "schemaVersion": 1,
        "contract": {
            "otsuTieBreak": "first threshold with strictly greater score",
            "foregroundRule": "pixel <= otsuThreshold",
            "perimeterRule": "four-neighbor exposed grid edges",
            "strokeWidthRule": "2 * foregroundArea / foregroundPerimeter",
        },
        "policy": {
            "relativeStrokeWidthThreshold": rule["threshold"],
            "smallTargetHeight": rule["small_height"],
            "largeTargetHeight": rule["large_height"],
            "largeWhenLessThanOrEqual": rule["large_when_le"],
        },
        "cases": cases,
    }
    fixture_path = output_dir / "fixtures.json"
    fixture_path.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(cases)} fixtures to {fixture_path}")


if __name__ == "__main__":
    main()
