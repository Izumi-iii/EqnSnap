#!/usr/bin/env python3
"""Evaluate official pix2tex against an EqnSnap manifest with verified labels."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")

import numpy as np
import pix2tex
import torch
from PIL import Image
from pix2tex import cli as pix2tex_cli
from pix2tex.cli import LatexOCR
from pix2tex.utils import post_process


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    SCRIPT_DIR
    / "datasets"
    / "evaluation"
    / "layer3-real-screenshots"
    / "manifest.jsonl"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "test-output" / "layer3-pix2tex-evaluation.json"
DEFAULT_CSV = SCRIPT_DIR / "test-output" / "layer3-pix2tex-samples.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pix2tex on verified EqnSnap evaluation samples."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "id" not in record or "image" not in record:
                raise RuntimeError(f"Invalid manifest record at line {line_number}")
            records.append(record)
    return records


def configure_device(model: LatexOCR, device: torch.device) -> None:
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    model.args.device = str(device)
    model.model.to(device).eval()
    if model.image_resizer is not None:
        model.image_resizer.to(device).eval()


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def sample_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def normalize_latex(value: str) -> str:
    normalized = post_process(value.strip())
    return re.sub(r"\s+", " ", normalized).strip()


def levenshtein(left: list[Any] | str, right: list[Any] | str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def edit_similarity(distance: int, left_length: int, right_length: int) -> float:
    denominator = max(left_length, right_length)
    return 1.0 if denominator == 0 else 1.0 - distance / denominator


def image_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def expected_label(record: dict[str, Any]) -> tuple[str | None, str | None]:
    if record.get("label_status") == "verified" and record.get("expected_latex"):
        return str(record["expected_latex"]), "human_verified"
    if record.get("normalized_latex"):
        return str(record["normalized_latex"]), "dataset_normalized_latex"
    if record.get("latex"):
        return str(record["latex"]), "dataset_latex"
    return None, None


def evaluate_sample(
    record: dict[str, Any],
    manifest_dir: Path,
    model: LatexOCR,
    device: torch.device,
    temperature: float,
    global_seed: int,
) -> dict[str, Any]:
    expected, label_source = expected_label(record)
    if expected is None:
        raise RuntimeError(f"Sample has no expected LaTeX: {record['id']}")
    image_path = (manifest_dir / record["image"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if record.get("sha256") and sha256(image_path) != record["sha256"]:
        raise RuntimeError(f"Image hash differs from manifest: {image_path}")

    deterministic_seed = sample_seed(global_seed, record["id"])
    torch.manual_seed(deterministic_seed)
    model.args.temperature = temperature
    with Image.open(image_path) as image:
        rgb_image = image_to_rgb(image)
        started = time.perf_counter()
        prediction = model(rgb_image)
        synchronize(device)
        inference_seconds = time.perf_counter() - started

    normalized_expected = normalize_latex(expected)
    normalized_prediction = normalize_latex(prediction)
    character_distance = levenshtein(normalized_prediction, normalized_expected)
    expected_tokens = model.tokenizer.encode(expected)
    predicted_tokens = model.tokenizer.encode(prediction)
    token_distance = levenshtein(predicted_tokens, expected_tokens)
    return {
        "id": record["id"],
        "source_filename": record.get("source_filename"),
        "image": str(image_path),
        "in_v0_1_scope": record.get("in_v0_1_scope"),
        "notes": record.get("notes", ""),
        "layer": record.get("layer"),
        "variant": record.get("transform", {}).get("variant"),
        "label_source": label_source,
        "seed": deterministic_seed,
        "expected_latex": expected,
        "predicted_latex": prediction,
        "normalized_expected": normalized_expected,
        "normalized_prediction": normalized_prediction,
        "raw_exact_match": prediction.strip() == expected.strip(),
        "normalized_exact_match": normalized_prediction == normalized_expected,
        "character_edit_distance": character_distance,
        "character_edit_similarity": edit_similarity(
            character_distance,
            len(normalized_prediction),
            len(normalized_expected),
        ),
        "token_edit_distance": token_distance,
        "token_edit_similarity": edit_similarity(
            token_distance, len(predicted_tokens), len(expected_tokens)
        ),
        "expected_token_count": len(expected_tokens),
        "predicted_token_count": len(predicted_tokens),
        "inference_seconds": inference_seconds,
        "requires_semantic_review": normalized_prediction != normalized_expected,
        "error": None,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("error") is None]
    failed = len(records) - len(successful)
    if not successful:
        return {"records": len(records), "successful": 0, "failed": failed}

    latencies = np.array(
        [record["inference_seconds"] for record in successful], dtype=np.float64
    )
    return {
        "records": len(records),
        "successful": len(successful),
        "failed": failed,
        "raw_exact_matches": sum(record["raw_exact_match"] for record in successful),
        "raw_exact_match_rate": float(
            np.mean([record["raw_exact_match"] for record in successful])
        ),
        "normalized_exact_matches": sum(
            record["normalized_exact_match"] for record in successful
        ),
        "normalized_exact_match_rate": float(
            np.mean([record["normalized_exact_match"] for record in successful])
        ),
        "mean_character_edit_similarity": float(
            np.mean([record["character_edit_similarity"] for record in successful])
        ),
        "mean_token_edit_similarity": float(
            np.mean([record["token_edit_similarity"] for record in successful])
        ),
        "semantic_review_required": sum(
            record["requires_semantic_review"] for record in successful
        ),
        "latency_seconds": {
            "mean": float(latencies.mean()),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(latencies.max()),
        },
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "id",
        "source_filename",
        "layer",
        "variant",
        "label_source",
        "in_v0_1_scope",
        "raw_exact_match",
        "normalized_exact_match",
        "character_edit_similarity",
        "token_edit_similarity",
        "inference_seconds",
        "expected_latex",
        "predicted_latex",
        "notes",
        "error",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_partial_report(
    path: Path,
    manifest_path: Path,
    completed: list[dict[str, Any]],
    total: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "completed": len(completed),
                "total": total,
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
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    all_records = load_manifest(manifest_path)
    eligible = [record for record in all_records if expected_label(record)[0] is not None]
    skipped = len(all_records) - len(eligible)
    if args.limit is not None:
        eligible = eligible[: args.limit]
    if not eligible:
        raise RuntimeError("Manifest contains no usable labels")

    device = torch.device(args.device)
    pix2tex_cli.clipboard.copy = lambda _: None
    print(f"Loading pix2tex on {device}...")
    load_started = time.perf_counter()
    model = LatexOCR()
    configure_device(model, device)
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_started

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.json")
    results = []
    for index, record in enumerate(eligible, start=1):
        try:
            result = evaluate_sample(
                record=record,
                manifest_dir=manifest_path.parent,
                model=model,
                device=device,
                temperature=args.temperature,
                global_seed=args.seed,
            )
            status = "normalized-match" if result["normalized_exact_match"] else "review"
            if index == 1 or index % args.checkpoint_every == 0 or index == len(eligible):
                print(
                    f"[{index}/{len(eligible)}] {record['id']} "
                    f"{result['inference_seconds']:.3f}s {status}"
                )
        except Exception as error:
            expected, label_source = expected_label(record)
            result = {
                "id": record["id"],
                "source_filename": record.get("source_filename"),
                "layer": record.get("layer"),
                "variant": record.get("transform", {}).get("variant"),
                "label_source": label_source,
                "in_v0_1_scope": record.get("in_v0_1_scope"),
                "expected_latex": expected,
                "predicted_latex": None,
                "notes": record.get("notes", ""),
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"[{index}/{len(eligible)}] {record['id']} failed: {error}")
        results.append(result)
        if index % args.checkpoint_every == 0:
            write_partial_report(partial_path, manifest_path, results, len(eligible))

    in_scope = [record for record in results if record.get("in_v0_1_scope") is True]
    out_of_scope = [
        record for record in results if record.get("in_v0_1_scope") is False
    ]
    variants = sorted(
        {record["variant"] for record in results if record.get("variant") is not None}
    )
    report = {
        "report_schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pix2tex": version("pix2tex"),
            "torch": torch.__version__,
            "device": str(device),
            "temperature": args.temperature,
            "global_seed": args.seed,
            "model_load_seconds": model_load_seconds,
        },
        "manifest": {
            "path": str(manifest_path),
            "records": len(all_records),
            "evaluated": len(eligible),
            "skipped_without_labels": skipped,
            "limited": args.limit is not None,
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
        "metric_notes": [
            "Raw exact match compares trimmed strings only.",
            "Normalized match applies pix2tex post_process and whitespace collapse.",
            "String mismatch does not prove mathematical or rendered-semantic mismatch.",
            "Samples marked requires_semantic_review need human or renderer-based review.",
        ],
        "samples": results,
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = args.csv.expanduser().resolve()
    write_csv(csv_path, results)
    partial_path.unlink(missing_ok=True)

    metrics = report["metrics"]["v0_1_in_scope"]
    print("\n=== pix2tex evaluation ===")
    print(f"Evaluated: {len(eligible)}, skipped without labels: {skipped}")
    print(f"v0.1 in-scope: {metrics['records']}")
    print(
        f"raw exact: {metrics['raw_exact_matches']}/{metrics['successful']} "
        f"({metrics['raw_exact_match_rate']:.1%})"
    )
    print(
        f"normalized exact: {metrics['normalized_exact_matches']}/"
        f"{metrics['successful']} ({metrics['normalized_exact_match_rate']:.1%})"
    )
    print(
        f"mean character similarity: "
        f"{metrics['mean_character_edit_similarity']:.3f}"
    )
    print(
        f"mean token similarity: {metrics['mean_token_edit_similarity']:.3f}"
    )
    print(
        f"latency P50/P95: {metrics['latency_seconds']['p50']:.3f}s / "
        f"{metrics['latency_seconds']['p95']:.3f}s"
    )
    print(f"JSON: {output_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
