#!/usr/bin/env python3
"""Evaluate UniMERNet Tiny against an EqnSnap manifest."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import time
from pathlib import Path
from typing import Any
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")

import numpy as np
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from PIL import Image
from unimernet.models.unimernet.unimernet import UniMERModel
from unimernet.processors import load_processor


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    SCRIPT_DIR
    / "datasets"
    / "evaluation"
    / "layer1b-complex-formulas"
    / "manifest.jsonl"
)
DEFAULT_MODEL_DIR = SCRIPT_DIR / ".cache" / "unimernet_tiny"
DEFAULT_OUTPUT = SCRIPT_DIR / "test-output" / "layer1b-unimernet-evaluation.json"
DEFAULT_CSV = SCRIPT_DIR / "test-output" / "layer1b-unimernet-samples.csv"
MODEL_REPOSITORY = "wanderkid/unimernet_tiny"
MODEL_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "unimernet_tiny.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UniMERNet Tiny on verified EqnSnap samples."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Fail instead of downloading when model files are missing.",
    )
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


def expected_label(record: dict[str, Any]) -> tuple[str | None, str | None]:
    if record.get("label_status") == "verified" and record.get("expected_latex"):
        return str(record["expected_latex"]), "human_verified"
    if record.get("normalized_latex"):
        return str(record["normalized_latex"]), "dataset_normalized_latex"
    if record.get("latex"):
        return str(record["latex"]), "dataset_latex"
    return None, None


def normalize_latex(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).strip()


def levenshtein(left: str, right: str) -> int:
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


def max_resident_megabytes() -> float:
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024 * 1024)
    return value / 1024


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def ensure_model_files(model_dir: Path, offline: bool) -> Path:
    if all((model_dir / name).is_file() for name in MODEL_FILES):
        return model_dir
    if offline:
        missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
        raise FileNotFoundError(f"Missing UniMERNet model files: {missing}")

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_REPOSITORY} to {model_dir}...")
    snapshot_download(
        repo_id=MODEL_REPOSITORY,
        local_dir=model_dir,
        allow_patterns=list(MODEL_FILES),
    )
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Downloaded model is incomplete: {missing}")
    return model_dir


def load_model(
    model_dir: Path, device: torch.device, max_tokens: int
) -> tuple[UniMERModel, Any, dict[str, Any]]:
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")

    model_config = OmegaConf.create(
        {"model_name": str(model_dir), "max_seq_len": max_tokens}
    )
    tokenizer_config = OmegaConf.create({"path": str(model_dir)})
    model = UniMERModel(
        model_name=str(model_dir),
        model_config=model_config,
        tokenizer_name="nougat",
        tokenizer_config=tokenizer_config,
    )
    checkpoint_path = model_dir / "unimernet_tiny.pth"
    load_result = model.load_checkpoint(str(checkpoint_path))
    model.to(device).eval()
    if device.type == "cpu":
        model.float()

    processor_config = OmegaConf.create({"image_size": [192, 672]})
    processor = load_processor("formula_image_eval", processor_config)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint_path),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    return model, processor, metadata


def evaluate_sample(
    record: dict[str, Any],
    manifest_dir: Path,
    model: UniMERModel,
    processor: Any,
    device: torch.device,
    max_tokens: int,
) -> dict[str, Any]:
    expected, label_source = expected_label(record)
    if expected is None:
        raise RuntimeError(f"Sample has no expected LaTeX: {record['id']}")
    image_path = (manifest_dir / record["image"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if record.get("sha256") and sha256(image_path) != record["sha256"]:
        raise RuntimeError(f"Image hash differs from manifest: {image_path}")

    with Image.open(image_path) as image:
        tensor = processor(image.convert("RGB")).unsqueeze(0).to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            {"image": tensor},
            temperature=1.0,
            do_sample=False,
            top_p=1.0,
        )
    synchronize(device)
    inference_seconds = time.perf_counter() - started

    prediction = str(output["pred_str"][0])
    token_ids = output["pred_ids"][0].detach().cpu().tolist()
    normalized_expected = normalize_latex(expected)
    normalized_prediction = normalize_latex(prediction)
    character_distance = levenshtein(normalized_prediction, normalized_expected)
    reached_limit = len(token_ids) >= max_tokens
    return {
        "id": record["id"],
        "source_filename": record.get("source_filename"),
        "image": str(image_path),
        "layer": record.get("layer"),
        "in_v0_1_scope": record.get("in_v0_1_scope"),
        "complex_formula_category": record.get("complex_formula_category"),
        "complex_selection_reasons": record.get("complex_selection_reasons", []),
        "label_source": label_source,
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
        "generated_token_count": len(token_ids),
        "reached_max_tokens": reached_limit,
        "inference_seconds": inference_seconds,
        "requires_semantic_review": normalized_prediction != normalized_expected,
        "notes": record.get("notes", ""),
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
        "mean_generated_token_count": float(
            np.mean([record["generated_token_count"] for record in successful])
        ),
        "reached_max_tokens": sum(
            record["reached_max_tokens"] for record in successful
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
        "complex_formula_category",
        "label_source",
        "raw_exact_match",
        "normalized_exact_match",
        "character_edit_similarity",
        "generated_token_count",
        "reached_max_tokens",
        "inference_seconds",
        "expected_latex",
        "predicted_latex",
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
    if args.max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2")
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

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model_dir = ensure_model_files(args.model_dir.expanduser().resolve(), args.offline)
    print(f"Loading UniMERNet Tiny on {device}...")
    load_started = time.perf_counter()
    model, processor, model_metadata = load_model(
        model_dir=model_dir,
        device=device,
        max_tokens=args.max_tokens,
    )
    synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    load_max_rss = max_resident_megabytes()
    if model_metadata["missing_keys"] or model_metadata["unexpected_keys"]:
        print(
            "Checkpoint key differences: "
            f"missing={len(model_metadata['missing_keys'])}, "
            f"unexpected={len(model_metadata['unexpected_keys'])}"
        )

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
                processor=processor,
                device=device,
                max_tokens=args.max_tokens,
            )
            status = "normalized-match" if result["normalized_exact_match"] else "review"
            print(
                f"[{index}/{len(eligible)}] {record['id']} "
                f"{result['inference_seconds']:.3f}s "
                f"tokens={result['generated_token_count']} {status}"
            )
        except Exception as error:
            expected, label_source = expected_label(record)
            result = {
                "id": record["id"],
                "complex_formula_category": record.get("complex_formula_category"),
                "label_source": label_source,
                "expected_latex": expected,
                "predicted_latex": None,
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"[{index}/{len(eligible)}] {record['id']} failed: {error}")
        results.append(result)
        if index % args.checkpoint_every == 0:
            write_partial_report(partial_path, manifest_path, results, len(eligible))

    categories = sorted(
        {
            record["complex_formula_category"]
            for record in results
            if record.get("complex_formula_category") is not None
        }
    )
    report = {
        "report_schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "unimernet": version("unimernet"),
            "transformers": version("transformers"),
            "torch": torch.__version__,
            "device": str(device),
            "max_tokens": args.max_tokens,
            "global_seed": args.seed,
            "model_load_seconds": model_load_seconds,
            "max_rss_after_load_megabytes": load_max_rss,
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            **model_metadata,
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
            "by_complex_formula_category": {
                category_name: aggregate(
                    [
                        record
                        for record in results
                        if record.get("complex_formula_category") == category_name
                    ]
                )
                for category_name in categories
            },
        },
        "metric_notes": [
            "Raw exact match compares trimmed strings only.",
            "Normalized match collapses whitespace without model-specific rewriting.",
            "String mismatch does not prove mathematical or rendered-semantic mismatch.",
            "Tokenizer distances are intentionally omitted from cross-model comparison.",
            "Samples marked requires_semantic_review need human or renderer review.",
        ],
        "samples": results,
    }
    report["environment"]["max_rss_after_evaluation_megabytes"] = (
        max_resident_megabytes()
    )

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = args.csv.expanduser().resolve()
    write_csv(csv_path, results)
    partial_path.unlink(missing_ok=True)

    metrics = report["metrics"]["all"]
    print("\n=== UniMERNet Tiny evaluation ===")
    print(f"Evaluated: {len(eligible)}, skipped without labels: {skipped}")
    print(
        f"normalized exact: {metrics['normalized_exact_matches']}/"
        f"{metrics['successful']} ({metrics['normalized_exact_match_rate']:.1%})"
    )
    print(
        "mean character similarity: "
        f"{metrics['mean_character_edit_similarity']:.3f}"
    )
    print(
        f"latency P50/P95: {metrics['latency_seconds']['p50']:.3f}s / "
        f"{metrics['latency_seconds']['p95']:.3f}s"
    )
    print(f"reached max tokens: {metrics['reached_max_tokens']}")
    print(f"JSON: {output_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
