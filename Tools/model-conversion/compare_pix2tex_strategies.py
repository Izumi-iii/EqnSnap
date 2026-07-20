#!/usr/bin/env python3
"""Compare pix2tex preprocessing and decoding strategies on short formulas."""

import argparse
import csv
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time
import types
from typing import Any
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pix2tex import cli as pix2tex_cli
from pix2tex.cli import LatexOCR, pad
from pix2tex.models.transformer import top_k

from evaluate_pix2tex import (
    configure_device,
    edit_similarity,
    expected_label,
    image_to_rgb,
    levenshtein,
    load_manifest,
    normalize_latex,
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
DEFAULT_BASELINE = SCRIPT_DIR / "test-output" / "layer1-pix2tex-evaluation.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "test-output" / "pix2tex-strategy-comparison.json"
DEFAULT_CSV = SCRIPT_DIR / "test-output" / "pix2tex-strategy-comparison.csv"


@dataclass(frozen=True)
class Strategy:
    name: str
    resize: bool
    temperature: float
    decoding: str
    max_tokens: int
    repeat_guard: bool = False
    input_scale: float = 1.0


STRATEGIES = (
    Strategy("no_image_resizer", False, 0.2, "top_k_sampling", 512),
    Strategy(
        "fixed_scale_0_75",
        False,
        0.2,
        "top_k_sampling",
        512,
        input_scale=0.75,
    ),
    Strategy("temperature_0_1", True, 0.1, "top_k_sampling", 512),
    Strategy("argmax", True, 1.0, "argmax", 512),
    Strategy("max_128", True, 0.2, "top_k_sampling", 128),
    Strategy("guarded_128", True, 0.2, "top_k_sampling", 128, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare image resizing, sampling temperature, Argmax, output limits, "
            "and repetition guards on short pix2tex formulas."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--controls", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Run the requested strategies on the full manifest.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=tuple(strategy.name for strategy in STRATEGIES),
        help="Run only the named strategies; baseline is always included.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit selected samples for a smoke test.",
    )
    return parser.parse_args()


def repeated_suffix(tokens: list[int]) -> str | None:
    if len(tokens) >= 8 and len(set(tokens[-8:])) == 1:
        return "same_token_x8"
    for width in range(2, min(12, len(tokens) // 4) + 1):
        suffix = tokens[-width:]
        if tokens[-4 * width :] == suffix * 4:
            return f"repeated_{width}_token_block_x4"
    return None


def make_generator(
    strategy: Strategy, generation_state: dict[str, Any]
) -> types.MethodType:
    def generate(
        decoder: Any,
        start_tokens: torch.Tensor,
        seq_len: int = 256,
        eos_token: int | None = None,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        device = start_tokens.device
        was_training = decoder.net.training
        num_dims = len(start_tokens.shape)
        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        _, initial_length = start_tokens.shape
        decoder.net.eval()
        output = start_tokens
        mask = kwargs.pop("mask", None)
        if mask is None:
            mask = torch.full_like(
                output, True, dtype=torch.bool, device=device
            )

        stop_reason = "max_tokens"
        repeat_pattern = None
        with torch.no_grad():
            for _ in range(seq_len):
                model_input = output[:, -decoder.max_seq_len :]
                model_mask = mask[:, -decoder.max_seq_len :]
                logits = decoder.net(
                    model_input, mask=model_mask, **kwargs
                )[:, -1, :]
                if strategy.decoding == "argmax":
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    filtered_logits = top_k(logits, thres=0.9)
                    probabilities = F.softmax(
                        filtered_logits / temperature, dim=-1
                    )
                    next_token = torch.multinomial(probabilities, 1)

                output = torch.cat((output, next_token), dim=-1)
                mask = F.pad(mask, (0, 1), value=True)
                generated = output[0, initial_length:].tolist()

                if eos_token is not None and next_token.eq(eos_token).all():
                    stop_reason = "eos"
                    break
                if strategy.repeat_guard:
                    repeat_pattern = repeated_suffix(generated)
                    if repeat_pattern is not None:
                        stop_reason = "repetition_guard"
                        break

        generated_output = output[:, initial_length:]
        generation_state.clear()
        generation_state.update(
            {
                "stop_reason": stop_reason,
                "repeat_pattern": repeat_pattern,
                "generated_token_count": generated_output.shape[-1],
            }
        )
        decoder.net.train(was_training)
        if num_dims == 1:
            generated_output = generated_output.squeeze(0)
        return generated_output

    return generate


def select_samples(
    manifest: list[dict[str, Any]],
    baseline_report: dict[str, Any],
    controls: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_by_id = {record["id"]: record for record in manifest}
    baseline_by_id = {
        record["id"]: record
        for record in baseline_report["samples"]
        if record["id"] in manifest_by_id and record.get("error") is None
    }
    if len(baseline_by_id) != len(manifest_by_id):
        missing = sorted(set(manifest_by_id) - set(baseline_by_id))
        raise RuntimeError(
            f"Baseline report is missing {len(missing)} subset samples"
        )

    length_expansions = []
    severe_mismatches = []
    control_pool = []
    for sample_id, baseline in baseline_by_id.items():
        expected_count = max(1, baseline["expected_token_count"])
        length_ratio = baseline["predicted_token_count"] / expected_count
        if length_ratio >= 2.5:
            length_expansions.append(sample_id)
        elif baseline["character_edit_similarity"] < 0.2:
            severe_mismatches.append(sample_id)
        elif baseline["character_edit_similarity"] >= 0.8 and length_ratio <= 1.5:
            control_pool.append(sample_id)

    length_expansions.sort()
    severe_mismatches.sort()
    if controls > len(control_pool):
        raise ValueError(
            f"Requested {controls} controls, but only {len(control_pool)} are eligible"
        )
    selected_controls = random.Random(seed).sample(control_pool, controls)
    selected_ids = length_expansions + severe_mismatches + sorted(selected_controls)

    selected = []
    length_expansion_ids = set(length_expansions)
    severe_mismatch_ids = set(severe_mismatches)
    for sample_id in selected_ids:
        record = dict(manifest_by_id[sample_id])
        if sample_id in length_expansion_ids:
            record["diagnostic_group"] = "length_expansion"
        elif sample_id in severe_mismatch_ids:
            record["diagnostic_group"] = "severe_mismatch"
        else:
            record["diagnostic_group"] = "stable_control"
        selected.append(record)
    return selected, baseline_by_id


def score_prediction(
    record: dict[str, Any],
    prediction: str,
    model: LatexOCR,
) -> dict[str, Any]:
    expected, _ = expected_label(record)
    if expected is None:
        raise RuntimeError(f"Sample has no expected LaTeX: {record['id']}")
    normalized_expected = normalize_latex(expected)
    normalized_prediction = normalize_latex(prediction)
    character_distance = levenshtein(normalized_prediction, normalized_expected)
    expected_tokens = model.tokenizer.encode(expected)
    predicted_tokens = model.tokenizer.encode(prediction)
    token_distance = levenshtein(predicted_tokens, expected_tokens)
    expected_count = len(expected_tokens)
    predicted_count = len(predicted_tokens)
    return {
        "expected_latex": expected,
        "predicted_latex": prediction,
        "normalized_expected": normalized_expected,
        "normalized_prediction": normalized_prediction,
        "raw_exact_match": prediction.strip() == expected.strip(),
        "normalized_exact_match": normalized_prediction == normalized_expected,
        "character_edit_similarity": edit_similarity(
            character_distance,
            len(normalized_prediction),
            len(normalized_expected),
        ),
        "token_edit_similarity": edit_similarity(
            token_distance, predicted_count, expected_count
        ),
        "expected_token_count": expected_count,
        "predicted_token_count": predicted_count,
        "prediction_length_ratio": predicted_count / max(1, expected_count),
        "runaway_2x": predicted_count >= 2 * max(1, expected_count),
    }


def baseline_results(
    selected: list[dict[str, Any]], baseline_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    results = []
    for record in selected:
        baseline = baseline_by_id[record["id"]]
        expected_count = max(1, baseline["expected_token_count"])
        result = dict(baseline)
        result.update(
            {
                "config": "baseline",
                "diagnostic_group": record["diagnostic_group"],
                "resize": True,
                "temperature": 0.2,
                "decoding": "top_k_sampling",
                "max_tokens": 512,
                "repeat_guard": False,
                "input_scale": 1.0,
                "stop_reason": "existing_report",
                "repeat_pattern": None,
                "generated_token_count": None,
                "prediction_length_ratio": baseline["predicted_token_count"]
                / expected_count,
                "runaway_2x": baseline["predicted_token_count"]
                >= 2 * expected_count,
                "matches_baseline_prediction": True,
                "character_similarity_delta": 0.0,
            }
        )
        results.append(result)
    return results


def run_strategy(
    strategy: Strategy,
    selected: list[dict[str, Any]],
    baseline_by_id: dict[str, dict[str, Any]],
    manifest_dir: Path,
    model: LatexOCR,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    results = []
    decoder = model.model.decoder
    original_generate = decoder.generate
    original_max_seq_len = model.args.max_seq_len
    generation_state: dict[str, Any] = {}
    decoder.generate = types.MethodType(
        make_generator(strategy, generation_state), decoder
    )
    model.args.max_seq_len = strategy.max_tokens
    model.args.temperature = strategy.temperature
    try:
        for index, record in enumerate(selected, start=1):
            image_path = (manifest_dir / record["image"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            deterministic_seed = sample_seed(seed, record["id"])
            torch.manual_seed(deterministic_seed)
            generation_state.clear()
            with Image.open(image_path) as image:
                rgb_image = image_to_rgb(image)
                if strategy.input_scale != 1.0:
                    cropped_image = pad(rgb_image)
                    scaled_size = tuple(
                        max(1, round(dimension * strategy.input_scale))
                        for dimension in cropped_image.size
                    )
                    rgb_image = cropped_image.resize(
                        scaled_size, Image.Resampling.LANCZOS
                    )
                started = time.perf_counter()
                prediction = model(rgb_image, resize=strategy.resize)
                synchronize(device)
                inference_seconds = time.perf_counter() - started

            score = score_prediction(record, prediction, model)
            baseline = baseline_by_id[record["id"]]
            strategy_fields = asdict(strategy)
            strategy_fields.pop("name")
            result = {
                "id": record["id"],
                "image": str(image_path),
                "diagnostic_group": record["diagnostic_group"],
                "seed": deterministic_seed,
                "config": strategy.name,
                **strategy_fields,
                **generation_state,
                **score,
                "inference_seconds": inference_seconds,
                "matches_baseline_prediction": prediction
                == baseline["predicted_latex"],
                "character_similarity_delta": score[
                    "character_edit_similarity"
                ]
                - baseline["character_edit_similarity"],
                "error": None,
            }
            results.append(result)
            status = generation_state.get("stop_reason", "unknown")
            print(
                f"[{strategy.name} {index}/{len(selected)}] {record['id']} "
                f"{inference_seconds:.3f}s {status} "
                f"char={score['character_edit_similarity']:.3f} "
                f"ratio={score['prediction_length_ratio']:.1f}x"
            )
    finally:
        decoder.generate = original_generate
        model.args.max_seq_len = original_max_seq_len
    return results


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"records": 0}
    latencies = np.array(
        [record["inference_seconds"] for record in records], dtype=np.float64
    )
    stop_reasons: dict[str, int] = {}
    for record in records:
        reason = record.get("stop_reason", "unknown")
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
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
        "runaway_2x": sum(record["runaway_2x"] for record in records),
        "matches_baseline_prediction": sum(
            record["matches_baseline_prediction"] for record in records
        ),
        "mean_character_similarity_delta": float(
            np.mean([record["character_similarity_delta"] for record in records])
        ),
        "improved_character_similarity": sum(
            record["character_similarity_delta"] > 0.05 for record in records
        ),
        "degraded_character_similarity": sum(
            record["character_similarity_delta"] < -0.05 for record in records
        ),
        "stop_reasons": stop_reasons,
        "latency_seconds": {
            "mean": float(latencies.mean()),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(latencies.max()),
        },
    }


def grouped_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    configurations = sorted({record["config"] for record in results})
    return {
        name: {
            "all": aggregate(
                [record for record in results if record["config"] == name]
            ),
            "failure_cases": aggregate(
                [
                    record
                    for record in results
                    if record["config"] == name
                    and record["diagnostic_group"] != "stable_control"
                ]
            ),
            "length_expansion": aggregate(
                [
                    record
                    for record in results
                    if record["config"] == name
                    and record["diagnostic_group"] == "length_expansion"
                ]
            ),
            "severe_mismatch": aggregate(
                [
                    record
                    for record in results
                    if record["config"] == name
                    and record["diagnostic_group"] == "severe_mismatch"
                ]
            ),
            "stable_control": aggregate(
                [
                    record
                    for record in results
                    if record["config"] == name
                    and record["diagnostic_group"] == "stable_control"
                ]
            ),
        }
        for name in configurations
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "config",
        "diagnostic_group",
        "id",
        "resize",
        "temperature",
        "decoding",
        "max_tokens",
        "repeat_guard",
        "input_scale",
        "stop_reason",
        "repeat_pattern",
        "generated_token_count",
        "normalized_exact_match",
        "character_edit_similarity",
        "token_edit_similarity",
        "prediction_length_ratio",
        "runaway_2x",
        "matches_baseline_prediction",
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


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    args = parse_args()
    if args.controls < 1:
        raise ValueError("--controls must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")

    manifest_path = args.manifest.expanduser().resolve()
    baseline_path = args.baseline_report.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline report not found: {baseline_path}")

    manifest = load_manifest(manifest_path)
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    selected, baseline_by_id = select_samples(
        manifest, baseline_report, args.controls, args.seed
    )
    if args.all_samples:
        selected = []
        for record in sorted(manifest, key=lambda item: item["id"]):
            selected_record = dict(record)
            selected_record["diagnostic_group"] = "all_subset"
            selected.append(selected_record)
    if args.limit is not None:
        selected = selected[: args.limit]

    length_expansion_count = sum(
        record["diagnostic_group"] == "length_expansion" for record in selected
    )
    severe_mismatch_count = sum(
        record["diagnostic_group"] == "severe_mismatch" for record in selected
    )
    failure_count = length_expansion_count + severe_mismatch_count
    control_count = sum(
        record["diagnostic_group"] == "stable_control" for record in selected
    )
    if args.all_samples:
        print(f"Selected all {len(selected)} manifest samples")
    else:
        print(
            f"Selected {len(selected)} samples: "
            f"{length_expansion_count} length expansion + "
            f"{severe_mismatch_count} severe mismatch + "
            f"{control_count} stable controls"
        )

    device = torch.device(args.device)
    pix2tex_cli.clipboard.copy = lambda _: None
    print(f"Loading pix2tex on {device}...")
    model = LatexOCR()
    configure_device(model, device)

    results = baseline_results(selected, baseline_by_id)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(".partial.json")

    requested_strategies = tuple(
        strategy
        for strategy in STRATEGIES
        if args.only is None or strategy.name in args.only
    )
    for strategy in requested_strategies:
        print(f"\n=== {strategy.name} ===")
        results.extend(
            run_strategy(
                strategy=strategy,
                selected=selected,
                baseline_by_id=baseline_by_id,
                manifest_dir=manifest_path.parent,
                model=model,
                device=device,
                seed=args.seed,
            )
        )
        partial_path.write_text(
            json.dumps(
                {
                    "completed_configurations": sorted(
                        {record["config"] for record in results}
                    ),
                    "selected_samples": len(selected),
                    "metrics": grouped_metrics(results),
                    "samples": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    report = {
        "report_schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pix2tex": package_version("pix2tex"),
            "torch": torch.__version__,
            "device": str(device),
            "seed": args.seed,
        },
        "selection": {
            "manifest": str(manifest_path),
            "baseline_report": str(baseline_path),
            "samples": len(selected),
            "failure_cases": failure_count,
            "length_expansion": length_expansion_count,
            "severe_mismatch": severe_mismatch_count,
            "stable_controls": control_count,
            "length_expansion_definition": (
                "predicted tokens >= 2.5 * expected tokens"
            ),
            "severe_mismatch_definition": (
                "baseline character similarity < 0.2 and not length expansion"
            ),
            "control_definition": (
                "baseline character similarity >= 0.8 and length ratio <= 1.5"
            ),
            "all_samples": args.all_samples,
        },
        "strategies": [
            {
                "name": "baseline",
                "resize": True,
                "temperature": 0.2,
                "decoding": "top_k_sampling",
                "max_tokens": 512,
                "repeat_guard": False,
                "input_scale": 1.0,
                "source": "existing Layer 1 report",
            },
            *[asdict(strategy) for strategy in requested_strategies],
        ],
        "metrics": grouped_metrics(results),
        "samples": results,
        "notes": [
            "String similarity is diagnostic and is not mathematical-semantic accuracy.",
            "max_128 contains long outputs but does not by itself correct recognition.",
            "guarded_128 stops one token repeated 8 times or a 2-12 token block repeated 4 times.",
        ],
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = args.csv.expanduser().resolve()
    write_csv(csv_path, results)
    partial_path.unlink(missing_ok=True)

    print("\n=== comparison complete ===")
    for name, metrics in report["metrics"].items():
        if args.all_samples:
            all_metrics = metrics["all"]
            print(
                f"{name}: length>=2x "
                f"{all_metrics.get('runaway_2x', 0)}/{all_metrics['records']}, "
                f"char {all_metrics.get('mean_character_edit_similarity', 0):.3f}, "
                f"P95 {all_metrics.get('latency_seconds', {}).get('p95', 0):.3f}s"
            )
            continue
        failure_metrics = metrics["failure_cases"]
        control_metrics = metrics["stable_control"]
        print(
            f"{name}: failure length>=2x "
            f"{failure_metrics.get('runaway_2x', 0)}/{failure_metrics['records']}, "
            f"failure char {failure_metrics.get('mean_character_edit_similarity', 0):.3f}, "
            f"control char {control_metrics.get('mean_character_edit_similarity', 0):.3f}"
        )
    print(f"JSON: {output_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
