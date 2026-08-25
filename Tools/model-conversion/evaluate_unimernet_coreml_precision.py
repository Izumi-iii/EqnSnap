#!/usr/bin/env python3
"""Evaluate a converted UniMERNet Core ML pipeline against PyTorch reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings(
    "ignore",
    message="The image_processor_class argument is deprecated.*",
    category=FutureWarning,
)

import coremltools as ct
import numpy as np
import torch
from ftfy import fix_text
from omegaconf import OmegaConf
from PIL import Image
from unimernet.models.unimernet.encoder_decoder import DonutTokenizer
from unimernet.processors import load_processor

from convert_unimernet_decoder_prefix import MAX_TOKEN_LENGTH, pad_tokens
from convert_unimernet_decoder_cached_step import (
    ATTENTION_HEADS,
    DECODER_LAYERS,
    KEY_DIMENSION,
    VALUE_DIMENSION,
    coreml_step,
)
from convert_unimernet_encoder import DEFAULT_ARTIFACTS
from evaluate_unimernet import DEFAULT_MODEL_DIR, load_manifest, sha256


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BALANCED_REPORT = SCRIPT_DIR / "test-output" / "unimernet-balanced-smoke-15.json"
DEFAULT_REAL_REPORT = SCRIPT_DIR / "test-output" / "layer3-unimernet-evaluation.json"
REAL_SAMPLE_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument(
        "--decoder-strategy",
        choices=("prefix", "cached"),
        default="prefix",
    )
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument(
        "--reference-report",
        type=Path,
        action="append",
        default=None,
        help="Evaluate every successful sample in each supplied PyTorch report.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def package_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative_path = file.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def quantile_samples(samples: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(samples, key=lambda sample: (sample["generated_token_count"], sample["id"]))
    if count >= len(ordered):
        return ordered
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def selected_references(
    report_paths: list[Path] | None,
) -> list[tuple[Path, dict[str, Any]]]:
    selections = []
    configured_reports = (
        [(path.expanduser().resolve(), lambda samples: samples) for path in report_paths]
        if report_paths is not None
        else [
            (DEFAULT_BALANCED_REPORT, lambda samples: samples),
            (
                DEFAULT_REAL_REPORT,
                lambda samples: quantile_samples(samples, REAL_SAMPLE_COUNT),
            ),
        ]
    )
    for report_path, selector in configured_reports:
        if not report_path.is_file():
            raise FileNotFoundError(f"Reference report not found: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        samples = [sample for sample in report["samples"] if sample.get("error") is None]
        for sample in selector(samples):
            selections.append((Path(report["manifest"]["path"]), sample))
    return selections


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def generate_prefix(
    decoder: ct.models.MLModel,
    output_name: str,
    encoder_context: np.ndarray,
    bos_token_id: int,
    pad_token_id: int,
    eos_token_id: int,
) -> tuple[list[int], list[float], bool]:
    tokens = [bos_token_id]
    durations = []
    stopped_on_eos = False
    while len(tokens) <= MAX_TOKEN_LENGTH:
        input_ids, token_mask = pad_tokens(tokens, pad_token_id)
        started = time.perf_counter()
        prediction = decoder.predict(
            {
                "input_ids": input_ids,
                "token_mask": token_mask,
                "encoder_context": encoder_context,
            }
        )
        durations.append(time.perf_counter() - started)
        next_token = int(np.asarray(prediction[output_name])[0, -1].argmax())
        tokens.append(next_token)
        if next_token == eos_token_id:
            stopped_on_eos = True
            break
    return tokens[1:], durations, stopped_on_eos


def generate_cached(
    decoder: ct.models.MLModel,
    encoder_context: np.ndarray,
    bos_token_id: int,
    eos_token_id: int,
) -> tuple[list[int], list[float], bool]:
    input_cache_names = []
    output_cache_names = []
    cache = []
    for layer in range(DECODER_LAYERS):
        input_cache_names.extend((f"self_key_{layer}", f"self_value_{layer}"))
        output_cache_names.extend(
            (f"next_self_key_{layer}", f"next_self_value_{layer}")
        )
        cache.extend(
            (
                np.zeros(
                    (1, ATTENTION_HEADS, 1, KEY_DIMENSION), dtype=np.float32
                ),
                np.zeros(
                    (1, ATTENTION_HEADS, 1, VALUE_DIMENSION), dtype=np.float32
                ),
            )
        )
    current_token = bos_token_id
    generated = []
    durations = []
    stopped_on_eos = False
    for _ in range(MAX_TOKEN_LENGTH):
        started = time.perf_counter()
        logits, cache = coreml_step(
            decoder,
            input_cache_names,
            output_cache_names,
            current_token,
            encoder_context,
            cache,
        )
        durations.append(time.perf_counter() - started)
        current_token = int(logits[0, -1].argmax())
        generated.append(current_token)
        if current_token == eos_token_id:
            stopped_on_eos = True
            break
    return generated, durations, stopped_on_eos


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    precision_label = args.precision.upper()
    encoder_path = artifacts_dir / f"UniMERNetTinyEncoder-{precision_label}.mlpackage"
    decoder_filename = (
        f"UniMERNetTinyDecoder-Prefix512-{precision_label}.mlpackage"
        if args.decoder_strategy == "prefix"
        else f"UniMERNetTinyDecoder-CachedStep-SelfKV-{precision_label}.mlpackage"
    )
    decoder_path = artifacts_dir / decoder_filename
    for path in (encoder_path, decoder_path):
        if not path.is_dir():
            raise FileNotFoundError(f"Core ML package not found: {path}")

    checkpoint_path = model_dir / "unimernet_tiny.pth"
    references = selected_references(args.reference_report)
    if args.sample_id is not None:
        requested_ids = set(args.sample_id)
        references = [item for item in references if item[1]["id"] in requested_ids]
        found_ids = {item[1]["id"] for item in references}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise RuntimeError(f"Sample IDs not found in parity selection: {missing_ids}")
    manifests: dict[Path, dict[str, dict[str, Any]]] = {}
    for manifest_path, _ in references:
        resolved = manifest_path.expanduser().resolve()
        if resolved not in manifests:
            manifests[resolved] = {
                record["id"]: record for record in load_manifest(resolved)
            }

    print(f"Loading {precision_label} Core ML models...")
    encoder = ct.models.MLModel(str(encoder_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    decoder = ct.models.MLModel(str(decoder_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    encoder_output_name = encoder.get_spec().description.output[0].name
    decoder_output_name = decoder.get_spec().description.output[0].name
    tokenizer_wrapper = DonutTokenizer(str(model_dir))
    tokenizer = tokenizer_wrapper.tokenizer
    processor = load_processor(
        "formula_image_eval", OmegaConf.create({"image_size": [192, 672]})
    )

    results = []
    for index, (manifest_path, reference) in enumerate(references, start=1):
        if reference["generated_token_count"] > MAX_TOKEN_LENGTH:
            results.append(
                {
                    "id": reference["id"],
                    "source": reference.get("complex_formula_category"),
                    "reference_generated_token_count": reference[
                        "generated_token_count"
                    ],
                    "status": "unsupported_token_length",
                    "maximum_supported_tokens": MAX_TOKEN_LENGTH,
                }
            )
            print(
                f"[{index}/{len(references)}] {reference['id']} SKIP "
                f"tokens={reference['generated_token_count']}>{MAX_TOKEN_LENGTH}",
                flush=True,
            )
            continue
        resolved_manifest = manifest_path.expanduser().resolve()
        record = manifests[resolved_manifest][reference["id"]]
        image_path = (resolved_manifest.parent / record["image"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if record.get("sha256") and sha256(image_path) != record["sha256"]:
            raise RuntimeError(f"Image hash differs from manifest: {image_path}")

        with Image.open(image_path) as image:
            grayscale = processor(image.convert("RGB")).unsqueeze(0)
        pixel_values = grayscale.repeat(1, 3, 1, 1).numpy().astype(np.float32)

        encoder_started = time.perf_counter()
        encoder_prediction = encoder.predict({"pixel_values": pixel_values})
        encoder_seconds = time.perf_counter() - encoder_started
        encoder_context = np.asarray(
            encoder_prediction[encoder_output_name], dtype=np.float32
        )

        if args.decoder_strategy == "prefix":
            generated_ids, decoder_durations, stopped_on_eos = generate_prefix(
                decoder,
                decoder_output_name,
                encoder_context,
                int(tokenizer.bos_token_id),
                int(tokenizer.pad_token_id),
                int(tokenizer.eos_token_id),
            )
        else:
            generated_ids, decoder_durations, stopped_on_eos = generate_cached(
                decoder,
                encoder_context,
                int(tokenizer.bos_token_id),
                int(tokenizer.eos_token_id),
            )
        predicted_latex = fix_text(
            tokenizer.decode(generated_ids, skip_special_tokens=True)
        )
        reference_latex = str(reference["predicted_latex"])
        exact_match = predicted_latex == reference_latex
        normalized_match = " ".join(predicted_latex.split()) == " ".join(
            reference_latex.split()
        )
        result = {
            "id": reference["id"],
            "status": "evaluated",
            "source": (
                "real_screenshot"
                if "layer3-real-screenshots" in str(resolved_manifest)
                else reference.get("complex_formula_category")
            ),
            "image": str(image_path),
            "reference_generated_token_count": reference["generated_token_count"],
            "coreml_generated_token_count": len(generated_ids),
            "stopped_on_eos": stopped_on_eos,
            "reached_token_capacity": not stopped_on_eos,
            "exact_latex_match": exact_match,
            "normalized_latex_match": normalized_match,
            "reference_latex": reference_latex,
            "coreml_latex": predicted_latex,
            "encoder_seconds": encoder_seconds,
            "decoder_seconds": sum(decoder_durations),
            "decoder_step_p50_seconds": percentile(decoder_durations, 50),
            "decoder_step_p95_seconds": percentile(decoder_durations, 95),
        }
        results.append(result)
        status = "match" if exact_match else "DIFF"
        print(
            f"[{index}/{len(references)}] {reference['id']} {status} "
            f"tokens={len(generated_ids)}/{reference['generated_token_count']} "
            f"decoder={result['decoder_seconds']:.2f}s",
            flush=True,
        )

    evaluated_results = [record for record in results if record["status"] == "evaluated"]
    unsupported_results = [
        record for record in results if record["status"] == "unsupported_token_length"
    ]
    encoder_times = [record["encoder_seconds"] for record in evaluated_results]
    decoder_times = [record["decoder_seconds"] for record in evaluated_results]
    report = {
        "report_schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "coremltools": ct.__version__,
            "compute_units": "cpuOnly",
            "precision": args.precision,
            "decoder_strategy": args.decoder_strategy,
            "maximum_token_prefix": MAX_TOKEN_LENGTH,
        },
        "model": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "encoder": str(encoder_path),
            "encoder_sha256": package_sha256(encoder_path),
            "decoder": str(decoder_path),
            "decoder_sha256": package_sha256(decoder_path),
        },
        "selection": {
            "requested_sample_ids": args.sample_id,
            "reference_reports": (
                [str(path.expanduser().resolve()) for path in args.reference_report]
                if args.reference_report is not None
                else None
            ),
            "total": len(references),
            "evaluated": len(evaluated_results),
            "unsupported_token_length": len(unsupported_results),
        },
        "summary": {
            "exact_latex_matches": sum(
                record["exact_latex_match"] for record in evaluated_results
            ),
            "normalized_latex_matches": sum(
                record["normalized_latex_match"] for record in evaluated_results
            ),
            "token_count_matches": sum(
                record["reference_generated_token_count"]
                == record["coreml_generated_token_count"]
                for record in evaluated_results
            ),
            "stopped_on_eos": sum(
                record["stopped_on_eos"] for record in evaluated_results
            ),
            "reached_token_capacity": sum(
                record["reached_token_capacity"] for record in evaluated_results
            ),
            "encoder_latency_seconds": {
                "mean": statistics.mean(encoder_times),
                "p50": percentile(encoder_times, 50),
                "p95": percentile(encoder_times, 95),
            },
            "decoder_latency_seconds": {
                "mean": statistics.mean(decoder_times),
                "p50": percentile(decoder_times, 50),
                "p95": percentile(decoder_times, 95),
            },
        },
        "samples": results,
        "notes": [
            "Reference strings come from prior PyTorch reports using the same checkpoint.",
            "Exact decoded-string plus generated-token-count parity is checked; prior reports do not contain raw token IDs.",
            "CPU Only timings are local feasibility measurements, not release benchmarks.",
            "References longer than the 512-token Core ML contract are reported as unsupported and not inferred.",
        ],
    }
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (
            SCRIPT_DIR
            / "test-output"
            / (
                f"unimernet-coreml-{args.precision}-{args.decoder_strategy}"
                f"-parity-{len(references)}.json"
            )
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    summary = report["summary"]
    print("\n=== UniMERNet Core ML precision parity ===")
    print(
        f"exact={summary['exact_latex_matches']}/{len(evaluated_results)}, "
        f"token-count={summary['token_count_matches']}/{len(evaluated_results)}, "
        f"eos={summary['stopped_on_eos']}/{len(evaluated_results)}, "
        f"unsupported={len(unsupported_results)}"
    )
    print(
        "latency P50: "
        f"encoder={summary['encoder_latency_seconds']['p50']:.3f}s, "
        f"decoder={summary['decoder_latency_seconds']['p50']:.3f}s"
    )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
