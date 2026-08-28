#!/usr/bin/env python3
"""Diagnose UniMERNet failures across PyTorch, Core ML, and Swift preprocessing."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from ftfy import fix_text
from PIL import Image
from unimernet.models.unimernet.encoder_decoder import DonutTokenizer

from convert_unimernet_encoder import DEFAULT_ARTIFACTS
from evaluate_unimernet import (
    DEFAULT_MODEL_DIR,
    ensure_model_files,
    load_model,
    sha256,
    synchronize,
)
from evaluate_unimernet_coreml_precision import (
    generate_cached,
    generate_prefix,
    package_sha256,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT = SCRIPT_DIR / "test-output" / "unimernet-failure-diagnosis.json"
MAXIMUM_COREML_TOKENS = 512
MAXIMUM_REPEATED_PATTERN_LENGTH = 8
MINIMUM_PATTERN_REPETITIONS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--pytorch-max-tokens", type=int, default=1536)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Fail instead of downloading when official model files are missing.",
    )
    parser.add_argument(
        "--skip-swift-preprocessing",
        action="store_true",
        help="Run only the Python-preprocessed PyTorch and Core ML paths.",
    )
    return parser.parse_args()


def repeated_suffix(token_ids: list[int]) -> dict[str, Any] | None:
    for prefix_length in range(1, len(token_ids) + 1):
        prefix = token_ids[:prefix_length]
        maximum_length = min(
            MAXIMUM_REPEATED_PATTERN_LENGTH,
            len(prefix) // MINIMUM_PATTERN_REPETITIONS,
        )
        for pattern_length in range(1, maximum_length + 1):
            repeated_length = pattern_length * MINIMUM_PATTERN_REPETITIONS
            pattern = prefix[-pattern_length:]
            expected = pattern * MINIMUM_PATTERN_REPETITIONS
            if prefix[-repeated_length:] == expected:
                return {
                    "detected_at_token_count": prefix_length,
                    "pattern": pattern,
                    "pattern_length": pattern_length,
                    "minimum_repetitions": MINIMUM_PATTERN_REPETITIONS,
                }
    return None


def first_divergence(left: list[int], right: list[int]) -> dict[str, Any] | None:
    common_length = min(len(left), len(right))
    for index in range(common_length):
        if left[index] != right[index]:
            return {
                "index": index,
                "left_token": left[index],
                "right_token": right[index],
            }
    if len(left) == len(right):
        return None
    return {
        "index": common_length,
        "left_token": left[common_length] if common_length < len(left) else None,
        "right_token": right[common_length] if common_length < len(right) else None,
    }


def tensor_summary(tensor: np.ndarray) -> dict[str, Any]:
    values = np.asarray(tensor, dtype=np.float64)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
    }


def tensor_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_match": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    left64 = np.asarray(left, dtype=np.float64).reshape(-1)
    right64 = np.asarray(right, dtype=np.float64).reshape(-1)
    difference = left64 - right64
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    return {
        "shape_match": True,
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "cosine_similarity": (
            float(np.dot(left64, right64) / denominator) if denominator else None
        ),
    }


def decode_result(
    token_ids: list[int],
    tokenizer: Any,
    eos_token_id: int,
    durations: list[float],
    stopped_on_eos: bool,
    maximum_tokens: int,
    encoder_seconds: float | None = None,
) -> dict[str, Any]:
    repetition = repeated_suffix(token_ids)
    content_ids = token_ids[:-1] if token_ids and token_ids[-1] == eos_token_id else token_ids
    result = {
        "token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "content_token_count": len(content_ids),
        "stopped_on_eos": stopped_on_eos,
        "maximum_token_length": maximum_tokens,
        "reached_token_limit": not stopped_on_eos and len(token_ids) >= maximum_tokens,
        "repetition": repetition,
        "latex": fix_text(tokenizer.decode(token_ids, skip_special_tokens=True)),
        "decoder_seconds": float(sum(durations)),
        "decoder_step_seconds": durations,
    }
    if encoder_seconds is not None:
        result["encoder_seconds"] = encoder_seconds
        result["total_coreml_seconds"] = encoder_seconds + sum(durations)
    return result


def error_result(error: Exception) -> dict[str, Any]:
    return {
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    }


def compile_swift_exporter(destination: Path) -> None:
    command = [
        "xcrun",
        "swiftc",
        str(REPOSITORY_ROOT / "EqnSnap/Recognition/UniMERNetFormulaImagePreprocessor.swift"),
        str(SCRIPT_DIR / "export_unimernet_swift_preprocessing.swift"),
        "-o",
        str(destination),
    ]
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def export_swift_tensor(
    exporter: Path, image_path: Path, directory: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_path = directory / f"{image_path.stem}.raw"
    metadata_path = directory / f"{image_path.stem}.json"
    subprocess.run(
        [str(exporter), str(image_path), str(raw_path), str(metadata_path)],
        check=True,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = tuple(int(value) for value in metadata["tensor_shape"])
    tensor = np.fromfile(raw_path, dtype="<f4")
    expected_count = int(np.prod(shape))
    if tensor.size != expected_count:
        raise RuntimeError(
            f"Swift tensor contains {tensor.size} values; expected {expected_count}"
        )
    return tensor.reshape(shape), metadata


def release_runtime() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def coreml_paths(artifacts_dir: Path) -> dict[str, Path]:
    return {
        "fp32_encoder": artifacts_dir / "UniMERNetTinyEncoder-FP32.mlpackage",
        "fp32_prefix_decoder": artifacts_dir / "UniMERNetTinyDecoder-Prefix512-FP32.mlpackage",
        "fp16_encoder": artifacts_dir / "UniMERNetTinyEncoder-FP16.mlpackage",
        "fp16_prefix_decoder": artifacts_dir / "UniMERNetTinyDecoder-Prefix512-FP16.mlpackage",
        "fp16_cached_decoder": artifacts_dir / "UniMERNetTinyDecoder-CachedStep-SelfKV-FP16.mlpackage",
    }


def package_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def run_coreml_path(
    samples: list[dict[str, Any]],
    encoder_path: Path,
    decoder_path: Path,
    strategy: str,
    tensor_key: str,
    result_key: str,
    tokenizer: Any,
) -> None:
    print(f"Loading {result_key}...", flush=True)
    encoder = ct.models.MLModel(str(encoder_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    decoder = ct.models.MLModel(str(decoder_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    encoder_output_name = encoder.get_spec().description.output[0].name
    decoder_output_name = decoder.get_spec().description.output[0].name
    bos = int(tokenizer.bos_token_id)
    pad = int(tokenizer.pad_token_id)
    eos = int(tokenizer.eos_token_id)

    for index, sample in enumerate(samples, start=1):
        if sample.get(tensor_key) is None:
            continue
        try:
            started = time.perf_counter()
            prediction = encoder.predict({"pixel_values": sample[tensor_key]})
            encoder_seconds = time.perf_counter() - started
            context = np.asarray(prediction[encoder_output_name], dtype=np.float32)
            if strategy == "prefix":
                token_ids, durations, stopped = generate_prefix(
                    decoder, decoder_output_name, context, bos, pad, eos
                )
            else:
                token_ids, durations, stopped = generate_cached(
                    decoder, context, bos, eos
                )
            sample["report"]["paths"][result_key] = decode_result(
                token_ids,
                tokenizer,
                eos,
                durations,
                stopped,
                MAXIMUM_COREML_TOKENS,
                encoder_seconds,
            )
            print(
                f"[{index}/{len(samples)}] {result_key}: "
                f"tokens={len(token_ids)} eos={stopped}",
                flush=True,
            )
        except Exception as error:
            sample["report"]["paths"][result_key] = error_result(error)
            print(f"[{index}/{len(samples)}] {result_key}: ERROR {error}", flush=True)

    del decoder
    del encoder
    release_runtime()


def add_divergences(report: dict[str, Any]) -> None:
    comparisons = [
        ("pytorch_python", "coreml_fp32_prefix_python"),
        ("coreml_fp32_prefix_python", "coreml_fp16_prefix_python"),
        ("coreml_fp16_prefix_python", "coreml_fp16_cached_python"),
        ("coreml_fp16_cached_python", "coreml_fp16_cached_swift"),
    ]
    report["token_divergences"] = {}
    for left_name, right_name in comparisons:
        left = report["paths"].get(left_name, {})
        right = report["paths"].get(right_name, {})
        key = f"{left_name}__vs__{right_name}"
        if "token_ids" not in left or "token_ids" not in right:
            report["token_divergences"][key] = {"unavailable": True}
            continue
        report["token_divergences"][key] = first_divergence(
            left["token_ids"], right["token_ids"]
        )


def main() -> None:
    args = parse_args()
    if args.pytorch_max_tokens < 1:
        raise ValueError("--pytorch-max-tokens must be positive")

    model_dir = ensure_model_files(args.model_dir.expanduser().resolve(), args.offline)
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    paths = coreml_paths(artifacts_dir)
    missing = [str(path) for path in paths.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing Core ML packages: {missing}")

    image_paths = [path.expanduser().resolve() for path in args.images]
    missing_images = [str(path) for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing input images: {missing_images}")

    tokenizer_wrapper = DonutTokenizer(str(model_dir))
    tokenizer = tokenizer_wrapper.tokenizer
    samples: list[dict[str, Any]] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            dimensions = list(image.size)
        samples.append(
            {
                "path": image_path,
                "python_tensor": None,
                "swift_tensor": None,
                "report": {
                    "image": {
                        "path": str(image_path),
                        "sha256": sha256(image_path),
                        "pixel_size": dimensions,
                    },
                    "preprocessing": {},
                    "paths": {},
                },
            }
        )

    with tempfile.TemporaryDirectory(prefix="eqnsnap-unimernet-diagnosis-") as temporary:
        temporary_path = Path(temporary)
        if not args.skip_swift_preprocessing:
            exporter = temporary_path / "export-unimernet-preprocessing"
            print("Compiling Swift preprocessing exporter...", flush=True)
            compile_swift_exporter(exporter)
            for index, sample in enumerate(samples, start=1):
                try:
                    tensor, metadata = export_swift_tensor(
                        exporter, sample["path"], temporary_path
                    )
                    sample["swift_tensor"] = tensor
                    sample["report"]["preprocessing"]["swift"] = {
                        "metadata": metadata,
                        "tensor": tensor_summary(tensor),
                    }
                    print(f"[{index}/{len(samples)}] Swift preprocessing exported", flush=True)
                except Exception as error:
                    sample["report"]["preprocessing"]["swift"] = error_result(error)
                    print(f"[{index}/{len(samples)}] Swift preprocessing: ERROR {error}", flush=True)

        device = torch.device(args.device)
        print(f"Loading official PyTorch model on {device}...", flush=True)
        model, processor, pytorch_metadata = load_model(
            model_dir, device, args.pytorch_max_tokens
        )
        for index, sample in enumerate(samples, start=1):
            try:
                with Image.open(sample["path"]) as image:
                    python_gray = processor(image.convert("RGB")).unsqueeze(0)
                python_tensor = python_gray.repeat(1, 3, 1, 1).numpy().astype(np.float32)
                sample["python_tensor"] = python_tensor
                preprocessing = sample["report"]["preprocessing"]
                preprocessing["python"] = {"tensor": tensor_summary(python_tensor)}
                if sample["swift_tensor"] is not None:
                    preprocessing["python_vs_swift"] = tensor_comparison(
                        python_tensor, sample["swift_tensor"]
                    )

                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(
                        {"image": python_gray.to(device)},
                        temperature=1.0,
                        do_sample=False,
                        top_p=1.0,
                    )
                synchronize(device)
                duration = time.perf_counter() - started
                token_ids = output["pred_ids"][0].detach().cpu().tolist()
                stopped = bool(token_ids and token_ids[-1] == int(tokenizer.eos_token_id))
                result = decode_result(
                    token_ids,
                    tokenizer,
                    int(tokenizer.eos_token_id),
                    [duration],
                    stopped,
                    args.pytorch_max_tokens,
                )
                result["inference_seconds"] = result.pop("decoder_seconds")
                result.pop("decoder_step_seconds")
                result["model_predicted_latex"] = str(output["pred_str"][0])
                sample["report"]["paths"]["pytorch_python"] = result
                print(
                    f"[{index}/{len(samples)}] pytorch_python: "
                    f"tokens={len(token_ids)} eos={stopped}",
                    flush=True,
                )
            except Exception as error:
                sample["report"]["paths"]["pytorch_python"] = error_result(error)
                print(f"[{index}/{len(samples)}] pytorch_python: ERROR {error}", flush=True)

        del processor
        del model
        release_runtime()

        run_coreml_path(
            samples, paths["fp32_encoder"], paths["fp32_prefix_decoder"],
            "prefix", "python_tensor", "coreml_fp32_prefix_python", tokenizer,
        )
        run_coreml_path(
            samples, paths["fp16_encoder"], paths["fp16_prefix_decoder"],
            "prefix", "python_tensor", "coreml_fp16_prefix_python", tokenizer,
        )
        run_coreml_path(
            samples, paths["fp16_encoder"], paths["fp16_cached_decoder"],
            "cached", "python_tensor", "coreml_fp16_cached_python", tokenizer,
        )
        if not args.skip_swift_preprocessing:
            run_coreml_path(
                samples, paths["fp16_encoder"], paths["fp16_cached_decoder"],
                "cached", "swift_tensor", "coreml_fp16_cached_swift", tokenizer,
            )

    for sample in samples:
        add_divergences(sample["report"])

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "coremltools": ct.__version__,
            "device": args.device,
        },
        "configuration": {
            "pytorch_max_tokens": args.pytorch_max_tokens,
            "coreml_max_tokens": MAXIMUM_COREML_TOKENS,
            "maximum_repeated_pattern_length": MAXIMUM_REPEATED_PATTERN_LENGTH,
            "minimum_pattern_repetitions": MINIMUM_PATTERN_REPETITIONS,
            "swift_preprocessing_included": not args.skip_swift_preprocessing,
            "special_tokens": {
                "bos": int(tokenizer.bos_token_id),
                "eos": int(tokenizer.eos_token_id),
                "pad": int(tokenizer.pad_token_id),
            },
        },
        "models": {
            "pytorch": pytorch_metadata,
            "coreml": {
                name: {
                    "path": str(path),
                    "precision": "fp32" if "fp32" in name else "fp16",
                    "component": "encoder" if "encoder" in name else "decoder",
                    "decoder_strategy": (
                        "cached" if "cached" in name
                        else "prefix" if "decoder" in name
                        else None
                    ),
                    "package_bytes": package_bytes(path),
                    "package_sha256": package_sha256(path),
                }
                for name, path in paths.items()
            },
        },
        "samples": [sample["report"] for sample in samples],
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote diagnosis report: {output_path}")


if __name__ == "__main__":
    main()
