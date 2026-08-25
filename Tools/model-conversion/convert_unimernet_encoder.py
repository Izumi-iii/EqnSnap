#!/usr/bin/env python3
"""Convert UniMERNet Tiny's fixed-shape Encoder to Core ML."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings(
    "ignore",
    message="The image_processor_class argument is deprecated.*",
    category=FutureWarning,
)

import coremltools as ct
import numpy as np
import torch
from PIL import Image

from evaluate_unimernet import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL_DIR,
    ensure_model_files,
    load_manifest,
    load_model,
    sha256,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = SCRIPT_DIR / "artifacts"
INPUT_SHAPE = (1, 3, 192, 672)
OUTPUT_SHAPE = (1, 126, 512)


class UniMERNetEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values, return_dict=False)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert UniMERNet Tiny Encoder from [1,3,192,672] to "
            "[1,126,512] as a Core ML ML Program."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Core ML internal compute and weight precision.",
    )
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


def select_image(args: argparse.Namespace) -> tuple[Path, str | None]:
    if args.image is not None:
        image_path = args.image.expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image_path, None

    manifest_path = args.manifest.expanduser().resolve()
    records = load_manifest(manifest_path)
    if not records:
        raise RuntimeError(f"Manifest contains no samples: {manifest_path}")
    record = records[0]
    if args.sample_id is not None:
        record = next(
            (candidate for candidate in records if candidate["id"] == args.sample_id),
            None,
        )
        if record is None:
            raise RuntimeError(f"Sample ID not found: {args.sample_id}")
    image_path = (manifest_path.parent / record["image"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if record.get("sha256") and sha256(image_path) != record["sha256"]:
        raise RuntimeError(f"Image hash differs from manifest: {image_path}")
    return image_path, str(record["id"])


def tensor_summary(array: np.ndarray) -> dict[str, Any]:
    values = array.astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(values).all()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    reference64 = reference.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    difference = candidate64 - reference64
    denominator = np.linalg.norm(reference64.ravel()) * np.linalg.norm(
        candidate64.ravel()
    )
    cosine_similarity = (
        float(np.dot(reference64.ravel(), candidate64.ravel()) / denominator)
        if denominator != 0
        else math.nan
    )
    return {
        "shape_match": True,
        "all_finite": bool(
            np.isfinite(reference64).all() and np.isfinite(candidate64).all()
        ),
        "max_absolute_error": float(np.abs(difference).max()),
        "mean_absolute_error": float(np.abs(difference).mean()),
        "root_mean_square_error": float(np.sqrt(np.mean(difference**2))),
        "cosine_similarity": cosine_similarity,
    }


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative_path = file.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def remove_existing_artifact(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def coreml_compute_precision(name: str) -> ct.precision:
    return ct.precision.FLOAT16 if name == "fp16" else ct.precision.FLOAT32


def run_torch(module: torch.nn.Module, input_array: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        output = module(torch.from_numpy(input_array))
    return output.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    model_dir = ensure_model_files(args.model_dir.expanduser().resolve(), args.offline)
    image_path, sample_id = select_image(args)
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    precision_label = args.precision.upper()
    package_path = artifacts_dir / f"UniMERNetTinyEncoder-{precision_label}.mlpackage"
    fixture_dir = artifacts_dir / "fixtures" / f"unimernet-encoder-{args.precision}"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    print("Loading UniMERNet Tiny on CPU...")
    load_started = time.perf_counter()
    model, processor, checkpoint = load_model(
        model_dir, torch.device("cpu"), max_tokens=1536
    )
    encoder = model.model.model.encoder.eval()
    wrapper = UniMERNetEncoderWrapper(encoder).eval()
    load_seconds = time.perf_counter() - load_started

    with Image.open(image_path) as source:
        image_metadata = {"mode": source.mode, "size": list(source.size)}
        grayscale = processor(source.convert("RGB")).unsqueeze(0)
    if tuple(grayscale.shape) != (1, 1, 192, 672):
        raise RuntimeError(f"Unexpected processor output shape: {tuple(grayscale.shape)}")
    encoder_input = grayscale.repeat(1, 3, 1, 1).float()
    input_array = encoder_input.numpy().astype(np.float32)

    print("Running PyTorch reference and tracing fixed-shape Encoder...")
    pytorch_output = run_torch(wrapper, input_array)
    if tuple(pytorch_output.shape) != OUTPUT_SHAPE:
        raise RuntimeError(f"Unexpected PyTorch output shape: {pytorch_output.shape}")
    trace_started = time.perf_counter()
    traced_encoder = torch.jit.trace(wrapper, encoder_input, strict=True)
    traced_encoder = torch.jit.freeze(traced_encoder.eval())
    traced_output = run_torch(traced_encoder, input_array)
    trace_seconds = time.perf_counter() - trace_started

    print(f"Converting fixed-shape {precision_label} ML Program...")
    conversion_started = time.perf_counter()
    coreml_model = ct.convert(
        traced_encoder,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=coreml_compute_precision(args.precision),
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(
                name="pixel_values",
                shape=INPUT_SHAPE,
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="encoder_context", dtype=np.float32)],
    )
    conversion_seconds = time.perf_counter() - conversion_started

    remove_existing_artifact(package_path)
    coreml_model.save(str(package_path))
    print("Reloading and running the saved ML Package...")
    coreml_load_started = time.perf_counter()
    saved_model = ct.models.MLModel(
        str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    coreml_load_seconds = time.perf_counter() - coreml_load_started
    prediction_started = time.perf_counter()
    prediction = saved_model.predict({"pixel_values": input_array})
    prediction_seconds = time.perf_counter() - prediction_started
    output_name = saved_model.get_spec().description.output[0].name
    coreml_output = np.asarray(prediction[output_name], dtype=np.float32)
    if tuple(coreml_output.shape) != OUTPUT_SHAPE:
        raise RuntimeError(f"Unexpected Core ML output shape: {coreml_output.shape}")
    if not np.isfinite(coreml_output).all():
        raise RuntimeError("Core ML output contains non-finite values")

    np.save(fixture_dir / "input.npy", input_array)
    np.save(fixture_dir / "pytorch-output.npy", pytorch_output)
    np.save(fixture_dir / "torchscript-output.npy", traced_output)
    np.save(fixture_dir / "coreml-output.npy", coreml_output)
    report = {
        "report_schema_version": 1,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "unimernet": version("unimernet"),
            "torch": torch.__version__,
            "coremltools": ct.__version__,
            "deployment_target": "macOS 13",
            "compute_precision": args.precision,
            "compute_units": "cpuOnly",
        },
        "checkpoint": checkpoint,
        "source_image": {
            "path": str(image_path),
            "sample_id": sample_id,
            "sha256": sha256(image_path),
            **image_metadata,
        },
        "encoder": {
            "class": f"{encoder.__class__.__module__}.{encoder.__class__.__qualname__}",
            "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
            "input_name": "pixel_values",
            "input_shape_nchw": list(INPUT_SHAPE),
            "output_name": output_name,
            "output_shape": list(OUTPUT_SHAPE),
            "input": tensor_summary(input_array),
            "pytorch_output": tensor_summary(pytorch_output),
            "coreml_output": tensor_summary(coreml_output),
        },
        "comparison": {
            "pytorch_vs_torchscript": compare(pytorch_output, traced_output),
            "pytorch_vs_coreml": compare(pytorch_output, coreml_output),
        },
        "artifacts": {
            "mlpackage": str(package_path),
            "mlpackage_size_bytes": directory_size(package_path),
            "mlpackage_sha256": directory_sha256(package_path),
            "fixtures": str(fixture_dir),
        },
        "timings_seconds": {
            "model_load": load_seconds,
            "torchscript_trace": trace_seconds,
            "coreml_conversion": conversion_seconds,
            "saved_coreml_model_load": coreml_load_seconds,
            "coreml_prediction": prediction_seconds,
        },
    }
    report_path = fixture_dir / "comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    metrics = report["comparison"]["pytorch_vs_coreml"]
    print("\n=== UniMERNet Tiny Encoder conversion complete ===")
    print(f"shape: {list(INPUT_SHAPE)} -> {list(coreml_output.shape)}")
    print(
        "PyTorch vs Core ML: "
        f"max={metrics['max_absolute_error']:.8g}, "
        f"mean={metrics['mean_absolute_error']:.8g}, "
        f"rmse={metrics['root_mean_square_error']:.8g}, "
        f"cosine={metrics['cosine_similarity']:.12f}"
    )
    print(
        f"ML Package: {package_path} "
        f"({report['artifacts']['mlpackage_size_bytes'] / (1024 * 1024):.1f} MB)"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
