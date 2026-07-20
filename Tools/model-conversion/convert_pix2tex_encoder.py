#!/usr/bin/env python3
"""Convert pix2tex Encoder to an FP32 Core ML model with enumerated shapes."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pix2tex.cli import LatexOCR
from timm.models.layers.pool2d_same import MaxPool2dSame
from timm.models.layers.std_conv import StdConv2dSame

from compare_pix2tex_target_height import prepare_target_height


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "test_images" / "formula.png"
DEFAULT_ARTIFACTS = SCRIPT_DIR / "artifacts"
TRACE_INPUT_SHAPE = (1, 1, 64, 448)
CANVAS_HEIGHTS = (32, 64)
CANVAS_WIDTHS = tuple(range(32, 673, 32))
TARGET_FOREGROUND_HEIGHTS = (24, 40)
EMBEDDING_DIMENSION = 256


class Pix2TexEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values)


def fixed_same_padding(
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return SAME padding for dimensions divisible by every layer stride."""
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride
    dilation_height, dilation_width = dilation
    effective_height = dilation_height * (kernel_height - 1) + 1
    effective_width = dilation_width * (kernel_width - 1) + 1
    total_height = max(effective_height - stride_height, 0)
    total_width = max(effective_width - stride_width, 0)
    top = total_height // 2
    left = total_width // 2
    return (
        left,
        total_width - left,
        top,
        total_height - top,
    )


class StaticSamePadStdConv2d(torch.nn.Module):
    def __init__(self, source: StdConv2dSame) -> None:
        super().__init__()
        self.source = source
        self.pad = fixed_same_padding(
            source.kernel_size, source.stride, source.dilation
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        source = self.source
        padded = F.pad(inputs, self.pad)
        weight = F.batch_norm(
            source.weight.reshape(1, source.out_channels, -1),
            None,
            None,
            training=True,
            momentum=0.0,
            eps=source.eps,
        ).reshape_as(source.weight)
        return F.conv2d(
            padded,
            weight,
            source.bias,
            source.stride,
            (0, 0),
            source.dilation,
            source.groups,
        )


class StaticSamePadMaxPool2d(torch.nn.Module):
    def __init__(self, source: MaxPool2dSame) -> None:
        super().__init__()
        self.source = source
        self.pad = fixed_same_padding(
            source.kernel_size, source.stride, source.dilation
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        source = self.source
        padded = F.pad(inputs, self.pad, value=-float("inf"))
        return F.max_pool2d(
            padded,
            source.kernel_size,
            source.stride,
            (0, 0),
            source.dilation,
            source.ceil_mode,
        )


def replace_dynamic_same_padding(
    module: torch.nn.Module, prefix: str = ""
) -> list[str]:
    replaced = []
    for name, child in list(module.named_children()):
        qualified_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, StdConv2dSame) and child.same_pad:
            setattr(module, name, StaticSamePadStdConv2d(child))
            replaced.append(qualified_name)
        elif isinstance(child, MaxPool2dSame):
            setattr(module, name, StaticSamePadMaxPool2d(child))
            replaced.append(qualified_name)
        else:
            replaced.extend(replace_dynamic_same_padding(child, qualified_name))
    return replaced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert pix2tex Encoder with 42 enumerated NCHW shapes: "
            "height 32/64 and width 32...672 in steps of 32."
        )
    )
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
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


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


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
        float(
            np.dot(reference64.ravel(), candidate64.ravel()) / denominator
        )
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


def expected_output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
    _, _, height, width = input_shape
    return (
        1,
        (height // 16) * (width // 16) + 1,
        EMBEDDING_DIMENSION,
    )


def supported_shapes() -> list[tuple[int, int, int, int]]:
    return [
        (1, 1, height, width)
        for height in CANVAS_HEIGHTS
        for width in CANVAS_WIDTHS
    ]


def remove_existing_artifact(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def run_encoder(
    encoder: torch.nn.Module, input_array: np.ndarray
) -> np.ndarray:
    with torch.inference_mode():
        output = encoder(torch.from_numpy(input_array))
    return output.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    package_path = artifacts_dir / "Pix2TexEncoder-Variable32.mlpackage"
    fixture_dir = artifacts_dir / "fixtures" / "encoder-variable32"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pix2tex on CPU...")
    load_started = time.perf_counter()
    ocr = LatexOCR()
    ocr.args.device = "cpu"
    ocr.model.to("cpu").eval()
    original_encoder = ocr.model.encoder.eval()
    export_encoder = copy.deepcopy(original_encoder).eval()
    replaced_modules = replace_dynamic_same_padding(export_encoder)
    load_seconds = time.perf_counter() - load_started
    print(f"Replaced {len(replaced_modules)} dynamic SAME-padding modules")

    original_wrapper = Pix2TexEncoderWrapper(original_encoder).eval()
    export_wrapper = Pix2TexEncoderWrapper(export_encoder).eval()
    trace_input = torch.zeros(TRACE_INPUT_SHAPE, dtype=torch.float32)

    print("Tracing Encoder with static SAME padding...")
    trace_started = time.perf_counter()
    traced_encoder = torch.jit.trace(export_wrapper, trace_input, strict=True)
    traced_encoder = torch.jit.freeze(traced_encoder.eval())
    trace_seconds = time.perf_counter() - trace_started

    shapes = supported_shapes()
    print(f"Converting FP32 ML Program with {len(shapes)} enumerated shapes...")
    conversion_started = time.perf_counter()
    coreml_model = ct.convert(
        traced_encoder,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(
                name="pixel_values",
                shape=ct.EnumeratedShapes(
                    shapes=shapes,
                    default=TRACE_INPUT_SHAPE,
                ),
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name="encoder_context", dtype=np.float32)],
    )
    conversion_seconds = time.perf_counter() - conversion_started

    remove_existing_artifact(package_path)
    coreml_model.save(str(package_path))

    print("Reloading saved ML Package...")
    coreml_load_started = time.perf_counter()
    saved_coreml_model = ct.models.MLModel(
        str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    coreml_load_seconds = time.perf_counter() - coreml_load_started
    output_name = saved_coreml_model.get_spec().description.output[0].name

    print("Validating all enumerated shapes...")
    validation_started = time.perf_counter()
    shape_validations = []
    for index, shape in enumerate(shapes, start=1):
        seed = shape[2] * 1000 + shape[3]
        input_array = np.random.default_rng(seed).normal(size=shape).astype(np.float32)
        original_array = run_encoder(original_wrapper, input_array)
        export_array = run_encoder(export_wrapper, input_array)
        with torch.inference_mode():
            traced_array = (
                traced_encoder(torch.from_numpy(input_array))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        prediction = saved_coreml_model.predict({"pixel_values": input_array})
        coreml_array = np.asarray(prediction[output_name], dtype=np.float32)
        expected_shape = expected_output_shape(shape)
        if tuple(original_array.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected PyTorch output for {shape}: {original_array.shape}"
            )
        if tuple(coreml_array.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected Core ML output for {shape}: {coreml_array.shape}"
            )
        shape_validations.append(
            {
                "input_shape": list(shape),
                "output_shape": list(expected_shape),
                "original_vs_static_padding": compare(
                    original_array, export_array
                ),
                "original_vs_torchscript": compare(
                    original_array, traced_array
                ),
                "original_vs_coreml": compare(original_array, coreml_array),
            }
        )
        if index == 1 or index % 10 == 0 or index == len(shapes):
            metrics = shape_validations[-1]["original_vs_coreml"]
            print(
                f"[{index}/{len(shapes)}] {shape[2]}x{shape[3]} -> "
                f"{expected_shape[1]} tokens; "
                f"max={metrics['max_absolute_error']:.8g}"
            )
    validation_seconds = time.perf_counter() - validation_started

    print("Validating real preprocessing outputs for target heights 24 and 40...")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    preprocessing_cases = []
    for target_height in TARGET_FOREGROUND_HEIGHTS:
        tensor, preprocessing = prepare_target_height(
            image, target_height, max_width=672, max_height=192
        )
        input_array = tensor.detach().cpu().numpy().astype(np.float32)
        original_array = run_encoder(original_wrapper, input_array)
        prediction = saved_coreml_model.predict({"pixel_values": input_array})
        coreml_array = np.asarray(prediction[output_name], dtype=np.float32)
        case_dir = fixture_dir / f"target-height-{target_height}"
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / "input.npy", input_array)
        np.save(case_dir / "pytorch-output.npy", original_array)
        np.save(case_dir / "coreml-output.npy", coreml_array)
        preprocessing_cases.append(
            {
                "target_foreground_height": target_height,
                "preprocessing": preprocessing,
                "input": tensor_summary(input_array),
                "output": tensor_summary(coreml_array),
                "pytorch_vs_coreml": compare(original_array, coreml_array),
                "fixtures": str(case_dir),
            }
        )

    coreml_metrics = [
        record["original_vs_coreml"] for record in shape_validations
    ]
    static_metrics = [
        record["original_vs_static_padding"] for record in shape_validations
    ]
    spec = saved_coreml_model.get_spec()
    enumerated_count = len(
        spec.description.input[0].type.multiArrayType.enumeratedShapes.shapes
    )
    if enumerated_count != len(shapes):
        raise RuntimeError(
            f"Saved model exposes {enumerated_count} shapes, expected {len(shapes)}"
        )

    report = {
        "report_schema_version": 2,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pix2tex": version("pix2tex"),
            "torch": torch.__version__,
            "coremltools": ct.__version__,
            "deployment_target": "macOS 13",
            "compute_precision": "float32",
            "compute_units": "cpuOnly",
        },
        "source_image": {
            "path": str(image_path),
            "sha256": sha256(image_path),
            "size": list(image.size),
        },
        "encoder": {
            "class": (
                f"{original_encoder.__class__.__module__}."
                f"{original_encoder.__class__.__qualname__}"
            ),
            "input_name": "pixel_values",
            "output_name": output_name,
            "input_shapes_nchw": [list(shape) for shape in shapes],
            "default_input_shape_nchw": list(TRACE_INPUT_SHAPE),
            "output_shape_rule": "[1, (height / 16) * (width / 16) + 1, 256]",
            "parameters": sum(
                parameter.numel() for parameter in original_encoder.parameters()
            ),
        },
        "static_same_padding_export": {
            "assumption": (
                "Every input height and width is divisible by 32, so dynamic "
                "SAME padding is equivalent to fixed asymmetric padding."
            ),
            "replaced_modules": replaced_modules,
            "maximum_error_vs_original": max(
                metric["max_absolute_error"] for metric in static_metrics
            ),
        },
        "artifacts": {
            "mlpackage": str(package_path),
            "mlpackage_size_bytes": directory_size(package_path),
            "fixtures": str(fixture_dir),
        },
        "shape_validation": {
            "enumerated_shapes_in_saved_spec": enumerated_count,
            "validated_shapes": len(shape_validations),
            "maximum_coreml_absolute_error": max(
                metric["max_absolute_error"] for metric in coreml_metrics
            ),
            "maximum_coreml_mean_absolute_error": max(
                metric["mean_absolute_error"] for metric in coreml_metrics
            ),
            "minimum_coreml_cosine_similarity": min(
                metric["cosine_similarity"] for metric in coreml_metrics
            ),
            "records": shape_validations,
        },
        "preprocessing_validation": preprocessing_cases,
        "timings_seconds": {
            "model_load_and_export_copy": load_seconds,
            "torchscript_trace": trace_seconds,
            "coreml_conversion": conversion_seconds,
            "saved_coreml_model_load": coreml_load_seconds,
            "all_shape_validation": validation_seconds,
        },
        "notes": [
            "The original pix2tex Encoder remains the numerical reference.",
            "The saved Core ML output context length varies with input shape.",
            "The existing single-step Decoder still has a fixed context length and must be converted separately.",
        ],
    }
    report_path = fixture_dir / "comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== Variable-shape Encoder conversion complete ===")
    print(f"enumerated shapes: {enumerated_count}")
    print(
        "static padding vs original max error: "
        f"{report['static_same_padding_export']['maximum_error_vs_original']:.8g}"
    )
    print(
        "PyTorch vs Core ML across all shapes: "
        f"max={report['shape_validation']['maximum_coreml_absolute_error']:.8g}, "
        f"mean-max={report['shape_validation']['maximum_coreml_mean_absolute_error']:.8g}, "
        f"min-cosine={report['shape_validation']['minimum_coreml_cosine_similarity']:.12f}"
    )
    print(
        f"ML Package: {package_path} "
        f"({report['artifacts']['mlpackage_size_bytes'] / (1024 * 1024):.1f} MB)"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
