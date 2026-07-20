#!/usr/bin/env python3
"""Convert one pix2tex Decoder step with padded and masked Encoder context."""

import argparse
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
from pix2tex.cli import LatexOCR


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = SCRIPT_DIR / "artifacts"
DEFAULT_ENCODER_REPORT = (
    DEFAULT_ARTIFACTS
    / "fixtures"
    / "encoder-variable32"
    / "comparison.json"
)
TOKEN_SHAPE = (1, 1)
CONTEXT_EMBEDDING_DIMENSION = 256


class Pix2TexDecoderStepWrapper(torch.nn.Module):
    def __init__(self, decoder_network: torch.nn.Module) -> None:
        super().__init__()
        self.decoder_network = decoder_network

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder_network(
            input_ids.to(torch.int64),
            context=encoder_context,
            context_mask=context_mask.to(torch.bool),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the BOS-only pix2tex Decoder step. Dynamic Encoder "
            "contexts are padded to the maximum length and masked."
        )
    )
    parser.add_argument(
        "--encoder-report",
        type=Path,
        default=DEFAULT_ENCODER_REPORT,
        help="Variable Encoder comparison report",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    return parser.parse_args()


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def remove_existing_artifact(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


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
    return {
        "shape_match": True,
        "all_finite": bool(
            np.isfinite(reference64).all() and np.isfinite(candidate64).all()
        ),
        "max_absolute_error": float(np.abs(difference).max()),
        "mean_absolute_error": float(np.abs(difference).mean()),
        "root_mean_square_error": float(np.sqrt(np.mean(difference**2))),
        "cosine_similarity": (
            float(np.dot(reference64.ravel(), candidate64.ravel()) / denominator)
            if denominator != 0
            else math.nan
        ),
    }


def top_candidates(
    logits: np.ndarray, tokenizer: Any, count: int = 10
) -> list[dict[str, Any]]:
    last_logits = logits[0, -1]
    indices = np.argsort(last_logits)[-count:][::-1]
    return [
        {
            "id": int(token_id),
            "token": tokenizer.convert_ids_to_tokens(int(token_id)),
            "logit": float(last_logits[token_id]),
        }
        for token_id in indices
    ]


def context_lengths_from_encoder_report(report: dict[str, Any]) -> list[int]:
    shapes = report["encoder"]["input_shapes_nchw"]
    lengths = sorted(
        {
            (int(shape[2]) // 16) * (int(shape[3]) // 16) + 1
            for shape in shapes
        }
    )
    if not lengths:
        raise RuntimeError("Encoder report produced no context lengths")
    return lengths


def real_encoder_contexts(
    encoder_report: dict[str, Any], encoder_report_path: Path
) -> list[dict[str, Any]]:
    cases = []
    for record in encoder_report.get("preprocessing_validation", []):
        fixture_dir = Path(record["fixtures"])
        if not fixture_dir.is_absolute():
            fixture_dir = (encoder_report_path.parent / fixture_dir).resolve()
        pytorch_path = fixture_dir / "pytorch-output.npy"
        coreml_path = fixture_dir / "coreml-output.npy"
        if not pytorch_path.is_file() or not coreml_path.is_file():
            raise FileNotFoundError(
                f"Encoder context fixtures missing under {fixture_dir}"
            )
        cases.append(
            {
                "target_foreground_height": record["target_foreground_height"],
                "encoder_input_shape": record["input"]["shape"],
                "pytorch_path": pytorch_path,
                "coreml_path": coreml_path,
            }
        )
    if not cases:
        raise RuntimeError("Encoder report contains no real validation contexts")
    return cases


def pad_context(
    context: np.ndarray, maximum_length: int
) -> tuple[np.ndarray, np.ndarray]:
    expected_prefix = (1, CONTEXT_EMBEDDING_DIMENSION)
    if context.ndim != 3 or (context.shape[0], context.shape[2]) != expected_prefix:
        raise ValueError(f"Unexpected Encoder context shape: {context.shape}")
    length = int(context.shape[1])
    if length > maximum_length:
        raise ValueError(
            f"Context length {length} exceeds maximum {maximum_length}"
        )
    padded = np.zeros(
        (1, maximum_length, CONTEXT_EMBEDDING_DIMENSION), dtype=np.float32
    )
    mask = np.zeros((1, maximum_length), dtype=np.int32)
    padded[:, :length, :] = context
    mask[:, :length] = 1
    return padded, mask


def run_raw_decoder(
    decoder_network: torch.nn.Module,
    input_ids: torch.Tensor,
    context: np.ndarray,
) -> np.ndarray:
    with torch.inference_mode():
        output = decoder_network(
            input_ids.to(torch.int64), context=torch.from_numpy(context)
        )
    return output.detach().cpu().numpy().astype(np.float32)


def run_padded_decoder(
    wrapper: torch.nn.Module,
    input_ids: torch.Tensor,
    padded_context: np.ndarray,
    context_mask: np.ndarray,
) -> np.ndarray:
    with torch.inference_mode():
        output = wrapper(
            input_ids,
            torch.from_numpy(padded_context),
            torch.from_numpy(context_mask),
        )
    return output.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    encoder_report_path = args.encoder_report.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    if not encoder_report_path.is_file():
        raise FileNotFoundError(
            f"Variable Encoder report not found: {encoder_report_path}. "
            "Run convert_pix2tex_encoder.py first."
        )

    encoder_report = json.loads(encoder_report_path.read_text(encoding="utf-8"))
    context_lengths = context_lengths_from_encoder_report(encoder_report)
    maximum_context_length = max(context_lengths)
    encoder_cases = real_encoder_contexts(encoder_report, encoder_report_path)

    package_path = (
        artifacts_dir
        / f"Pix2TexDecoder-Step1-PaddedContext{maximum_context_length}.mlpackage"
    )
    fixture_dir = (
        artifacts_dir
        / "fixtures"
        / f"decoder-step1-padded-context{maximum_context_length}"
    )
    fixture_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pix2tex Decoder on CPU...")
    load_started = time.perf_counter()
    ocr = LatexOCR()
    ocr.args.device = "cpu"
    ocr.model.to("cpu").eval()
    decoder_network = ocr.model.decoder.net.eval()
    wrapper = Pix2TexDecoderStepWrapper(decoder_network).eval()
    load_seconds = time.perf_counter() - load_started

    input_ids = torch.tensor([[int(ocr.args.bos_token)]], dtype=torch.int32)
    trace_context = torch.zeros(
        (1, maximum_context_length, CONTEXT_EMBEDDING_DIMENSION),
        dtype=torch.float32,
    )
    trace_mask = torch.ones(
        (1, maximum_context_length), dtype=torch.int32
    )

    print("Tracing fixed padded-context single-step Decoder...")
    trace_started = time.perf_counter()
    traced_decoder = torch.jit.trace(
        wrapper, (input_ids, trace_context, trace_mask), strict=True
    )
    traced_decoder = torch.jit.freeze(traced_decoder.eval())
    trace_seconds = time.perf_counter() - trace_started

    print("Converting fixed-shape FP32 ML Program for macOS 13...")
    conversion_started = time.perf_counter()
    coreml_model = ct.convert(
        traced_decoder,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=TOKEN_SHAPE,
                dtype=np.int32,
            ),
            ct.TensorType(
                name="encoder_context",
                shape=(
                    1,
                    maximum_context_length,
                    CONTEXT_EMBEDDING_DIMENSION,
                ),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="context_mask",
                shape=(1, maximum_context_length),
                dtype=np.int32,
            ),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
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
    input_ids_array = input_ids.numpy().astype(np.int32)
    expected_logits_shape = (1, 1, int(ocr.args.num_tokens))

    print(f"Validating {len(context_lengths)} supported context lengths...")
    validation_started = time.perf_counter()
    length_validations = []
    for index, length in enumerate(context_lengths, start=1):
        raw_context = (
            np.random.default_rng(length)
            .normal(size=(1, length, CONTEXT_EMBEDDING_DIMENSION))
            .astype(np.float32)
        )
        padded_context, context_mask = pad_context(
            raw_context, maximum_context_length
        )
        raw_reference = run_raw_decoder(
            decoder_network, input_ids, raw_context
        )
        padded_pytorch = run_padded_decoder(
            wrapper, input_ids, padded_context, context_mask
        )
        with torch.inference_mode():
            traced_array = (
                traced_decoder(
                    input_ids,
                    torch.from_numpy(padded_context),
                    torch.from_numpy(context_mask),
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        coreml_array = np.asarray(
            saved_coreml_model.predict(
                {
                    "input_ids": input_ids_array,
                    "encoder_context": padded_context,
                    "context_mask": context_mask,
                }
            )[output_name],
            dtype=np.float32,
        )
        if tuple(coreml_array.shape) != expected_logits_shape:
            raise RuntimeError(
                f"Unexpected Core ML logits for length {length}: "
                f"{coreml_array.shape}"
            )
        reference_argmax = int(raw_reference[0, -1].argmax())
        coreml_argmax = int(coreml_array[0, -1].argmax())
        length_validations.append(
            {
                "source_context_length": length,
                "padded_context_shape": list(padded_context.shape),
                "valid_mask_elements": int(context_mask.sum()),
                "raw_vs_padded_pytorch": compare(
                    raw_reference, padded_pytorch
                ),
                "padded_pytorch_vs_torchscript": compare(
                    padded_pytorch, traced_array
                ),
                "raw_pytorch_vs_coreml": compare(
                    raw_reference, coreml_array
                ),
                "reference_argmax": reference_argmax,
                "coreml_argmax": coreml_argmax,
                "argmax_match": reference_argmax == coreml_argmax,
            }
        )
        if index == 1 or index % 8 == 0 or index == len(context_lengths):
            metrics = length_validations[-1]["raw_pytorch_vs_coreml"]
            print(
                f"[{index}/{len(context_lengths)}] length={length}; "
                f"max={metrics['max_absolute_error']:.8g}; "
                f"argmax={reference_argmax}/{coreml_argmax}"
            )
    validation_seconds = time.perf_counter() - validation_started

    print("Validating real PyTorch/Core ML Encoder contexts...")
    composed_validations = []
    for case in encoder_cases:
        target_height = int(case["target_foreground_height"])
        pytorch_context = np.load(case["pytorch_path"]).astype(np.float32)
        coreml_context = np.load(case["coreml_path"]).astype(np.float32)
        context_length = int(pytorch_context.shape[1])
        if context_length not in context_lengths:
            raise RuntimeError(
                f"Real context length {context_length} is unsupported"
            )
        pytorch_padded, context_mask = pad_context(
            pytorch_context, maximum_context_length
        )
        coreml_padded, coreml_mask = pad_context(
            coreml_context, maximum_context_length
        )
        if not np.array_equal(context_mask, coreml_mask):
            raise RuntimeError("PyTorch/Core ML Encoder masks differ")

        reference_logits = run_raw_decoder(
            decoder_network, input_ids, pytorch_context
        )
        decoder_only_logits = np.asarray(
            saved_coreml_model.predict(
                {
                    "input_ids": input_ids_array,
                    "encoder_context": pytorch_padded,
                    "context_mask": context_mask,
                }
            )[output_name],
            dtype=np.float32,
        )
        composed_logits = np.asarray(
            saved_coreml_model.predict(
                {
                    "input_ids": input_ids_array,
                    "encoder_context": coreml_padded,
                    "context_mask": coreml_mask,
                }
            )[output_name],
            dtype=np.float32,
        )

        case_dir = fixture_dir / f"target-height-{target_height}"
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / "input-ids.npy", input_ids_array)
        np.save(case_dir / "padded-pytorch-context.npy", pytorch_padded)
        np.save(case_dir / "padded-coreml-context.npy", coreml_padded)
        np.save(case_dir / "context-mask.npy", context_mask)
        np.save(case_dir / "pytorch-reference-logits.npy", reference_logits)
        np.save(case_dir / "coreml-decoder-only-logits.npy", decoder_only_logits)
        np.save(case_dir / "composed-coreml-logits.npy", composed_logits)

        reference_candidates = top_candidates(reference_logits, ocr.tokenizer)
        composed_candidates = top_candidates(composed_logits, ocr.tokenizer)
        composed_validations.append(
            {
                "target_foreground_height": target_height,
                "encoder_input_shape": case["encoder_input_shape"],
                "source_context_shape": list(pytorch_context.shape),
                "padded_context_shape": list(pytorch_padded.shape),
                "valid_mask_elements": int(context_mask.sum()),
                "decoder_only_pytorch_vs_coreml": compare(
                    reference_logits, decoder_only_logits
                ),
                "composed_encoder_decoder_vs_pytorch": compare(
                    reference_logits, composed_logits
                ),
                "pytorch_top_candidates": reference_candidates,
                "composed_coreml_top_candidates": composed_candidates,
                "argmax_match": (
                    reference_candidates[0]["id"]
                    == composed_candidates[0]["id"]
                ),
                "fixtures": str(case_dir),
            }
        )

    coreml_metrics = [
        record["raw_pytorch_vs_coreml"] for record in length_validations
    ]
    padding_metrics = [
        record["raw_vs_padded_pytorch"] for record in length_validations
    ]
    argmax_matches = sum(record["argmax_match"] for record in length_validations)
    report = {
        "report_schema_version": 3,
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
        "encoder_contract": {
            "report": str(encoder_report_path),
            "encoder_input_shapes": encoder_report["encoder"][
                "input_shapes_nchw"
            ],
            "supported_context_lengths": context_lengths,
            "context_length_rule": "(height / 16) * (width / 16) + 1",
        },
        "decoder": {
            "class": (
                f"{decoder_network.__class__.__module__}."
                f"{decoder_network.__class__.__qualname__}"
            ),
            "strategy": "pad context to maximum length and apply context_mask",
            "input_ids": {
                "name": "input_ids",
                "shape": list(TOKEN_SHAPE),
                "dtype": "int32",
                "value": input_ids_array.tolist(),
            },
            "encoder_context": {
                "name": "encoder_context",
                "shape": [
                    1,
                    maximum_context_length,
                    CONTEXT_EMBEDDING_DIMENSION,
                ],
                "dtype": "float32",
                "padding_value": 0.0,
            },
            "context_mask": {
                "name": "context_mask",
                "shape": [1, maximum_context_length],
                "dtype": "int32",
                "valid_value": 1,
                "padding_value": 0,
            },
            "output": {
                "name": output_name,
                "shape": list(expected_logits_shape),
                "dtype": "float32",
            },
            "parameters": sum(
                parameter.numel() for parameter in decoder_network.parameters()
            ),
        },
        "tokenizer": {
            "vocabulary_size": len(ocr.tokenizer),
            "decoder_output_size": int(ocr.args.num_tokens),
            "bos_token_id": int(ocr.args.bos_token),
            "eos_token_id": int(ocr.args.eos_token),
            "pad_token_id": int(ocr.args.pad_token),
        },
        "artifacts": {
            "mlpackage": str(package_path),
            "mlpackage_size_bytes": directory_size(package_path),
            "fixtures": str(fixture_dir),
        },
        "length_validation": {
            "validated_lengths": len(length_validations),
            "argmax_matches": argmax_matches,
            "maximum_padding_equivalence_error": max(
                metric["max_absolute_error"] for metric in padding_metrics
            ),
            "maximum_coreml_absolute_error": max(
                metric["max_absolute_error"] for metric in coreml_metrics
            ),
            "maximum_coreml_mean_absolute_error": max(
                metric["mean_absolute_error"] for metric in coreml_metrics
            ),
            "minimum_coreml_cosine_similarity": min(
                metric["cosine_similarity"] for metric in coreml_metrics
            ),
            "records": length_validations,
        },
        "composed_encoder_decoder_validation": composed_validations,
        "timings_seconds": {
            "model_load": load_seconds,
            "torchscript_trace": trace_seconds,
            "coreml_conversion": conversion_seconds,
            "saved_coreml_model_load": coreml_load_seconds,
            "all_length_validation": validation_seconds,
        },
        "notes": [
            "This model performs only the BOS Decoder step.",
            "The fixed padded context avoids Core ML flexible-shape Decoder numerical instability.",
            "Future autoregressive export must still define generated token-prefix input shapes.",
        ],
    }
    report_path = fixture_dir / "comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== Padded-context single-step Decoder conversion complete ===")
    print(f"supported source lengths: {context_lengths}")
    print(
        f"Core ML context: [1,{maximum_context_length},"
        f"{CONTEXT_EMBEDDING_DIMENSION}] + mask [1,{maximum_context_length}]"
    )
    print(
        "Raw PyTorch vs Core ML across all lengths: "
        f"max={report['length_validation']['maximum_coreml_absolute_error']:.8g}, "
        f"mean-max={report['length_validation']['maximum_coreml_mean_absolute_error']:.8g}, "
        f"min-cosine={report['length_validation']['minimum_coreml_cosine_similarity']:.12f}, "
        f"argmax={argmax_matches}/{len(length_validations)}"
    )
    for record in composed_validations:
        metrics = record["composed_encoder_decoder_vs_pytorch"]
        print(
            f"composed h={record['target_foreground_height']} "
            f"context={record['source_context_shape'][1]}: "
            f"max={metrics['max_absolute_error']:.8g}, "
            f"argmax_match={record['argmax_match']}"
        )
    print(
        f"ML Package: {package_path} "
        f"({report['artifacts']['mlpackage_size_bytes'] / (1024 * 1024):.1f} MB)"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
