#!/usr/bin/env python3
"""Convert UniMERNet Tiny's incremental self-KV Decoder step to Core ML."""

from __future__ import annotations

import argparse
import json
import os
import platform
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

from convert_unimernet_encoder import (
    DEFAULT_ARTIFACTS,
    coreml_compute_precision,
    directory_sha256,
    directory_size,
    remove_existing_artifact,
    version,
)
from evaluate_unimernet import DEFAULT_MODEL_DIR, ensure_model_files, load_model


DEFAULT_ENCODER_REPORT = (
    DEFAULT_ARTIFACTS
    / "fixtures"
    / "unimernet-encoder-fp16"
    / "comparison.json"
)
DECODER_LAYERS = 8
ATTENTION_HEADS = 16
KEY_DIMENSION = 16
VALUE_DIMENSION = 32
CONTEXT_SHAPE = (1, 126, 512)
MAXIMUM_PAST_LENGTH = 512
MINIMUM_PAST_LENGTH = 1
VALIDATION_LENGTHS = {1, 2, 16, 128, 256, 448, 511}


class UniMERNetDecoderCachedStepWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        for layer in range(DECODER_LAYERS):
            self.register_buffer(
                f"dummy_cross_key_{layer}",
                torch.zeros((1, ATTENTION_HEADS, 1, KEY_DIMENSION)),
            )
            self.register_buffer(
                f"dummy_cross_value_{layer}",
                torch.zeros((1, ATTENTION_HEADS, 1, VALUE_DIMENSION)),
            )

    def forward(
        self,
        input_id: torch.Tensor,
        encoder_context: torch.Tensor,
        self_mask: torch.Tensor,
        *self_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        past_key_values = []
        for layer in range(DECODER_LAYERS):
            past_key_values.append(
                (
                    self_cache[layer * 2],
                    self_cache[layer * 2 + 1],
                    getattr(self, f"dummy_cross_key_{layer}"),
                    getattr(self, f"dummy_cross_value_{layer}"),
                )
            )
        decoder_outputs = self.decoder.model.decoder(
            input_ids=input_id.to(torch.int64),
            attention_mask=torch.cat(
                (self_mask, torch.ones_like(input_id)), dim=1
            ).to(torch.int64),
            encoder_hidden_states=encoder_context,
            encoder_attention_mask=None,
            past_key_values=tuple(past_key_values),
            use_cache=True,
            return_dict=False,
        )
        logits = self.decoder.lm_head(decoder_outputs[0])
        next_cache = decoder_outputs[1]
        outputs = [logits]
        for layer_cache in next_cache:
            outputs.extend(layer_cache[:2])
        return tuple(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--encoder-report", type=Path, default=DEFAULT_ENCODER_REPORT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Fail instead of downloading when model files are missing.",
    )
    return parser.parse_args()


def self_cache_from_full(
    past_key_values: tuple[tuple[torch.Tensor, ...], ...]
) -> list[np.ndarray]:
    arrays = []
    for layer_cache in past_key_values:
        arrays.extend(
            tensor.detach().cpu().numpy().astype(np.float32)
            for tensor in layer_cache[:2]
        )
    return arrays


def full_cache_from_output(output: Any) -> tuple[tuple[torch.Tensor, ...], ...]:
    return output.past_key_values


def run_pytorch_step(
    decoder: torch.nn.Module,
    token_id: int,
    encoder_context: np.ndarray,
    past_key_values: tuple[tuple[torch.Tensor, ...], ...],
) -> Any:
    past_length = int(past_key_values[0][0].shape[2])
    with torch.inference_mode():
        return decoder(
            input_ids=torch.tensor([[token_id]], dtype=torch.int64),
            attention_mask=torch.ones((1, past_length + 1), dtype=torch.int64),
            encoder_hidden_states=torch.from_numpy(encoder_context),
            encoder_attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )


def coreml_step(
    model: ct.models.MLModel,
    input_names: list[str],
    output_names: list[str],
    token_id: int,
    encoder_context: np.ndarray,
    self_cache: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    cache_length = int(self_cache[0].shape[2])
    self_mask = np.ones((1, cache_length), dtype=np.int32)
    self_mask[:, 0] = 0
    inputs: dict[str, np.ndarray] = {
        "input_id": np.asarray([[token_id]], dtype=np.int32),
        "encoder_context": encoder_context,
        "self_mask": self_mask,
    }
    for name, value in zip(input_names, self_cache):
        inputs[name] = value
    prediction = model.predict(inputs)
    logits = np.asarray(prediction["logits"], dtype=np.float32)
    next_cache = [
        np.asarray(prediction[name], dtype=np.float32) for name in output_names
    ]
    return logits, next_cache


def array_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.abs(
        candidate.astype(np.float64, copy=False)
        - reference.astype(np.float64, copy=False)
    )
    return {
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
    }


def cache_comparison(
    reference: list[np.ndarray], candidate: list[np.ndarray]
) -> dict[str, float]:
    metrics = [
        array_comparison(reference_value, candidate_value)
        for reference_value, candidate_value in zip(reference, candidate)
    ]
    return {
        "max_absolute_error": max(item["max_absolute_error"] for item in metrics),
        "maximum_mean_absolute_error": max(
            item["mean_absolute_error"] for item in metrics
        ),
    }


def without_dummy_cache(cache: list[np.ndarray]) -> list[np.ndarray]:
    return [value[:, :, 1:, :] for value in cache]


def main() -> None:
    args = parse_args()
    model_dir = ensure_model_files(args.model_dir.expanduser().resolve(), args.offline)
    encoder_report_path = args.encoder_report.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    if not encoder_report_path.is_file():
        raise FileNotFoundError(f"Encoder report not found: {encoder_report_path}")
    encoder_report = json.loads(encoder_report_path.read_text(encoding="utf-8"))
    encoder_context = np.load(
        Path(encoder_report["artifacts"]["fixtures"]) / "coreml-output.npy"
    ).astype(np.float32)
    if tuple(encoder_context.shape) != CONTEXT_SHAPE:
        raise RuntimeError(f"Unexpected Encoder context shape: {encoder_context.shape}")

    precision_label = args.precision.upper()
    package_path = (
        artifacts_dir
        / f"UniMERNetTinyDecoder-CachedStep-SelfKV-{precision_label}.mlpackage"
    )
    fixture_dir = (
        artifacts_dir
        / "fixtures"
        / f"unimernet-decoder-cached-step-selfkv-{args.precision}"
    )
    fixture_dir.mkdir(parents=True, exist_ok=True)

    print("Loading UniMERNet Tiny Decoder on CPU...")
    load_started = time.perf_counter()
    root_model, _, checkpoint = load_model(
        model_dir, torch.device("cpu"), max_tokens=1536
    )
    decoder = root_model.model.model.decoder.eval()
    tokenizer = root_model.tokenizer.tokenizer
    wrapper = UniMERNetDecoderCachedStepWrapper(decoder).eval()
    load_seconds = time.perf_counter() - load_started

    with torch.inference_mode():
        first_output = decoder(
            input_ids=torch.tensor([[tokenizer.bos_token_id]], dtype=torch.int64),
            attention_mask=torch.ones((1, 1), dtype=torch.int64),
            encoder_hidden_states=torch.from_numpy(encoder_context),
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
    first_token = int(first_output.logits[0, -1].argmax())
    trace_cache = []
    for _ in range(DECODER_LAYERS):
        trace_cache.extend(
            (
                np.zeros((1, ATTENTION_HEADS, 1, KEY_DIMENSION), dtype=np.float32),
                np.zeros((1, ATTENTION_HEADS, 1, VALUE_DIMENSION), dtype=np.float32),
            )
        )
    trace_inputs = (
        torch.tensor([[tokenizer.bos_token_id]], dtype=torch.int32),
        torch.from_numpy(encoder_context),
        torch.zeros((1, 1), dtype=torch.int32),
        *(torch.from_numpy(value) for value in trace_cache),
    )

    print("Tracing incremental self-KV Decoder step...")
    trace_started = time.perf_counter()
    traced_decoder = torch.jit.trace(wrapper, trace_inputs, strict=True)
    traced_decoder = torch.jit.freeze(traced_decoder.eval())
    trace_seconds = time.perf_counter() - trace_started

    input_cache_names = []
    output_cache_names = []
    for layer in range(DECODER_LAYERS):
        input_cache_names.extend((f"self_key_{layer}", f"self_value_{layer}"))
        output_cache_names.extend(
            (f"next_self_key_{layer}", f"next_self_value_{layer}")
        )
    past_length = ct.RangeDim(
        lower_bound=MINIMUM_PAST_LENGTH,
        upper_bound=MAXIMUM_PAST_LENGTH,
        default=1,
        symbol="past_length",
    )
    coreml_inputs = [
        ct.TensorType(name="input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="encoder_context", shape=CONTEXT_SHAPE, dtype=np.float32),
        ct.TensorType(name="self_mask", shape=(1, past_length), dtype=np.int32),
    ]
    for layer in range(DECODER_LAYERS):
        coreml_inputs.extend(
            (
                ct.TensorType(
                    name=f"self_key_{layer}",
                    shape=(1, ATTENTION_HEADS, past_length, KEY_DIMENSION),
                    dtype=np.float32,
                ),
                ct.TensorType(
                    name=f"self_value_{layer}",
                    shape=(1, ATTENTION_HEADS, past_length, VALUE_DIMENSION),
                    dtype=np.float32,
                ),
            )
        )
    coreml_outputs = [ct.TensorType(name="logits", dtype=np.float32)] + [
        ct.TensorType(name=name, dtype=np.float32) for name in output_cache_names
    ]

    print(f"Converting dynamic-cache {precision_label} ML Program...")
    conversion_started = time.perf_counter()
    coreml_model = ct.convert(
        traced_decoder,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=coreml_compute_precision(args.precision),
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=coreml_inputs,
        outputs=coreml_outputs,
    )
    conversion_seconds = time.perf_counter() - conversion_started
    remove_existing_artifact(package_path)
    coreml_model.save(str(package_path))

    print("Reloading the saved cached-step ML Package...")
    load_coreml_started = time.perf_counter()
    saved_model = ct.models.MLModel(
        str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    coreml_load_seconds = time.perf_counter() - load_coreml_started

    print("Validating masked-dummy BOS prefill...")
    dummy_cache = []
    for _ in range(DECODER_LAYERS):
        dummy_cache.extend(
            (
                np.zeros(
                    (1, ATTENTION_HEADS, 1, KEY_DIMENSION), dtype=np.float32
                ),
                np.zeros(
                    (1, ATTENTION_HEADS, 1, VALUE_DIMENSION), dtype=np.float32
                ),
            )
        )
    prefill_started = time.perf_counter()
    coreml_first_logits, coreml_first_cache = coreml_step(
        saved_model,
        input_cache_names,
        output_cache_names,
        int(tokenizer.bos_token_id),
        encoder_context,
        dummy_cache,
    )
    prefill_seconds = time.perf_counter() - prefill_started
    pytorch_first_logits = (
        first_output.logits.detach().cpu().numpy().astype(np.float32)
    )
    pytorch_first_cache = self_cache_from_full(first_output.past_key_values)
    prefill_logits_metrics = array_comparison(
        pytorch_first_logits, coreml_first_logits
    )
    prefill_cache_metrics = cache_comparison(
        pytorch_first_cache, without_dummy_cache(coreml_first_cache)
    )
    coreml_first_token = int(coreml_first_logits[0, -1].argmax())

    print("Validating growing effective Cache lengths 1...511...")
    pytorch_past = full_cache_from_output(first_output)
    coreml_cache = coreml_first_cache
    validation_records = []
    prediction_durations = []
    argmax_matches = 0
    maximum_logits_error = 0.0
    maximum_cache_error = 0.0
    for current_past_length in range(1, MAXIMUM_PAST_LENGTH):
        token_id = 4 + ((current_past_length * 37) % (len(tokenizer) - 4))
        pytorch_output = run_pytorch_step(
            decoder, token_id, encoder_context, pytorch_past
        )
        prediction_started = time.perf_counter()
        coreml_logits, next_coreml_cache = coreml_step(
            saved_model,
            input_cache_names,
            output_cache_names,
            token_id,
            encoder_context,
            coreml_cache,
        )
        prediction_durations.append(time.perf_counter() - prediction_started)
        pytorch_logits = (
            pytorch_output.logits.detach().cpu().numpy().astype(np.float32)
        )
        next_pytorch_cache = self_cache_from_full(pytorch_output.past_key_values)
        logits_metrics = array_comparison(pytorch_logits, coreml_logits)
        cache_metrics = cache_comparison(
            next_pytorch_cache, without_dummy_cache(next_coreml_cache)
        )
        maximum_logits_error = max(
            maximum_logits_error, logits_metrics["max_absolute_error"]
        )
        maximum_cache_error = max(
            maximum_cache_error, cache_metrics["max_absolute_error"]
        )
        reference_argmax = int(pytorch_logits[0, -1].argmax())
        coreml_argmax = int(coreml_logits[0, -1].argmax())
        argmax_matches += reference_argmax == coreml_argmax
        if current_past_length in VALIDATION_LENGTHS:
            validation_records.append(
                {
                    "past_length": current_past_length,
                    "output_cache_length": int(next_coreml_cache[0].shape[2]),
                    "logits": logits_metrics,
                    "cache": cache_metrics,
                    "reference_argmax": reference_argmax,
                    "coreml_argmax": coreml_argmax,
                    "argmax_match": reference_argmax == coreml_argmax,
                }
            )
            print(
                f"[past {current_past_length}] "
                f"logits-max={logits_metrics['max_absolute_error']:.6g}, "
                f"cache-max={cache_metrics['max_absolute_error']:.6g}, "
                f"argmax={reference_argmax}/{coreml_argmax}"
            )
        pytorch_past = pytorch_output.past_key_values
        coreml_cache = next_coreml_cache

    print("Running one real cached autoregressive parity check...")
    reference_tokens = [int(tokenizer.bos_token_id), first_token]
    coreml_tokens = [int(tokenizer.bos_token_id), coreml_first_token]
    pytorch_past = first_output.past_key_values
    coreml_cache = coreml_first_cache
    divergence_step = 1 if first_token != coreml_first_token else None
    stopped_on_eos = first_token == int(tokenizer.eos_token_id)
    autoregressive_durations = []
    while (
        divergence_step is None
        and not stopped_on_eos
        and len(reference_tokens) <= MAXIMUM_PAST_LENGTH
    ):
        current_token = reference_tokens[-1]
        pytorch_output = run_pytorch_step(
            decoder, current_token, encoder_context, pytorch_past
        )
        prediction_started = time.perf_counter()
        coreml_logits, next_coreml_cache = coreml_step(
            saved_model,
            input_cache_names,
            output_cache_names,
            coreml_tokens[-1],
            encoder_context,
            coreml_cache,
        )
        autoregressive_durations.append(time.perf_counter() - prediction_started)
        reference_next = int(pytorch_output.logits[0, -1].argmax())
        coreml_next = int(coreml_logits[0, -1].argmax())
        reference_tokens.append(reference_next)
        coreml_tokens.append(coreml_next)
        if reference_next != coreml_next:
            divergence_step = len(reference_tokens) - 1
            break
        stopped_on_eos = reference_next == int(tokenizer.eos_token_id)
        pytorch_past = pytorch_output.past_key_values
        coreml_cache = next_coreml_cache

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
        "contract": {
            "input_id": {"shape": [1, 1], "dtype": "int32"},
            "encoder_context": {"shape": list(CONTEXT_SHAPE), "dtype": "float32"},
            "self_cache_inputs": len(input_cache_names),
            "self_cache_outputs": len(output_cache_names),
            "past_length_range": [MINIMUM_PAST_LENGTH, MAXIMUM_PAST_LENGTH],
            "self_mask": {"shape": [1, "past_length"], "dtype": "int32"},
            "key_shape": [1, ATTENTION_HEADS, "past_length", KEY_DIMENSION],
            "value_shape": [1, ATTENTION_HEADS, "past_length", VALUE_DIMENSION],
            "cross_cache_strategy": "recompute from encoder_context using internal dummy cache",
            "logits_shape": [1, 1, 50_000],
        },
        "dynamic_cache_validation": {
            "steps": len(prediction_durations),
            "argmax_matches": argmax_matches,
            "maximum_logits_absolute_error": maximum_logits_error,
            "maximum_cache_absolute_error": maximum_cache_error,
            "prediction_seconds": {
                "total": sum(prediction_durations),
                "p50": float(np.percentile(prediction_durations, 50)),
                "p95": float(np.percentile(prediction_durations, 95)),
            },
            "checkpoints": validation_records,
        },
        "masked_dummy_prefill_validation": {
            "input_cache_length": 1,
            "masked_dummy_slots": 1,
            "output_cache_length": int(coreml_first_cache[0].shape[2]),
            "logits": prefill_logits_metrics,
            "cache": prefill_cache_metrics,
            "reference_next_token": first_token,
            "coreml_next_token": coreml_first_token,
            "argmax_match": first_token == coreml_first_token,
            "prediction_seconds": prefill_seconds,
        },
        "autoregressive_validation": {
            "steps_compared": len(reference_tokens) - 1,
            "stopped_on_eos": stopped_on_eos,
            "divergence_step": divergence_step,
            "token_sequences_match": reference_tokens == coreml_tokens,
            "reference_token_ids": reference_tokens,
            "coreml_token_ids": coreml_tokens,
            "reference_latex": tokenizer.decode(
                reference_tokens[1:], skip_special_tokens=True
            ),
            "coreml_latex": tokenizer.decode(
                coreml_tokens[1:], skip_special_tokens=True
            ),
            "prediction_seconds": {
                "total": sum(autoregressive_durations),
                "p50": float(np.percentile(autoregressive_durations, 50)),
                "p95": float(np.percentile(autoregressive_durations, 95)),
            },
            "initial_cache_source": "Core ML masked-dummy BOS prefill",
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
        },
        "notes": [
            "A permanently masked dummy cache slot avoids zero-length MLMultiArray inputs in Swift.",
            "Cross-attention K/V is recomputed each step and is not exposed as model I/O.",
        ],
    }
    report_path = fixture_dir / "comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== UniMERNet cached-step Decoder conversion complete ===")
    print(
        f"dynamic cache: {argmax_matches}/{len(prediction_durations)} argmax matches, "
        f"logits-max={maximum_logits_error:.6g}"
    )
    print(
        "cached step P50/P95: "
        f"{report['dynamic_cache_validation']['prediction_seconds']['p50']:.4f}s / "
        f"{report['dynamic_cache_validation']['prediction_seconds']['p95']:.4f}s"
    )
    print(
        f"autoregressive: steps={len(reference_tokens) - 1}, "
        f"eos={stopped_on_eos}, divergence={divergence_step}"
    )
    print(
        f"ML Package: {package_path} "
        f"({report['artifacts']['mlpackage_size_bytes'] / (1024 * 1024):.1f} MB)"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
