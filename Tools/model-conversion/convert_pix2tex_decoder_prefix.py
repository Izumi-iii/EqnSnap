#!/usr/bin/env python3
"""Convert pix2tex Decoder with fixed padded token and context inputs."""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")

import coremltools as ct
import numpy as np
import torch
from pix2tex.cli import LatexOCR

from convert_pix2tex_decoder_step import (
    CONTEXT_EMBEDDING_DIMENSION,
    DEFAULT_ARTIFACTS,
    DEFAULT_ENCODER_REPORT,
    compare,
    context_lengths_from_encoder_report,
    directory_size,
    pad_context,
    real_encoder_contexts,
    remove_existing_artifact,
    top_candidates,
    version,
)


DEFAULT_MAX_TOKEN_LENGTH = 128
CONTEXT_VALIDATION_PREFIX_LENGTH = 16


class Pix2TexDecoderPrefixWrapper(torch.nn.Module):
    def __init__(
        self, decoder_network: torch.nn.Module, output_size: int
    ) -> None:
        super().__init__()
        self.decoder_network = decoder_network
        self.output_size = output_size

    def forward(
        self,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor,
        encoder_context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.decoder_network(
            input_ids.to(torch.int64),
            mask=token_mask.to(torch.bool),
            context=encoder_context,
            context_mask=context_mask.to(torch.bool),
        )
        last_indices = token_mask.to(torch.int64).sum(dim=1) - 1
        gather_indices = last_indices.reshape(-1, 1, 1).expand(
            -1, 1, self.output_size
        )
        return torch.gather(logits, dim=1, index=gather_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder-report",
        type=Path,
        default=DEFAULT_ENCODER_REPORT,
    )
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--max-token-length", type=int, default=DEFAULT_MAX_TOKEN_LENGTH
    )
    return parser.parse_args()


def pad_tokens(
    tokens: list[int], maximum_length: int, pad_token_id: int
) -> tuple[np.ndarray, np.ndarray]:
    if not tokens or len(tokens) > maximum_length:
        raise ValueError(
            f"Token prefix length must be within 1...{maximum_length}"
        )
    input_ids = np.full(
        (1, maximum_length), pad_token_id, dtype=np.int32
    )
    token_mask = np.zeros((1, maximum_length), dtype=np.int32)
    input_ids[0, : len(tokens)] = tokens
    token_mask[0, : len(tokens)] = 1
    return input_ids, token_mask


def deterministic_prefix(
    length: int, bos_token_id: int, vocabulary_size: int
) -> list[int]:
    if length == 1:
        return [bos_token_id]
    usable_tokens = max(1, vocabulary_size - 4)
    return [bos_token_id] + [
        4 + ((index * 37) % usable_tokens)
        for index in range(length - 1)
    ]


def run_raw_decoder(
    decoder_network: torch.nn.Module,
    tokens: list[int],
    context: np.ndarray,
) -> np.ndarray:
    input_ids = torch.tensor([tokens], dtype=torch.int64)
    with torch.inference_mode():
        logits = decoder_network(
            input_ids, context=torch.from_numpy(context)
        )
    return logits[:, -1:, :].detach().cpu().numpy().astype(np.float32)


def run_padded_decoder(
    wrapper: torch.nn.Module,
    input_ids: np.ndarray,
    token_mask: np.ndarray,
    context: np.ndarray,
    context_mask: np.ndarray,
) -> np.ndarray:
    with torch.inference_mode():
        logits = wrapper(
            torch.from_numpy(input_ids),
            torch.from_numpy(token_mask),
            torch.from_numpy(context),
            torch.from_numpy(context_mask),
        )
    return logits.detach().cpu().numpy().astype(np.float32)


def coreml_predict(
    model: ct.models.MLModel,
    output_name: str,
    input_ids: np.ndarray,
    token_mask: np.ndarray,
    context: np.ndarray,
    context_mask: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        model.predict(
            {
                "input_ids": input_ids,
                "token_mask": token_mask,
                "encoder_context": context,
                "context_mask": context_mask,
            }
        )[output_name],
        dtype=np.float32,
    )


def max_comparison(
    records: list[dict[str, Any]], key: str
) -> dict[str, float]:
    metrics = [record[key] for record in records]
    return {
        "maximum_absolute_error": max(
            metric["max_absolute_error"] for metric in metrics
        ),
        "maximum_mean_absolute_error": max(
            metric["mean_absolute_error"] for metric in metrics
        ),
        "minimum_cosine_similarity": min(
            metric["cosine_similarity"] for metric in metrics
        ),
    }


def main() -> None:
    args = parse_args()
    encoder_report_path = args.encoder_report.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    maximum_token_length = args.max_token_length
    if maximum_token_length < 2:
        raise ValueError("--max-token-length must be at least 2")
    if not encoder_report_path.is_file():
        raise FileNotFoundError(
            f"Variable Encoder report not found: {encoder_report_path}"
        )

    encoder_report = json.loads(encoder_report_path.read_text(encoding="utf-8"))
    context_lengths = context_lengths_from_encoder_report(encoder_report)
    maximum_context_length = max(context_lengths)
    encoder_cases = real_encoder_contexts(
        encoder_report, encoder_report_path
    )
    real_cases_by_height = {
        int(record["target_foreground_height"]): record
        for record in encoder_cases
    }
    token_validation_case = real_cases_by_height.get(40, encoder_cases[-1])
    token_validation_context = np.load(
        token_validation_case["pytorch_path"]
    ).astype(np.float32)
    token_validation_padded, token_validation_context_mask = pad_context(
        token_validation_context, maximum_context_length
    )

    package_path = (
        artifacts_dir
        / (
            f"Pix2TexDecoder-Prefix{maximum_token_length}"
            f"-Context{maximum_context_length}.mlpackage"
        )
    )
    fixture_dir = (
        artifacts_dir
        / "fixtures"
        / (
            f"decoder-prefix{maximum_token_length}"
            f"-context{maximum_context_length}"
        )
    )
    fixture_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pix2tex Decoder on CPU...")
    load_started = time.perf_counter()
    ocr = LatexOCR()
    ocr.args.device = "cpu"
    ocr.model.to("cpu").eval()
    decoder_network = ocr.model.decoder.net.eval()
    output_size = int(ocr.args.num_tokens)
    wrapper = Pix2TexDecoderPrefixWrapper(
        decoder_network, output_size
    ).eval()
    load_seconds = time.perf_counter() - load_started

    trace_tokens = deterministic_prefix(
        min(17, maximum_token_length),
        int(ocr.args.bos_token),
        len(ocr.tokenizer),
    )
    trace_input_ids, trace_token_mask = pad_tokens(
        trace_tokens, maximum_token_length, int(ocr.args.pad_token)
    )
    trace_context = np.zeros(
        (1, maximum_context_length, CONTEXT_EMBEDDING_DIMENSION),
        dtype=np.float32,
    )
    trace_context_mask = np.ones(
        (1, maximum_context_length), dtype=np.int32
    )

    print("Tracing fixed token-prefix Decoder...")
    trace_started = time.perf_counter()
    traced_decoder = torch.jit.trace(
        wrapper,
        tuple(
            torch.from_numpy(array)
            for array in (
                trace_input_ids,
                trace_token_mask,
                trace_context,
                trace_context_mask,
            )
        ),
        strict=True,
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
                shape=(1, maximum_token_length),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="token_mask",
                shape=(1, maximum_token_length),
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
    saved_model = ct.models.MLModel(
        str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    output_name = saved_model.get_spec().description.output[0].name
    expected_output_shape = (1, 1, output_size)

    print(
        f"Validating every token prefix length 1...{maximum_token_length}..."
    )
    token_validation_started = time.perf_counter()
    token_length_records = []
    for length in range(1, maximum_token_length + 1):
        tokens = deterministic_prefix(
            length,
            int(ocr.args.bos_token),
            len(ocr.tokenizer),
        )
        input_ids, token_mask = pad_tokens(
            tokens, maximum_token_length, int(ocr.args.pad_token)
        )
        raw_reference = run_raw_decoder(
            decoder_network, tokens, token_validation_context
        )
        padded_pytorch = run_padded_decoder(
            wrapper,
            input_ids,
            token_mask,
            token_validation_padded,
            token_validation_context_mask,
        )
        with torch.inference_mode():
            traced_output = (
                traced_decoder(
                    torch.from_numpy(input_ids),
                    torch.from_numpy(token_mask),
                    torch.from_numpy(token_validation_padded),
                    torch.from_numpy(token_validation_context_mask),
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        coreml_output = coreml_predict(
            saved_model,
            output_name,
            input_ids,
            token_mask,
            token_validation_padded,
            token_validation_context_mask,
        )
        if tuple(coreml_output.shape) != expected_output_shape:
            raise RuntimeError(
                f"Unexpected logits for token length {length}: "
                f"{coreml_output.shape}"
            )
        reference_argmax = int(raw_reference[0, -1].argmax())
        coreml_argmax = int(coreml_output[0, -1].argmax())
        token_length_records.append(
            {
                "token_length": length,
                "valid_token_mask_elements": int(token_mask.sum()),
                "raw_vs_padded_pytorch": compare(
                    raw_reference, padded_pytorch
                ),
                "padded_pytorch_vs_torchscript": compare(
                    padded_pytorch, traced_output
                ),
                "raw_pytorch_vs_coreml": compare(
                    raw_reference, coreml_output
                ),
                "reference_argmax": reference_argmax,
                "coreml_argmax": coreml_argmax,
                "argmax_match": reference_argmax == coreml_argmax,
            }
        )
        if length == 1 or length % 32 == 0:
            metrics = token_length_records[-1]["raw_pytorch_vs_coreml"]
            print(
                f"[tokens {length}/{maximum_token_length}] "
                f"max={metrics['max_absolute_error']:.8g}; "
                f"argmax={reference_argmax}/{coreml_argmax}"
            )
    token_validation_seconds = (
        time.perf_counter() - token_validation_started
    )

    print(f"Validating {len(context_lengths)} Encoder context lengths...")
    context_validation_records = []
    context_tokens = deterministic_prefix(
        min(CONTEXT_VALIDATION_PREFIX_LENGTH, maximum_token_length),
        int(ocr.args.bos_token),
        len(ocr.tokenizer),
    )
    context_input_ids, context_token_mask = pad_tokens(
        context_tokens, maximum_token_length, int(ocr.args.pad_token)
    )
    for index, context_length in enumerate(context_lengths, start=1):
        raw_context = (
            np.random.default_rng(context_length)
            .normal(
                size=(1, context_length, CONTEXT_EMBEDDING_DIMENSION)
            )
            .astype(np.float32)
        )
        padded_context, context_mask = pad_context(
            raw_context, maximum_context_length
        )
        raw_reference = run_raw_decoder(
            decoder_network, context_tokens, raw_context
        )
        padded_pytorch = run_padded_decoder(
            wrapper,
            context_input_ids,
            context_token_mask,
            padded_context,
            context_mask,
        )
        coreml_output = coreml_predict(
            saved_model,
            output_name,
            context_input_ids,
            context_token_mask,
            padded_context,
            context_mask,
        )
        reference_argmax = int(raw_reference[0, -1].argmax())
        coreml_argmax = int(coreml_output[0, -1].argmax())
        context_validation_records.append(
            {
                "source_context_length": context_length,
                "raw_vs_padded_pytorch": compare(
                    raw_reference, padded_pytorch
                ),
                "raw_pytorch_vs_coreml": compare(
                    raw_reference, coreml_output
                ),
                "reference_argmax": reference_argmax,
                "coreml_argmax": coreml_argmax,
                "argmax_match": reference_argmax == coreml_argmax,
            }
        )
        if index == 1 or index % 8 == 0 or index == len(context_lengths):
            metrics = context_validation_records[-1][
                "raw_pytorch_vs_coreml"
            ]
            print(
                f"[context {index}/{len(context_lengths)}] "
                f"length={context_length}; "
                f"max={metrics['max_absolute_error']:.8g}"
            )

    print("Running real autoregressive argmax parity checks...")
    autoregressive_records = []
    for case in encoder_cases:
        target_height = int(case["target_foreground_height"])
        reference_context = np.load(case["pytorch_path"]).astype(np.float32)
        coreml_context = np.load(case["coreml_path"]).astype(np.float32)
        padded_coreml_context, context_mask = pad_context(
            coreml_context, maximum_context_length
        )
        reference_tokens = [int(ocr.args.bos_token)]
        coreml_tokens = [int(ocr.args.bos_token)]
        steps = []
        divergence_step = None
        stopped_on_eos = False

        while len(reference_tokens) < maximum_token_length:
            reference_logits = run_raw_decoder(
                decoder_network, reference_tokens, reference_context
            )
            input_ids, token_mask = pad_tokens(
                coreml_tokens,
                maximum_token_length,
                int(ocr.args.pad_token),
            )
            coreml_logits = coreml_predict(
                saved_model,
                output_name,
                input_ids,
                token_mask,
                padded_coreml_context,
                context_mask,
            )
            prefixes_match = reference_tokens == coreml_tokens
            next_reference = int(reference_logits[0, -1].argmax())
            next_coreml = int(coreml_logits[0, -1].argmax())
            steps.append(
                {
                    "step": len(reference_tokens),
                    "prefixes_match": prefixes_match,
                    "logits_comparison": (
                        compare(reference_logits, coreml_logits)
                        if prefixes_match
                        else None
                    ),
                    "reference_next_token": next_reference,
                    "coreml_next_token": next_coreml,
                    "next_token_match": next_reference == next_coreml,
                }
            )
            if next_reference != next_coreml:
                divergence_step = len(reference_tokens)
                break
            reference_tokens.append(next_reference)
            coreml_tokens.append(next_coreml)
            if next_reference == int(ocr.args.eos_token):
                stopped_on_eos = True
                break

        case_dir = fixture_dir / f"target-height-{target_height}"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "autoregressive.json").write_text(
            json.dumps(steps, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        step_metrics = [
            record["logits_comparison"]
            for record in steps
            if record["logits_comparison"] is not None
        ]
        autoregressive_records.append(
            {
                "target_foreground_height": target_height,
                "encoder_input_shape": case["encoder_input_shape"],
                "source_context_shape": list(reference_context.shape),
                "steps_compared": len(steps),
                "stopped_on_eos": stopped_on_eos,
                "divergence_step": divergence_step,
                "tokens_match": reference_tokens == coreml_tokens,
                "reference_token_ids": reference_tokens,
                "coreml_token_ids": coreml_tokens,
                "reference_tokens": [
                    ocr.tokenizer.convert_ids_to_tokens(token)
                    for token in reference_tokens
                ],
                "maximum_logits_absolute_error": max(
                    metric["max_absolute_error"]
                    for metric in step_metrics
                ),
                "maximum_logits_mean_absolute_error": max(
                    metric["mean_absolute_error"]
                    for metric in step_metrics
                ),
                "fixtures": str(case_dir),
            }
        )
        print(
            f"[autoregressive h={target_height}] "
            f"steps={len(steps)}, eos={stopped_on_eos}, "
            f"divergence={divergence_step}"
        )

    token_summary = max_comparison(
        token_length_records, "raw_pytorch_vs_coreml"
    )
    context_summary = max_comparison(
        context_validation_records, "raw_pytorch_vs_coreml"
    )
    report = {
        "report_schema_version": 1,
        "environment": {
            "pix2tex": version("pix2tex"),
            "torch": torch.__version__,
            "coremltools": ct.__version__,
            "deployment_target": "macOS 13",
            "compute_precision": "float32",
            "compute_units": "cpuOnly",
        },
        "encoder_contract": {
            "report": str(encoder_report_path),
            "supported_context_lengths": context_lengths,
            "maximum_context_length": maximum_context_length,
        },
        "decoder_contract": {
            "strategy": "fixed padded token prefix and context with masks",
            "input_ids": {
                "shape": [1, maximum_token_length],
                "dtype": "int32",
                "padding_value": int(ocr.args.pad_token),
            },
            "token_mask": {
                "shape": [1, maximum_token_length],
                "dtype": "int32",
                "valid_value": 1,
                "padding_value": 0,
            },
            "encoder_context": {
                "shape": [
                    1,
                    maximum_context_length,
                    CONTEXT_EMBEDDING_DIMENSION,
                ],
                "dtype": "float32",
                "padding_value": 0.0,
            },
            "context_mask": {
                "shape": [1, maximum_context_length],
                "dtype": "int32",
                "valid_value": 1,
                "padding_value": 0,
            },
            "logits": {
                "shape": list(expected_output_shape),
                "selection": "last position where token_mask is valid",
            },
        },
        "tokenizer": {
            "vocabulary_size": len(ocr.tokenizer),
            "decoder_output_size": output_size,
            "bos_token_id": int(ocr.args.bos_token),
            "eos_token_id": int(ocr.args.eos_token),
            "pad_token_id": int(ocr.args.pad_token),
        },
        "artifacts": {
            "mlpackage": str(package_path),
            "mlpackage_size_bytes": directory_size(package_path),
            "fixtures": str(fixture_dir),
        },
        "token_length_validation": {
            "validated_lengths": maximum_token_length,
            "argmax_matches": sum(
                record["argmax_match"] for record in token_length_records
            ),
            **token_summary,
            "records": token_length_records,
        },
        "context_length_validation": {
            "validated_lengths": len(context_validation_records),
            "argmax_matches": sum(
                record["argmax_match"]
                for record in context_validation_records
            ),
            **context_summary,
            "records": context_validation_records,
        },
        "autoregressive_validation": autoregressive_records,
        "timings_seconds": {
            "model_load": load_seconds,
            "torchscript_trace": trace_seconds,
            "coreml_conversion": conversion_seconds,
            "all_token_lengths": token_validation_seconds,
        },
        "notes": [
            "This is a correctness baseline without KV cache.",
            "Every decoding step recomputes the full fixed 128-token graph.",
            "Autoregressive validation uses deterministic argmax, not official temperature sampling.",
        ],
    }
    report_path = fixture_dir / "comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== Fixed-prefix Decoder conversion complete ===")
    print(
        f"token lengths: {maximum_token_length}/"
        f"{maximum_token_length} argmax matches"
    )
    print(
        f"token max error={token_summary['maximum_absolute_error']:.8g}; "
        f"context max error={context_summary['maximum_absolute_error']:.8g}"
    )
    print(
        f"ML Package: {package_path} "
        f"({report['artifacts']['mlpackage_size_bytes'] / (1024 * 1024):.1f} MB)"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
