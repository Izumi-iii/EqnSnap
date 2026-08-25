#!/usr/bin/env python3
"""Convert UniMERNet Tiny's no-cache fixed-prefix Decoder to Core ML."""

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
    compare,
    coreml_compute_precision,
    directory_sha256,
    directory_size,
    remove_existing_artifact,
    version,
)
from evaluate_unimernet import DEFAULT_MODEL_DIR, ensure_model_files, load_model


MAX_TOKEN_LENGTH = 512
CONTEXT_SHAPE = (1, 126, 512)
VOCABULARY_SIZE = 50_000
VALIDATION_LENGTHS = (1, 2, 16, 128, 448, 512)


class UniMERNetDecoderPrefixWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor,
        encoder_context: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.decoder.model.decoder(
            input_ids=input_ids.to(torch.int64),
            attention_mask=token_mask.to(torch.int64),
            encoder_hidden_states=encoder_context,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            return_dict=False,
        )[0]
        last_indices = token_mask.to(torch.int64).sum(dim=1) - 1
        gather_indices = last_indices.reshape(-1, 1, 1).expand(-1, 1, 512)
        last_hidden_state = torch.gather(hidden_states, dim=1, index=gather_indices)
        return self.decoder.lm_head(last_hidden_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--encoder-report", type=Path, default=None)
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


def pad_tokens(tokens: list[int], pad_token_id: int) -> tuple[np.ndarray, np.ndarray]:
    if not tokens or len(tokens) > MAX_TOKEN_LENGTH:
        raise ValueError(f"Token prefix length must be within 1...{MAX_TOKEN_LENGTH}")
    input_ids = np.full((1, MAX_TOKEN_LENGTH), pad_token_id, dtype=np.int32)
    token_mask = np.zeros((1, MAX_TOKEN_LENGTH), dtype=np.int32)
    input_ids[0, : len(tokens)] = tokens
    token_mask[0, : len(tokens)] = 1
    return input_ids, token_mask


def deterministic_prefix(
    length: int, bos_token_id: int, vocabulary_size: int
) -> list[int]:
    if length == 1:
        return [bos_token_id]
    usable_tokens = vocabulary_size - 4
    return [bos_token_id] + [
        4 + ((index * 37) % usable_tokens) for index in range(length - 1)
    ]


def run_raw_decoder(
    decoder: torch.nn.Module,
    tokens: list[int],
    encoder_context: np.ndarray,
) -> np.ndarray:
    input_ids = torch.tensor([tokens], dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        hidden_states = decoder.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=torch.from_numpy(encoder_context),
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            return_dict=False,
        )[0]
        logits = decoder.lm_head(hidden_states[:, -1:, :])
    return logits.detach().cpu().numpy().astype(np.float32)


def run_wrapper(
    wrapper: torch.nn.Module,
    input_ids: np.ndarray,
    token_mask: np.ndarray,
    encoder_context: np.ndarray,
) -> np.ndarray:
    with torch.inference_mode():
        output = wrapper(
            torch.from_numpy(input_ids),
            torch.from_numpy(token_mask),
            torch.from_numpy(encoder_context),
        )
    return output.detach().cpu().numpy().astype(np.float32)


def coreml_predict(
    model: ct.models.MLModel,
    output_name: str,
    input_ids: np.ndarray,
    token_mask: np.ndarray,
    encoder_context: np.ndarray,
) -> np.ndarray:
    prediction = model.predict(
        {
            "input_ids": input_ids,
            "token_mask": token_mask,
            "encoder_context": encoder_context,
        }
    )
    return np.asarray(prediction[output_name], dtype=np.float32)


def comparison_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    metrics = [record[key] for record in records]
    return {
        "maximum_absolute_error": max(item["max_absolute_error"] for item in metrics),
        "maximum_mean_absolute_error": max(
            item["mean_absolute_error"] for item in metrics
        ),
        "minimum_cosine_similarity": min(item["cosine_similarity"] for item in metrics),
    }


def main() -> None:
    args = parse_args()
    model_dir = ensure_model_files(args.model_dir.expanduser().resolve(), args.offline)
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    encoder_report_path = (
        args.encoder_report.expanduser().resolve()
        if args.encoder_report is not None
        else (
            artifacts_dir
            / "fixtures"
            / f"unimernet-encoder-{args.precision}"
            / "comparison.json"
        )
    )
    if not encoder_report_path.is_file():
        raise FileNotFoundError(f"Encoder report not found: {encoder_report_path}")

    encoder_report = json.loads(encoder_report_path.read_text(encoding="utf-8"))
    if tuple(encoder_report["encoder"]["output_shape"]) != CONTEXT_SHAPE:
        raise RuntimeError("Encoder report does not match the fixed Decoder context contract")
    encoder_fixture_dir = Path(encoder_report["artifacts"]["fixtures"])
    pytorch_context = np.load(encoder_fixture_dir / "pytorch-output.npy").astype(
        np.float32
    )
    coreml_context = np.load(encoder_fixture_dir / "coreml-output.npy").astype(
        np.float32
    )
    if tuple(pytorch_context.shape) != CONTEXT_SHAPE:
        raise RuntimeError(f"Unexpected Encoder context shape: {pytorch_context.shape}")

    precision_label = args.precision.upper()
    package_path = (
        artifacts_dir
        / f"UniMERNetTinyDecoder-Prefix512-{precision_label}.mlpackage"
    )
    fixture_dir = (
        artifacts_dir
        / "fixtures"
        / f"unimernet-decoder-prefix512-{args.precision}"
    )
    fixture_dir.mkdir(parents=True, exist_ok=True)

    print("Loading UniMERNet Tiny Decoder on CPU...")
    load_started = time.perf_counter()
    model, _, checkpoint = load_model(model_dir, torch.device("cpu"), max_tokens=1536)
    decoder = model.model.model.decoder.eval()
    tokenizer = model.tokenizer.tokenizer
    wrapper = UniMERNetDecoderPrefixWrapper(decoder).eval()
    if int(decoder.config.vocab_size) != VOCABULARY_SIZE:
        raise RuntimeError(f"Unexpected vocabulary size: {decoder.config.vocab_size}")
    load_seconds = time.perf_counter() - load_started

    trace_tokens = deterministic_prefix(16, tokenizer.bos_token_id, len(tokenizer))
    trace_input_ids, trace_token_mask = pad_tokens(trace_tokens, tokenizer.pad_token_id)
    print("Tracing no-cache fixed-prefix Decoder...")
    trace_started = time.perf_counter()
    traced_decoder = torch.jit.trace(
        wrapper,
        (
            torch.from_numpy(trace_input_ids),
            torch.from_numpy(trace_token_mask),
            torch.from_numpy(pytorch_context),
        ),
        strict=True,
    )
    traced_decoder = torch.jit.freeze(traced_decoder.eval())
    trace_seconds = time.perf_counter() - trace_started

    print(f"Converting fixed-prefix {precision_label} ML Program for macOS 13...")
    conversion_started = time.perf_counter()
    coreml_model = ct.convert(
        traced_decoder,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=coreml_compute_precision(args.precision),
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(1, MAX_TOKEN_LENGTH),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="token_mask",
                shape=(1, MAX_TOKEN_LENGTH),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="encoder_context",
                shape=CONTEXT_SHAPE,
                dtype=np.float32,
            ),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
    )
    conversion_seconds = time.perf_counter() - conversion_started

    remove_existing_artifact(package_path)
    coreml_model.save(str(package_path))
    print("Reloading the saved ML Package...")
    coreml_load_started = time.perf_counter()
    saved_model = ct.models.MLModel(
        str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    coreml_load_seconds = time.perf_counter() - coreml_load_started
    output_name = saved_model.get_spec().description.output[0].name

    print(f"Validating token lengths {VALIDATION_LENGTHS}...")
    validation_started = time.perf_counter()
    records = []
    for length in VALIDATION_LENGTHS:
        tokens = deterministic_prefix(length, tokenizer.bos_token_id, len(tokenizer))
        input_ids, token_mask = pad_tokens(tokens, tokenizer.pad_token_id)
        raw_output = run_raw_decoder(decoder, tokens, pytorch_context)
        padded_output = run_wrapper(wrapper, input_ids, token_mask, pytorch_context)
        with torch.inference_mode():
            traced_output = traced_decoder(
                torch.from_numpy(input_ids),
                torch.from_numpy(token_mask),
                torch.from_numpy(pytorch_context),
            ).detach().cpu().numpy().astype(np.float32)
        coreml_output = coreml_predict(
            saved_model, output_name, input_ids, token_mask, pytorch_context
        )
        pipeline_output = coreml_predict(
            saved_model, output_name, input_ids, token_mask, coreml_context
        )
        expected_shape = (1, 1, VOCABULARY_SIZE)
        if tuple(coreml_output.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected Core ML output for prefix {length}: {coreml_output.shape}"
            )
        if not np.isfinite(coreml_output).all():
            raise RuntimeError(f"Non-finite Core ML logits for prefix {length}")
        reference_argmax = int(raw_output[0, -1].argmax())
        coreml_argmax = int(coreml_output[0, -1].argmax())
        pipeline_argmax = int(pipeline_output[0, -1].argmax())
        records.append(
            {
                "token_length": length,
                "raw_vs_padded_pytorch": compare(raw_output, padded_output),
                "padded_pytorch_vs_torchscript": compare(padded_output, traced_output),
                "raw_pytorch_vs_coreml_decoder": compare(raw_output, coreml_output),
                "pytorch_pipeline_vs_coreml_pipeline": compare(
                    raw_output, pipeline_output
                ),
                "reference_argmax": reference_argmax,
                "coreml_decoder_argmax": coreml_argmax,
                "coreml_pipeline_argmax": pipeline_argmax,
                "decoder_argmax_match": reference_argmax == coreml_argmax,
                "pipeline_argmax_match": reference_argmax == pipeline_argmax,
            }
        )
        print(
            f"[tokens {length}] max="
            f"{records[-1]['raw_pytorch_vs_coreml_decoder']['max_absolute_error']:.8g}; "
            f"argmax={reference_argmax}/{coreml_argmax}/{pipeline_argmax}"
        )
    validation_seconds = time.perf_counter() - validation_started

    print("Running one real autoregressive parity check...")
    autoregressive_started = time.perf_counter()
    reference_tokens = [int(tokenizer.bos_token_id)]
    coreml_tokens = [int(tokenizer.bos_token_id)]
    autoregressive_steps = []
    coreml_prediction_durations = []
    divergence_step = None
    stopped_on_eos = False
    while len(reference_tokens) <= MAX_TOKEN_LENGTH:
        raw_output = run_raw_decoder(decoder, reference_tokens, pytorch_context)
        input_ids, token_mask = pad_tokens(coreml_tokens, tokenizer.pad_token_id)
        coreml_prediction_started = time.perf_counter()
        pipeline_output = coreml_predict(
            saved_model, output_name, input_ids, token_mask, coreml_context
        )
        coreml_prediction_durations.append(
            time.perf_counter() - coreml_prediction_started
        )
        logits_comparison = compare(raw_output, pipeline_output)
        reference_next = int(raw_output[0, -1].argmax())
        coreml_next = int(pipeline_output[0, -1].argmax())
        autoregressive_steps.append(
            {
                "step": len(reference_tokens),
                "logits_comparison": logits_comparison,
                "reference_next_token": reference_next,
                "coreml_next_token": coreml_next,
                "next_token_match": reference_next == coreml_next,
            }
        )
        reference_tokens.append(reference_next)
        coreml_tokens.append(coreml_next)
        if reference_next != coreml_next:
            divergence_step = len(reference_tokens) - 1
            break
        if reference_next == int(tokenizer.eos_token_id):
            stopped_on_eos = True
            break
    autoregressive_seconds = time.perf_counter() - autoregressive_started

    np.save(fixture_dir / "encoder-context-pytorch.npy", pytorch_context)
    np.save(fixture_dir / "encoder-context-coreml.npy", coreml_context)
    final_tokens = deterministic_prefix(
        VALIDATION_LENGTHS[-1], tokenizer.bos_token_id, len(tokenizer)
    )
    final_input_ids, final_token_mask = pad_tokens(final_tokens, tokenizer.pad_token_id)
    np.save(fixture_dir / "input-ids.npy", final_input_ids)
    np.save(fixture_dir / "token-mask.npy", final_token_mask)

    decoder_summary = comparison_summary(records, "raw_pytorch_vs_coreml_decoder")
    pipeline_summary = comparison_summary(records, "pytorch_pipeline_vs_coreml_pipeline")
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
        "encoder_contract": {
            "report": str(encoder_report_path),
            "context_shape": list(CONTEXT_SHAPE),
            "pytorch_vs_coreml": encoder_report["comparison"]["pytorch_vs_coreml"],
        },
        "decoder_contract": {
            "strategy": "fixed right-padded token prefix with mask, without KV cache",
            "input_ids": {
                "shape": [1, MAX_TOKEN_LENGTH],
                "dtype": "int32",
                "padding_value": int(tokenizer.pad_token_id),
            },
            "token_mask": {
                "shape": [1, MAX_TOKEN_LENGTH],
                "dtype": "int32",
                "valid_value": 1,
                "padding_value": 0,
                "requirement": "valid tokens are contiguous from index zero",
            },
            "encoder_context": {"shape": list(CONTEXT_SHAPE), "dtype": "float32"},
            "logits": {
                "shape": [1, 1, VOCABULARY_SIZE],
                "dtype": "float32",
                "selection": "last position where token_mask is valid",
            },
            "use_kv_cache": False,
        },
        "tokenizer": {
            "vocabulary_size": len(tokenizer),
            "bos_token_id": int(tokenizer.bos_token_id),
            "pad_token_id": int(tokenizer.pad_token_id),
            "eos_token_id": int(tokenizer.eos_token_id),
        },
        "validation": {
            "token_lengths": list(VALIDATION_LENGTHS),
            "decoder_argmax_matches": sum(
                record["decoder_argmax_match"] for record in records
            ),
            "pipeline_argmax_matches": sum(
                record["pipeline_argmax_match"] for record in records
            ),
            "decoder_summary": decoder_summary,
            "pipeline_summary": pipeline_summary,
            "records": records,
        },
        "autoregressive_validation": {
            "source_sample_id": encoder_report["source_image"]["sample_id"],
            "steps_compared": len(autoregressive_steps),
            "stopped_on_eos": stopped_on_eos,
            "reached_token_capacity": (
                not stopped_on_eos and divergence_step is None
            ),
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
            "maximum_logits_absolute_error": max(
                step["logits_comparison"]["max_absolute_error"]
                for step in autoregressive_steps
            ),
            "coreml_prediction_seconds": {
                "total": sum(coreml_prediction_durations),
                "p50": float(np.percentile(coreml_prediction_durations, 50)),
                "p95": float(np.percentile(coreml_prediction_durations, 95)),
            },
            "steps": autoregressive_steps,
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
            "validation": validation_seconds,
            "autoregressive_validation": autoregressive_seconds,
        },
        "notes": [
            "This is the correctness baseline without KV cache.",
            "Every call recomputes all 512 token positions, including masked padding positions.",
            "Only the selected hidden state is projected to the 50,000-token vocabulary.",
        ],
    }
    report_path = fixture_dir / "comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== UniMERNet Tiny fixed-prefix Decoder conversion complete ===")
    print(
        f"contract: [1,{MAX_TOKEN_LENGTH}] + {list(CONTEXT_SHAPE)} "
        f"-> [1,1,{VOCABULARY_SIZE}]"
    )
    print(
        "PyTorch vs Core ML Decoder: "
        f"max={decoder_summary['maximum_absolute_error']:.8g}, "
        f"mean-max={decoder_summary['maximum_mean_absolute_error']:.8g}, "
        f"min-cosine={decoder_summary['minimum_cosine_similarity']:.12f}"
    )
    print(
        f"argmax matches: decoder={report['validation']['decoder_argmax_matches']}/"
        f"{len(records)}, pipeline={report['validation']['pipeline_argmax_matches']}/"
        f"{len(records)}"
    )
    print(
        "autoregressive: "
        f"steps={len(autoregressive_steps)}, eos={stopped_on_eos}, "
        f"divergence={divergence_step}"
    )
    print(
        f"ML Package: {package_path} "
        f"({report['artifacts']['mlpackage_size_bytes'] / (1024 * 1024):.1f} MB)"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
