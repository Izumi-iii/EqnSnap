#!/usr/bin/env python3
"""Inspect the real UniMERNet Tiny encoder/decoder inference contract."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings(
    "ignore",
    message="The image_processor_class argument is deprecated.*",
    category=FutureWarning,
)

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
DEFAULT_OUTPUT = SCRIPT_DIR / "test-output" / "unimernet-inspection.json"
CACHE_TENSOR_NAMES = (
    "self_attention_key",
    "self_attention_value",
    "cross_attention_key",
    "cross_attention_value",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect UniMERNet Tiny encoder output, one-step decoder I/O, "
            "KV cache shapes, and tokenizer special tokens."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the complete report in addition to writing it to disk.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Fail instead of downloading when model files are missing.",
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tensor_info(tensor: torch.Tensor, include_stats: bool = True) -> dict[str, Any]:
    detached = tensor.detach()
    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).removeprefix("torch."),
        "device": str(detached.device),
    }
    if include_stats and detached.numel() > 0:
        values = detached.float()
        result["min"] = float(values.min().cpu())
        result["max"] = float(values.max().cpu())
        result["mean"] = float(values.mean().cpu())
        result["std"] = float(values.std(unbiased=False).cpu())
    return result


def parameter_info(module: torch.nn.Module) -> dict[str, Any]:
    parameters = list(module.parameters())
    by_dtype: dict[str, int] = {}
    for parameter in parameters:
        dtype = str(parameter.dtype).removeprefix("torch.")
        by_dtype[dtype] = by_dtype.get(dtype, 0) + parameter.numel()
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        "by_dtype": by_dtype,
    }


def select_image(args: argparse.Namespace) -> tuple[Path, dict[str, Any] | None]:
    if args.image is not None:
        image_path = args.image.expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return image_path, None

    manifest_path = args.manifest.expanduser().resolve()
    records = load_manifest(manifest_path)
    if not records:
        raise RuntimeError(f"Manifest contains no samples: {manifest_path}")
    if args.sample_id is None:
        record = records[0]
    else:
        record = next(
            (candidate for candidate in records if candidate["id"] == args.sample_id),
            None,
        )
        if record is None:
            raise RuntimeError(f"Sample ID not found in manifest: {args.sample_id}")
    image_path = (manifest_path.parent / record["image"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if record.get("sha256") and sha256(image_path) != record["sha256"]:
        raise RuntimeError(f"Image hash differs from manifest: {image_path}")
    return image_path, record


def cache_info(past_key_values: tuple[tuple[torch.Tensor, ...], ...]) -> dict[str, Any]:
    layers = []
    for layer_index, layer_cache in enumerate(past_key_values):
        tensors = []
        for tensor_index, tensor in enumerate(layer_cache):
            name = (
                CACHE_TENSOR_NAMES[tensor_index]
                if tensor_index < len(CACHE_TENSOR_NAMES)
                else f"unknown_{tensor_index}"
            )
            tensors.append({"name": name, **tensor_info(tensor, include_stats=False)})
        layers.append({"layer": layer_index, "tensor_count": len(layer_cache), "tensors": tensors})
    return {"layer_count": len(past_key_values), "layers": layers}


def special_token_info(tokenizer: Any) -> dict[str, Any]:
    names = ("bos", "pad", "eos", "unk", "mask", "sep", "cls")
    values = {}
    for name in names:
        values[name] = {
            "token": getattr(tokenizer, f"{name}_token", None),
            "id": getattr(tokenizer, f"{name}_token_id", None),
        }
    return values


def coreml_recommendation(
    encoder_input: torch.Tensor,
    encoder_context: torch.Tensor,
    first_cache: tuple[tuple[torch.Tensor, ...], ...],
    decoder_config: Any,
) -> dict[str, Any]:
    cache_tensor_count = sum(len(layer) for layer in first_cache)
    first_layer = first_cache[0]
    key_dimension = int(first_layer[0].shape[-1])
    value_dimension = int(first_layer[1].shape[-1])
    cross_cache_lengths = sorted(
        {int(layer[index].shape[2]) for layer in first_cache for index in (2, 3)}
    )
    return {
        "initial_conversion_split": "encoder + fixed-prefix decoder without cache",
        "initial_conversion_reason": (
            "This preserves one copy of the 81M-parameter decoder and avoids making 32 "
            "cache tensors part of the first Core ML conversion contract."
        ),
        "encoder": {
            "input": "pixel_values",
            "input_shape": list(encoder_input.shape),
            "output": "encoder_hidden_states",
            "output_shape": list(encoder_context.shape),
            "note": "Export the internal encoder after the grayscale-to-RGB repeat.",
        },
        "decoder": {
            "initial_contract": [
                "fixed maximum input_ids prefix",
                "token attention_mask",
                "encoder_hidden_states",
            ],
            "initial_outputs": ["last-position logits"],
            "max_position_embeddings": int(decoder_config.max_position_embeddings),
        },
        "cache_optimization_candidate": {
            "inputs": [
                "input_ids for the current token",
                "attention_mask for past plus current tokens",
                "encoder_hidden_states",
                f"{cache_tensor_count} flattened cache tensors after the first step",
            ],
            "outputs": ["current-token logits", f"{cache_tensor_count} updated cache tensors"],
            "decoder_layers": len(first_cache),
            "attention_heads": int(decoder_config.decoder_attention_heads),
            "key_dimension_per_head": key_dimension,
            "value_dimension_per_head": value_dimension,
            "encoder_context_lengths_in_cross_cache": cross_cache_lengths,
            "note": (
                "The cache path is numerically valid, but Core ML conversion must prove that "
                "growing self-cache shapes work without duplicating the decoder weights. "
                "Cross-attention cache stays constant after the first step."
            ),
        },
        "rejected_initial_split": {
            "choice": "single monolithic generate model",
            "reason": (
                "Autoregressive control flow, EOS handling, and growing cache shapes should "
                "remain explicit in Swift/Core ML wrappers."
            ),
        },
    }


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    model_dir = ensure_model_files(args.model_dir.expanduser().resolve(), args.offline)
    image_path, manifest_record = select_image(args)
    model, processor, checkpoint = load_model(model_dir, device, max_tokens=1536)

    donut_model = model.model
    vision_model = donut_model.model
    encoder = vision_model.encoder
    decoder = vision_model.decoder
    tokenizer = model.tokenizer.tokenizer

    with Image.open(image_path) as image:
        source_image = {"mode": image.mode, "size": list(image.size)}
        processed = processor(image.convert("RGB")).unsqueeze(0).to(device)
    encoder_input = processed.repeat(1, 3, 1, 1) if processed.shape[1] == 1 else processed

    bos_token_id = int(tokenizer.bos_token_id)
    eos_token_id = int(tokenizer.eos_token_id)
    first_input_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
    first_attention_mask = torch.ones_like(first_input_ids)

    with torch.inference_mode():
        encoder_outputs = encoder(encoder_input, return_dict=True)
        encoder_context = encoder_outputs.last_hidden_state
        first_output = decoder(
            input_ids=first_input_ids,
            attention_mask=first_attention_mask,
            encoder_hidden_states=encoder_context,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        first_next_token = first_output.logits[:, -1, :].argmax(dim=-1)

        cached_input_ids = first_next_token.unsqueeze(1)
        cached_attention_mask = torch.ones((1, 2), dtype=torch.long, device=device)
        cached_output = decoder(
            input_ids=cached_input_ids,
            attention_mask=cached_attention_mask,
            encoder_hidden_states=encoder_context,
            encoder_attention_mask=None,
            past_key_values=first_output.past_key_values,
            use_cache=True,
            return_dict=True,
        )

        full_input_ids = torch.cat((first_input_ids, cached_input_ids), dim=1)
        full_attention_mask = torch.ones_like(full_input_ids)
        full_output = decoder(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            encoder_hidden_states=encoder_context,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )

    cached_logits = cached_output.logits[:, -1, :].float()
    full_logits = full_output.logits[:, -1, :].float()
    difference = (cached_logits - full_logits).abs()
    cached_next_token_id = int(cached_logits.argmax(dim=-1).item())
    full_next_token_id = int(full_logits.argmax(dim=-1).item())

    sample_text = r"x^2 + y^2 = z^2"
    sample_encoding = tokenizer(sample_text, add_special_tokens=True)
    first_token_id = int(first_next_token.item())
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "unimernet": package_version("unimernet"),
        },
        "model": {
            "model_dir": str(model_dir),
            **checkpoint,
            "types": {
                "root": type(model).__name__,
                "encoder_decoder": type(donut_model).__name__,
                "vision_encoder_decoder": type(vision_model).__name__,
                "encoder": type(encoder).__name__,
                "decoder": type(decoder).__name__,
            },
            "parameters": {
                "total": parameter_info(model),
                "encoder": parameter_info(encoder),
                "decoder": parameter_info(decoder),
            },
            "decoder_config": {
                "vocab_size": int(decoder.config.vocab_size),
                "d_model": int(decoder.config.d_model),
                "layers": int(decoder.config.decoder_layers),
                "attention_heads": int(decoder.config.decoder_attention_heads),
                "standard_head_dimension": int(
                    decoder.config.d_model // decoder.config.decoder_attention_heads
                ),
                "key_dimension_per_head": int(first_output.past_key_values[0][0].shape[-1]),
                "value_dimension_per_head": int(first_output.past_key_values[0][1].shape[-1]),
                "qk_squeeze_factor": int(
                    first_output.past_key_values[0][1].shape[-1]
                    // first_output.past_key_values[0][0].shape[-1]
                ),
                "decoder_start_token_id": vision_model.config.decoder_start_token_id,
                "max_position_embeddings": int(decoder.config.max_position_embeddings),
                "use_cache": bool(decoder.config.use_cache),
                "counting_input_used_in_inference": False,
            },
        },
        "tokenizer": {
            "type": type(tokenizer).__name__,
            "length": len(tokenizer),
            "vocab_size": int(tokenizer.vocab_size),
            "special_tokens": special_token_info(tokenizer),
            "all_special_tokens": tokenizer.all_special_tokens,
            "all_special_ids": tokenizer.all_special_ids,
            "sample": {
                "text": sample_text,
                "input_ids": sample_encoding["input_ids"],
                "tokens": tokenizer.convert_ids_to_tokens(sample_encoding["input_ids"]),
                "decoded": tokenizer.decode(sample_encoding["input_ids"]),
            },
        },
        "sample": {
            "manifest_record_id": manifest_record.get("id") if manifest_record else None,
            "image": str(image_path),
            "source_image": source_image,
        },
        "preprocessing": {
            "processor_output": tensor_info(processed),
            "encoder_input": tensor_info(encoder_input),
            "channel_adaptation": (
                "repeat grayscale channel 3 times"
                if processed.shape[1] == 1
                else "none"
            ),
        },
        "encoder": {
            "output": tensor_info(encoder_context),
            "context_length": int(encoder_context.shape[1]),
            "hidden_size": int(encoder_context.shape[2]),
        },
        "decoder_first_step": {
            "input_ids": first_input_ids.cpu().tolist(),
            "attention_mask": first_attention_mask.cpu().tolist(),
            "encoder_hidden_states": tensor_info(encoder_context, include_stats=False),
            "logits": tensor_info(first_output.logits),
            "argmax_token": {
                "id": first_token_id,
                "token": tokenizer.convert_ids_to_tokens(first_token_id),
                "decoded": tokenizer.decode([first_token_id]),
                "is_eos": first_token_id == eos_token_id,
            },
            "past_key_values": cache_info(first_output.past_key_values),
        },
        "decoder_cached_step": {
            "input_ids": cached_input_ids.cpu().tolist(),
            "attention_mask": cached_attention_mask.cpu().tolist(),
            "logits": tensor_info(cached_output.logits),
            "argmax_token": {
                "id": cached_next_token_id,
                "token": tokenizer.convert_ids_to_tokens(cached_next_token_id),
                "decoded": tokenizer.decode([cached_next_token_id]),
                "is_eos": cached_next_token_id == eos_token_id,
            },
            "past_key_values": cache_info(cached_output.past_key_values),
        },
        "cached_vs_full_prefix": {
            "full_prefix_input_ids": full_input_ids.cpu().tolist(),
            "compared_prefix_length": 2,
            "max_absolute_logit_difference": float(difference.max().cpu()),
            "mean_absolute_logit_difference": float(difference.mean().cpu()),
            "cached_argmax_token_id": cached_next_token_id,
            "full_prefix_argmax_token_id": full_next_token_id,
            "argmax_matches": cached_next_token_id == full_next_token_id,
        },
        "coreml_recommendation": coreml_recommendation(
            encoder_input,
            encoder_context,
            first_output.past_key_values,
            decoder.config,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.print_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Encoder: {list(encoder_input.shape)} -> {list(encoder_context.shape)}")
    print(
        "Decoder first step: "
        f"{list(first_input_ids.shape)} -> {list(first_output.logits.shape)}, "
        f"next token {first_token_id}"
    )
    print(
        "KV cache: "
        f"{len(first_output.past_key_values)} layers x "
        f"{len(first_output.past_key_values[0])} tensors, "
        f"key/value dimensions "
        f"{first_output.past_key_values[0][0].shape[-1]}/"
        f"{first_output.past_key_values[0][1].shape[-1]}"
    )
    print(
        "Cached/full-prefix comparison: "
        f"max abs diff {float(difference.max().cpu()):.8g}, "
        f"argmax match {cached_next_token_id == full_next_token_id}"
    )
    print(f"Wrote inspection report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
