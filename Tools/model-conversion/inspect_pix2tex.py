#!/usr/bin/env python3
"""Inspect the installed pix2tex model and trace one real inference.

The script intentionally uses pix2tex's public LatexOCR entry point, then adds
read-only PyTorch hooks to capture the tensors that cross the model boundaries.
It does not modify pix2tex or save user image/tensor contents in the report.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

# Disable network version checks performed while importing albumentations.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")

import pix2tex
import torch
from PIL import Image
from pix2tex import cli as pix2tex_cli
from pix2tex.cli import LatexOCR


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "test_images" / "formula.png"
DEFAULT_OUTPUT = SCRIPT_DIR / "test-output" / "pix2tex-inspection.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect pix2tex Encoder, Decoder, Tokenizer, weights, and the "
            "intermediate shapes from one inference."
        )
    )
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PyTorch seed used by pix2tex's multinomial Decoder sampling",
    )
    parser.add_argument(
        "--top-logits",
        type=int,
        default=5,
        help="Number of top raw logits to record for every Decoder step",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Skip the optional pix2tex Image Resizer during inference",
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def class_name(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def tensor_info(
    value: Any, *, include_statistics: bool = False
) -> dict[str, Any] | None:
    if not torch.is_tensor(value):
        return None
    info = {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
    }
    if include_statistics and value.numel() > 0:
        sample = value.detach().float().cpu()
        info["statistics"] = {
            "min": float(sample.min()),
            "max": float(sample.max()),
            "mean": float(sample.mean()),
            "std": float(sample.std(unbiased=False)),
        }
    return info


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "size_mb": path.stat().st_size / (1024 * 1024),
        "sha256": sha256(path),
    }


def module_report(module: torch.nn.Module) -> dict[str, Any]:
    parameters = list(module.named_parameters())
    total = sum(parameter.numel() for _, parameter in parameters)
    trainable = sum(
        parameter.numel() for _, parameter in parameters if parameter.requires_grad
    )
    bytes_by_dtype: dict[str, int] = defaultdict(int)

    largest_parameters = []
    for name, parameter in parameters:
        byte_count = parameter.numel() * parameter.element_size()
        dtype = str(parameter.dtype).removeprefix("torch.")
        bytes_by_dtype[dtype] += byte_count
        largest_parameters.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": dtype,
                "parameters": parameter.numel(),
                "size_mb": byte_count / (1024 * 1024),
            }
        )

    largest_parameters.sort(key=lambda item: item["parameters"], reverse=True)
    return {
        "class": class_name(module),
        "parameters": total,
        "trainable_parameters": trainable,
        "parameter_size_mb": sum(bytes_by_dtype.values()) / (1024 * 1024),
        "parameter_bytes_by_dtype": dict(sorted(bytes_by_dtype.items())),
        "state_dict_tensor_count": len(module.state_dict()),
        "largest_parameters": largest_parameters[:10],
    }


def token_description(tokenizer: Any, token_id: int) -> dict[str, Any]:
    token = tokenizer.convert_ids_to_tokens(token_id)
    return {
        "id": token_id,
        "token": token,
        "decodes_to": tokenizer.decode([token_id]),
        "in_tokenizer_vocab": token is not None,
    }


def tokenizer_report(ocr: LatexOCR, tokenizer_path: Path) -> dict[str, Any]:
    tokenizer_json = json.loads(tokenizer_path.read_text())
    configured_tokens = {
        "pad": token_description(ocr.tokenizer, int(ocr.args.pad_token)),
        "bos": token_description(ocr.tokenizer, int(ocr.args.bos_token)),
        "eos": token_description(ocr.tokenizer, int(ocr.args.eos_token)),
    }
    added_tokens = [
        {
            "id": item.get("id"),
            "content": item.get("content"),
            "special": item.get("special"),
        }
        for item in tokenizer_json.get("added_tokens", [])
    ]
    vocabulary_size = len(ocr.tokenizer)
    decoder_output_size = int(ocr.args.num_tokens)

    return {
        "class": class_name(ocr.tokenizer),
        "path": str(tokenizer_path),
        "sha256": sha256(tokenizer_path),
        "vocabulary_size": vocabulary_size,
        "decoder_output_size": decoder_output_size,
        "decoder_output_minus_vocabulary": decoder_output_size - vocabulary_size,
        "tokenizer_api_special_tokens_map": ocr.tokenizer.special_tokens_map,
        "tokenizer_api_special_token_ids": ocr.tokenizer.all_special_ids,
        "configured_special_tokens": configured_tokens,
        "tokenizer_json_added_tokens": added_tokens,
        "note": (
            "pix2tex config is the source of truth for PAD/BOS/EOS IDs. "
            "PreTrainedTokenizerFast does not expose them through its special-token API."
        ),
    }


def selected_model_config(ocr: LatexOCR) -> dict[str, Any]:
    keys = (
        "encoder_structure",
        "backbone_layers",
        "channels",
        "max_height",
        "max_width",
        "min_height",
        "min_width",
        "patch_size",
        "dim",
        "encoder_depth",
        "num_layers",
        "heads",
        "num_tokens",
        "max_seq_len",
        "pad_token",
        "bos_token",
        "eos_token",
        "temperature",
        "decoder_args",
    )
    return {key: ocr.args.get(key) for key in keys}


def top_logits(
    logits: torch.Tensor, tokenizer: Any, count: int
) -> list[dict[str, Any]]:
    values, indices = torch.topk(
        logits[0, -1].detach().float().cpu(), k=min(count, logits.shape[-1])
    )
    return [
        {
            "id": int(token_id),
            "token": tokenizer.convert_ids_to_tokens(int(token_id)),
            "logit": float(value),
            "in_tokenizer_vocab": (
                tokenizer.convert_ids_to_tokens(int(token_id)) is not None
            ),
        }
        for value, token_id in zip(values.tolist(), indices.tolist())
    ]


def configure_device(ocr: LatexOCR, device: torch.device) -> None:
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    ocr.args.device = str(device)
    ocr.model.to(device)
    if ocr.image_resizer is not None:
        ocr.image_resizer.to(device)


def trace_inference(
    ocr: LatexOCR,
    image: Image.Image,
    temperature: float,
    use_resizer: bool,
    top_logit_count: int,
) -> tuple[str, dict[str, Any]]:
    trace: dict[str, Any] = {
        "image_resizer_calls": [],
        "decoder_steps": [],
    }
    handles = []

    def image_resizer_hook(module: torch.nn.Module, inputs: tuple, output: Any) -> None:
        output_tensor = output if torch.is_tensor(output) else None
        predicted_width = None
        if output_tensor is not None:
            predicted_width = (
                int(output_tensor.detach().argmax(dim=-1).cpu().item()) + 1
            ) * 32
        trace["image_resizer_calls"].append(
            {
                "input": (
                    tensor_info(inputs[0], include_statistics=True)
                    if inputs
                    else None
                ),
                "output": tensor_info(output, include_statistics=True),
                "predicted_width": predicted_width,
            }
        )

    def patch_embed_hook(module: torch.nn.Module, inputs: tuple, output: Any) -> None:
        trace["patch_embedding"] = {
            "class": class_name(module),
            "input": (
                tensor_info(inputs[0], include_statistics=True) if inputs else None
            ),
            "output": tensor_info(output, include_statistics=True),
        }

    def encoder_hook(module: torch.nn.Module, inputs: tuple, output: Any) -> None:
        trace["encoder"] = {
            "class": class_name(module),
            "input": (
                tensor_info(inputs[0], include_statistics=True) if inputs else None
            ),
            "output": tensor_info(output, include_statistics=True),
        }

    def decoder_hook(
        module: torch.nn.Module,
        inputs: tuple,
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        input_ids = inputs[0] if inputs else None
        trace["decoder_steps"].append(
            {
                "step": len(trace["decoder_steps"]),
                "input_ids": tensor_info(input_ids),
                "context": tensor_info(kwargs.get("context")),
                "mask": tensor_info(kwargs.get("mask")),
                "output_logits": tensor_info(output),
                "top_raw_logits": (
                    top_logits(output, ocr.tokenizer, top_logit_count)
                    if torch.is_tensor(output)
                    else []
                ),
            }
        )

    if ocr.image_resizer is not None:
        handles.append(ocr.image_resizer.register_forward_hook(image_resizer_hook))
    handles.append(ocr.model.encoder.patch_embed.register_forward_hook(patch_embed_hook))
    handles.append(ocr.model.encoder.register_forward_hook(encoder_hook))
    handles.append(
        ocr.model.decoder.net.register_forward_hook(decoder_hook, with_kwargs=True)
    )

    original_generate = ocr.model.generate

    def generate_with_capture(*args: Any, **kwargs: Any) -> torch.Tensor:
        generated = original_generate(*args, **kwargs)
        trace["generated_token_tensor"] = tensor_info(generated)
        trace["generated_token_ids"] = [int(value) for value in generated[0].tolist()]
        trace["generated_tokens"] = [
            ocr.tokenizer.convert_ids_to_tokens(token_id)
            for token_id in trace["generated_token_ids"]
        ]
        trace["raw_tokenizer_decode"] = ocr.tokenizer.decode(
            trace["generated_token_ids"]
        )
        return generated

    ocr.model.generate = generate_with_capture
    ocr.args.temperature = temperature

    started = time.perf_counter()
    try:
        with torch.inference_mode():
            latex = ocr(image.copy(), resize=use_resizer)
        if ocr.args.device == "mps":
            torch.mps.synchronize()
    finally:
        ocr.model.generate = original_generate
        for handle in handles:
            handle.remove()

    trace["inference_seconds"] = time.perf_counter() - started
    trace["official_post_processed_latex"] = latex
    trace["decoder_step_count"] = len(trace["decoder_steps"])
    trace["used_image_resizer"] = bool(
        use_resizer and ocr.image_resizer is not None
    )
    return latex, trace


def print_summary(report: dict[str, Any]) -> None:
    model = report["model"]
    tokenizer = report["tokenizer"]
    trace = report["inference"]
    encoder = trace.get("encoder", {})
    decoder_steps = trace.get("decoder_steps", [])

    print("\n=== Environment ===")
    for key, value in report["environment"].items():
        print(f"{key}: {value}")

    print("\n=== Model weights ===")
    for name in ("complete_model", "encoder", "decoder", "image_resizer"):
        item = model.get(name)
        if item is not None:
            print(
                f"{name}: {item['parameters']:,} parameters, "
                f"{item['parameter_size_mb']:.1f} MB"
            )
    total = model["total_with_image_resizer"]
    print(
        f"total with Image Resizer: {total['parameters']:,} parameters, "
        f"{total['parameter_size_mb']:.1f} MB"
    )
    for name, item in report["checkpoints"].items():
        if item["exists"]:
            print(f"{name}: {item['size_mb']:.1f} MB, sha256={item['sha256'][:12]}…")

    print("\n=== Tokenizer ===")
    print(f"class: {tokenizer['class']}")
    print(f"vocabulary size: {tokenizer['vocabulary_size']}")
    print(f"Decoder output size: {tokenizer['decoder_output_size']}")
    if tokenizer["decoder_output_minus_vocabulary"] != 0:
        print(
            "warning: Decoder output size and Tokenizer vocabulary differ by "
            f"{tokenizer['decoder_output_minus_vocabulary']} IDs"
        )
    print(
        "configured special tokens: "
        + ", ".join(
            f"{name}={item['id']} ({item['token']})"
            for name, item in tokenizer["configured_special_tokens"].items()
        )
    )

    print("\n=== One inference ===")
    print(
        f"source image: {report['input_image']['size'][0]}x"
        f"{report['input_image']['size'][1]} {report['input_image']['mode']}"
    )
    for index, call in enumerate(trace["image_resizer_calls"]):
        print(
            f"Image Resizer #{index + 1}: "
            f"{call['input']['shape']} -> {call['output']['shape']}, "
            f"predicted width={call['predicted_width']}"
        )
    print(
        f"Encoder: {encoder.get('input', {}).get('shape')} -> "
        f"{encoder.get('output', {}).get('shape')}"
    )
    if decoder_steps:
        first = decoder_steps[0]
        last = decoder_steps[-1]
        print(
            f"Decoder first step: {first['input_ids']['shape']} + "
            f"context {first['context']['shape']} -> {first['output_logits']['shape']}"
        )
        print(
            f"Decoder last step: {last['input_ids']['shape']} + "
            f"context {last['context']['shape']} -> {last['output_logits']['shape']}"
        )
    print(f"generated tokens: {len(trace.get('generated_token_ids', []))}")
    print(f"Decoder calls: {trace['decoder_step_count']}")
    print(f"inference: {trace['inference_seconds']:.3f}s")
    print("LaTeX:")
    print(trace["official_post_processed_latex"])
    print(f"\nReport: {report['report_path']}")


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")
    if args.top_logits < 1:
        raise ValueError("--top-logits must be at least 1")

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device(args.device)
    print(f"Loading pix2tex on {device}...")

    # LatexOCR copies every result to the system clipboard by default.
    pix2tex_cli.clipboard.copy = lambda _: None

    load_started = time.perf_counter()
    ocr = LatexOCR()
    configure_device(ocr, device)
    torch.manual_seed(args.seed)
    load_seconds = time.perf_counter() - load_started

    package_root = Path(pix2tex.__file__).resolve().parent
    model_root = package_root / "model"
    checkpoints_root = model_root / "checkpoints"
    tokenizer_path = model_root / str(ocr.args.tokenizer)

    with Image.open(image_path) as source:
        input_image = {
            "path": str(image_path),
            "sha256": sha256(image_path),
            "size": list(source.size),
            "mode": source.mode,
        }
        rgb_image = source.convert("RGB")

    _, inference_trace = trace_inference(
        ocr=ocr,
        image=rgb_image,
        temperature=args.temperature,
        use_resizer=not args.no_resize,
        top_logit_count=args.top_logits,
    )

    output_path = args.output.expanduser().resolve()
    complete_model_report = module_report(ocr.model)
    image_resizer_report = (
        module_report(ocr.image_resizer) if ocr.image_resizer is not None else None
    )
    report = {
        "report_schema_version": 1,
        "report_path": str(output_path),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "device": str(device),
            "random_seed": args.seed,
            "pix2tex": package_version("pix2tex"),
            "torch": torch.__version__,
            "torchvision": package_version("torchvision"),
            "timm": package_version("timm"),
            "x-transformers": package_version("x-transformers"),
            "transformers": package_version("transformers"),
            "model_load_seconds": load_seconds,
        },
        "input_image": input_image,
        "model_config": selected_model_config(ocr),
        "model": {
            "complete_model": complete_model_report,
            "encoder": module_report(ocr.model.encoder),
            "decoder": module_report(ocr.model.decoder),
            "image_resizer": image_resizer_report,
            "total_with_image_resizer": {
                "parameters": complete_model_report["parameters"]
                + (image_resizer_report["parameters"] if image_resizer_report else 0),
                "parameter_size_mb": complete_model_report["parameter_size_mb"]
                + (
                    image_resizer_report["parameter_size_mb"]
                    if image_resizer_report
                    else 0
                ),
            },
            "boundary_classes": {
                "model": class_name(ocr.model),
                "encoder": class_name(ocr.model.encoder),
                "encoder_patch_embedding": class_name(ocr.model.encoder.patch_embed),
                "decoder_wrapper": class_name(ocr.model.decoder),
                "decoder_network": class_name(ocr.model.decoder.net),
                "decoder_attention_layers": class_name(
                    ocr.model.decoder.net.attn_layers
                ),
                "image_resizer": (
                    class_name(ocr.image_resizer)
                    if ocr.image_resizer is not None
                    else None
                ),
            },
        },
        "checkpoints": {
            "weights": checkpoint_report(checkpoints_root / "weights.pth"),
            "image_resizer": checkpoint_report(
                checkpoints_root / "image_resizer.pth"
            ),
        },
        "tokenizer": tokenizer_report(ocr, tokenizer_path),
        "decoding": {
            "implementation": "pix2tex CustomARWrapper.generate",
            "strategy": "top-k filtering followed by multinomial sampling",
            "filter_threshold": 0.9,
            "temperature": args.temperature,
            "random_seed": args.seed,
        },
        "inference": inference_trace,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_summary(report)


if __name__ == "__main__":
    main()
