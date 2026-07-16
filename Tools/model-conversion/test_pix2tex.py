import argparse
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any

import pix2tex
import torch
from PIL import Image
from pix2tex import cli as pix2tex_cli
from pix2tex.cli import LatexOCR


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "test-output" / "pix2tex-results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pix2tex on formula images.")
    parser.add_argument("images", nargs="*", type=Path, help="Formula image paths")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--runs", type=int, default=1, help="Measured runs per image")
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return value / divisor


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def configure_device(model: LatexOCR, device: torch.device) -> None:
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    model.args.device = str(device)
    model.model.to(device)
    if model.image_resizer is not None:
        model.image_resizer.to(device)


def checkpoint_size_mb() -> float:
    checkpoint_dir = Path(pix2tex.__file__).resolve().parent / "model" / "checkpoints"
    files = (checkpoint_dir / "weights.pth", checkpoint_dir / "image_resizer.pth")
    return sum(path.stat().st_size for path in files) / (1024 * 1024)


def mps_memory_mb(device: torch.device) -> float | None:
    if device.type != "mps":
        return None
    return torch.mps.current_allocated_memory() / (1024 * 1024)


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")

    device = torch.device(args.device)
    print(f"Device: {device}")
    print("Loading pix2tex model...")

    # LatexOCR copies every prediction to the system clipboard by default.
    # Disable that side effect during benchmarks.
    pix2tex_cli.clipboard.copy = lambda _: None

    load_started = time.perf_counter()
    model = LatexOCR()
    model.args.temperature = args.temperature
    configure_device(model, device)
    synchronize(device)
    load_seconds = time.perf_counter() - load_started
    print(f"Model loaded in {load_seconds:.3f}s")

    results: list[dict[str, Any]] = []
    for image_path in args.images:
        image_path = image_path.resolve()
        predictions: list[str] = []
        inference_seconds: list[float] = []
        with Image.open(image_path) as image:
            source_size = list(image.size)
            rgb_image = image.convert("RGB")
            for _ in range(args.runs):
                started = time.perf_counter()
                prediction = model(rgb_image.copy())
                synchronize(device)
                inference_seconds.append(time.perf_counter() - started)
                predictions.append(prediction)

        result = {
            "image": str(image_path),
            "source_size": source_size,
            "inference_seconds": inference_seconds,
            "latex": predictions[-1],
            "all_predictions": predictions,
            "outputs_consistent": len(set(predictions)) == 1,
        }
        results.append(result)

        print(f"\nImage: {image_path}")
        print(f"Source size: {source_size[0]}x{source_size[1]}")
        print("Inference: " + ", ".join(f"{value:.3f}s" for value in inference_seconds))
        print(f"Outputs consistent: {result['outputs_consistent']}")
        print("LaTeX:")
        print(result["latex"])

    report = {
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "temperature": args.temperature,
        "checkpoint_size_mb": checkpoint_size_mb(),
        "model_load_seconds": load_seconds,
        "peak_rss_mb": peak_rss_mb(),
        "mps_allocated_mb": mps_memory_mb(device),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(f"\nCheckpoint files: {report['checkpoint_size_mb']:.1f} MB")
    print(f"Peak RSS: {report['peak_rss_mb']:.1f} MB")
    if report["mps_allocated_mb"] is not None:
        print(f"MPS allocated: {report['mps_allocated_mb']:.1f} MB")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
