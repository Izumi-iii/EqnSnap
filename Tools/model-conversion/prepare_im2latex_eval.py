#!/usr/bin/env python3
"""Build two reproducible EqnSnap evaluation layers from im2latex-100k."""

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageFilter
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "datasets"
ZENODO_BASE = "https://zenodo.org/records/56198/files"
FILES = {
    "im2latex_test.lst": {
        "url": f"{ZENODO_BASE}/im2latex_test.lst?download=1",
        "md5": "1bc17b865796dca5df15250b4da7804f",
    },
    "im2latex_formulas.lst": {
        "url": f"{ZENODO_BASE}/im2latex_formulas.lst?download=1",
        "md5": "974c0a14f0daa6d91ecd0e625f1ddf52",
    },
    "formula_images.tar.gz": {
        "url": f"{ZENODO_BASE}/formula_images.tar.gz?download=1",
        "md5": "cf25f2408f1ea09bbd096890a6361533",
    },
}
EXCLUDED_ENVIRONMENTS = re.compile(
    r"\\begin\s*\{(?:array|matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|"
    r"aligned|align|alignat|cases|gather|multline|split|eqnarray)\*?\}"
)
LABEL_COMMAND = re.compile(r"\\label\s*\{[^{}]*\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download im2latex-100k and build a fixed official test subset "
            "plus deterministic screenshot-style variants."
        )
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-latex-chars", type=int, default=240)
    return parser.parse_args()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with temporary.open("wb") as output, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
                    progress.update(len(chunk))
    temporary.replace(destination)


def ensure_official_files(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, spec in FILES.items():
        path = raw_dir / name
        if not path.exists() or file_md5(path) != spec["md5"]:
            print(f"Downloading {name}...")
            download(spec["url"], path)
        actual_md5 = file_md5(path)
        if actual_md5 != spec["md5"]:
            raise RuntimeError(
                f"MD5 mismatch for {path}: expected {spec['md5']}, got {actual_md5}"
            )
        result[name] = path
        print(f"Verified {name}: {actual_md5}")
    return result


def read_formulas(path: Path) -> list[str]:
    # The official file is mostly ASCII but contains a few non-UTF-8 bytes.
    # Latin-1 preserves every byte and, critically, keeps formula line indices stable.
    with path.open("r", encoding="latin-1", newline="\n") as file:
        return file.read().split("\n")


def read_test_entries(path: Path) -> list[dict[str, Any]]:
    entries = []
    with path.open("r", encoding="utf-8", newline="\n") as file:
        for test_index, line in enumerate(file):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 3:
                raise RuntimeError(f"Unexpected test entry: {line!r}")
            formula_index, image_name, render_type = fields
            entries.append(
                {
                    "test_index": test_index,
                    "formula_index": int(formula_index),
                    "image_name": image_name,
                    "render_type": render_type,
                }
            )
    return entries


def normalized_latex(latex: str) -> str:
    without_labels = LABEL_COMMAND.sub("", latex)
    return re.sub(r"\s+", " ", without_labels.replace("\t", " ")).strip()


def scope_reasons(latex: str, max_latex_chars: int) -> list[str]:
    reasons = []
    if EXCLUDED_ENVIRONMENTS.search(latex):
        reasons.append("multi_line_environment")
    if r"\\" in latex:
        reasons.append("explicit_line_break")
    if r"\substack" in latex or r"\shortstack" in latex:
        reasons.append("stacked_expression")
    if len(latex) > max_latex_chars:
        reasons.append("latex_too_long")
    return reasons


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def image_geometry(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        bbox = rgba.getchannel("A").getbbox()
        if bbox is None:
            raise RuntimeError(f"No visible formula pixels in {path}")
        return {
            "original_size": list(rgba.size),
            "formula_bbox": list(bbox),
            "formula_size": [bbox[2] - bbox[0], bbox[3] - bbox[1]],
        }


def build_layer1(
    archive_path: Path,
    formulas: list[str],
    test_entries: list[dict[str, Any]],
    output_dir: Path,
    sample_size: int,
    seed: int,
    max_latex_chars: int,
) -> list[dict[str, Any]]:
    if sample_size > len(test_entries):
        raise ValueError(
            f"--sample-size {sample_size} exceeds test split size {len(test_entries)}"
        )

    selected = random.Random(seed).sample(test_entries, sample_size)
    selected.sort(key=lambda item: item["test_index"])
    shutil.rmtree(output_dir, ignore_errors=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)

    selected_filenames = {
        f"{entry['image_name']}.png": entry for entry in selected
    }
    remaining = set(selected_filenames)
    with tarfile.open(archive_path, "r:gz") as archive:
        progress = tqdm(total=len(selected), desc="Layer 1 extract", unit="image")
        for member in archive:
            filename = Path(member.name).name
            if filename not in remaining:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot extract archive member: {member.name}")
            destination = images_dir / filename
            with destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            remaining.remove(filename)
            progress.update(1)
            if not remaining:
                break
        progress.close()
    if remaining:
        raise RuntimeError(
            f"Images missing from archive: {', '.join(sorted(remaining)[:10])}"
        )

    records = []
    for entry in tqdm(selected, desc="Layer 1 manifest", unit="image"):
        filename = f"{entry['image_name']}.png"
        destination = images_dir / filename
        formula_index = entry["formula_index"]
        if formula_index >= len(formulas):
            raise RuntimeError(f"Formula index out of range: {formula_index}")
        raw_latex = formulas[formula_index]
        normalized = normalized_latex(raw_latex)
        reasons = scope_reasons(normalized, max_latex_chars)
        geometry = image_geometry(destination)
        if geometry["formula_size"][1] > 192:
            reasons.append("formula_too_tall")

        records.append(
            {
                "id": f"im2latex-test-{entry['test_index']:05d}",
                "layer": 1,
                "source_dataset": "im2latex-100k",
                "source_split": "test",
                "test_index": entry["test_index"],
                "formula_index": formula_index,
                "image_name": entry["image_name"],
                "render_type": entry["render_type"],
                "image": str(Path("images") / filename),
                "latex": raw_latex,
                "normalized_latex": normalized,
                "in_v0_1_scope": not reasons,
                "out_of_scope_reasons": reasons,
                **geometry,
            }
        )

    write_jsonl(output_dir / "manifest.jsonl", records)
    return records


def deterministic_random(seed: int, sample_id: str, variant: str) -> random.Random:
    value = hashlib.sha256(f"{seed}:{sample_id}:{variant}".encode()).digest()
    return random.Random(int.from_bytes(value[:8], "big"))


def formula_mask(path: Path) -> Image.Image:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise RuntimeError(f"No formula alpha mask in {path}")
    return alpha.crop(bbox)


def render_variant(
    source_path: Path,
    destination: Path,
    sample_id: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    rng = deterministic_random(seed, sample_id, variant)
    mask = formula_mask(source_path)

    if variant == "degraded":
        target_height = rng.randint(32, 48)
    else:
        target_height = rng.randint(48, 72)
    scale = target_height / mask.height
    target_width = max(1, round(mask.width * scale))
    if target_width > 1200:
        target_width = 1200
        target_height = max(1, round(mask.height * (target_width / mask.width)))
    mask = mask.resize((target_width, target_height), Image.Resampling.LANCZOS)

    padding_x = rng.randint(18, 52)
    padding_y = rng.randint(12, 30)
    if variant == "dark":
        background = tuple(rng.randint(20, 42) for _ in range(3))
        foreground = tuple(rng.randint(225, 250) for _ in range(3))
    else:
        background_value = rng.randint(244, 255)
        background = (background_value,) * 3
        foreground_value = rng.randint(5, 28)
        foreground = (foreground_value,) * 3

    canvas = Image.new(
        "RGB",
        (target_width + padding_x * 2, target_height + padding_y * 2),
        background,
    )
    ink = Image.new("RGB", mask.size, foreground)
    canvas.paste(ink, (padding_x, padding_y), mask)

    jpeg_quality = None
    blur_radius = None
    if variant == "degraded":
        blur_radius = round(rng.uniform(0.25, 0.75), 2)
        jpeg_quality = rng.randint(68, 86)
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur_radius))
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        canvas = Image.open(buffer).convert("RGB")

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=True)
    return {
        "variant": variant,
        "output_size": list(canvas.size),
        "formula_size": [target_width, target_height],
        "padding": [padding_x, padding_y],
        "background_rgb": list(background),
        "foreground_rgb": list(foreground),
        "blur_radius": blur_radius,
        "jpeg_quality": jpeg_quality,
    }


def build_layer2(
    layer1_dir: Path,
    layer1_records: list[dict[str, Any]],
    output_dir: Path,
    seed: int,
) -> list[dict[str, Any]]:
    shutil.rmtree(output_dir, ignore_errors=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)
    variants = ("light", "dark", "degraded")
    in_scope = [record for record in layer1_records if record["in_v0_1_scope"]]
    records = []

    for source_record in tqdm(in_scope, desc="Layer 2", unit="formula"):
        source_path = layer1_dir / source_record["image"]
        for variant in variants:
            filename = f"{source_record['id']}--{variant}.png"
            destination = images_dir / filename
            transform = render_variant(
                source_path,
                destination,
                source_record["id"],
                variant,
                seed,
            )
            records.append(
                {
                    "id": f"{source_record['id']}--{variant}",
                    "layer": 2,
                    "derived_from": source_record["id"],
                    "image": str(Path("images") / filename),
                    "latex": source_record["latex"],
                    "normalized_latex": source_record["normalized_latex"],
                    "in_v0_1_scope": True,
                    "transform": transform,
                }
            )

    write_jsonl(output_dir / "manifest.jsonl", records)
    return records


def write_summary(
    output_root: Path,
    layer1_records: list[dict[str, Any]],
    layer2_records: list[dict[str, Any]],
    test_split_size: int,
    seed: int,
    max_latex_chars: int,
) -> Path:
    in_scope_count = sum(record["in_v0_1_scope"] for record in layer1_records)
    reason_counts: dict[str, int] = {}
    for record in layer1_records:
        for reason in record["out_of_scope_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        "schema_version": 1,
        "source": {
            "name": "im2latex-100k",
            "url": "https://zenodo.org/records/56198",
            "license": "CC0-1.0",
            "test_split_size": test_split_size,
            "formula_file_encoding": "latin-1 (byte-preserving)",
            "files": FILES,
        },
        "selection": {
            "seed": seed,
            "layer1_sample_size": len(layer1_records),
            "max_latex_chars_for_v0_1_scope": max_latex_chars,
        },
        "layer1": {
            "description": "Fixed sample from the official im2latex test split",
            "records": len(layer1_records),
            "in_v0_1_scope": in_scope_count,
            "out_of_scope": len(layer1_records) - in_scope_count,
            "out_of_scope_reason_counts": reason_counts,
            "manifest": "layer1-im2latex-test/manifest.jsonl",
        },
        "layer2": {
            "description": "Deterministic screenshot-style variants of Layer 1 in-scope formulas",
            "records": len(layer2_records),
            "variants_per_formula": 3,
            "manifest": "layer2-screenshot-styles/manifest.jsonl",
        },
        "limitations": [
            "Layer 1 is synthetic and may overlap data seen while training pix2tex.",
            "Layer 2 simulates screenshot appearance but is not a substitute for real app screenshots.",
            "The v0.1 scope flag is a documented heuristic, not a mathematical guarantee.",
        ],
    }
    path = output_root / "summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    if args.sample_size < 1:
        raise ValueError("--sample-size must be at least 1")
    if args.max_latex_chars < 1:
        raise ValueError("--max-latex-chars must be at least 1")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    raw_files = ensure_official_files(dataset_dir / "im2latex-100k" / "raw")
    formulas = read_formulas(raw_files["im2latex_formulas.lst"])
    test_entries = read_test_entries(raw_files["im2latex_test.lst"])
    output_root = dataset_dir / "evaluation"
    layer1_dir = output_root / "layer1-im2latex-test"
    layer2_dir = output_root / "layer2-screenshot-styles"

    layer1_records = build_layer1(
        archive_path=raw_files["formula_images.tar.gz"],
        formulas=formulas,
        test_entries=test_entries,
        output_dir=layer1_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        max_latex_chars=args.max_latex_chars,
    )
    layer2_records = build_layer2(
        layer1_dir=layer1_dir,
        layer1_records=layer1_records,
        output_dir=layer2_dir,
        seed=args.seed,
    )
    summary_path = write_summary(
        output_root=output_root,
        layer1_records=layer1_records,
        layer2_records=layer2_records,
        test_split_size=len(test_entries),
        seed=args.seed,
        max_latex_chars=args.max_latex_chars,
    )

    in_scope_count = sum(record["in_v0_1_scope"] for record in layer1_records)
    print("\n=== EqnSnap evaluation datasets ready ===")
    print(f"Official test split: {len(test_entries)}")
    print(f"Layer 1: {len(layer1_records)} records")
    print(f"Layer 1 in v0.1 scope: {in_scope_count}")
    print(f"Layer 2: {len(layer2_records)} records")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
