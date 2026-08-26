#!/usr/bin/env python3
"""Export UniMERNet tokenizer and preprocessing fixtures for Swift tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from unimernet.models.unimernet.encoder_decoder import DonutTokenizer
from unimernet.processors import load_processor

from evaluate_unimernet import DEFAULT_MODEL_DIR


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent.parent
TOKENIZER_SOURCE = DEFAULT_MODEL_DIR / "tokenizer.json"
TOKENIZER_OUTPUT = (
    REPOSITORY_ROOT
    / "EqnSnap"
    / "Resources"
    / "Models"
    / "UniMERNet"
    / "UniMERNetTokenizer.json"
)
FIXTURE_DIRECTORY = (
    REPOSITORY_ROOT / "EqnSnapTests" / "Fixtures" / "UniMERNet"
)
SOURCE_IMAGE = SCRIPT_DIR / "test_images" / "formula.png"


def export_tokenizer() -> tuple[object, dict[str, object]]:
    source = json.loads(TOKENIZER_SOURCE.read_text(encoding="utf-8"))
    vocabulary = source["model"]["vocab"]
    tokens_by_id = [""] * len(vocabulary)
    for token, token_id in vocabulary.items():
        tokens_by_id[int(token_id)] = token
    if any(token == "" for token in tokens_by_id):
        raise RuntimeError("UniMERNet tokenizer vocabulary has missing IDs")

    special_token_ids = sorted(
        int(item["id"])
        for item in source["added_tokens"]
        if item.get("special") is True
    )
    resource = {
        "schemaVersion": 1,
        "modelID": "UniMERNet Tiny",
        "vocabularySize": len(tokens_by_id),
        "specialTokens": {"bos": 0, "pad": 1, "eos": 2, "unk": 3},
        "specialTokenIDs": special_token_ids,
        "tokensByID": tokens_by_id,
    }
    TOKENIZER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TOKENIZER_OUTPUT.write_text(
        json.dumps(resource, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tokenizer = DonutTokenizer(str(DEFAULT_MODEL_DIR)).tokenizer
    return tokenizer, resource


def export_tokenizer_fixtures(tokenizer: object) -> None:
    texts = [
        r"x = \frac { - b \pm \sqrt { b ^ { 2 } - 4 a c } } { 2 a }",
        r"\begin{array} { c c } a & b \\ c & d \end{array}",
        "alpha: α + β = γ; currency: ¥",
    ]
    cases = []
    for index, text in enumerate(texts, start=1):
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        cases.append(
            {
                "name": f"case-{index}",
                "tokenIDs": token_ids,
                "decoded": tokenizer.decode(token_ids, skip_special_tokens=True),
            }
        )
    (FIXTURE_DIRECTORY / "tokenizer-fixtures.json").write_text(
        json.dumps({"cases": cases}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def export_preprocessing_fixture() -> None:
    processor = load_processor(
        "formula_image_eval", OmegaConf.create({"image_size": [192, 672]})
    )
    with Image.open(SOURCE_IMAGE) as image:
        rgb = image.convert("RGB")
        tensor = processor(rgb).unsqueeze(0).repeat(1, 3, 1, 1)
        source_size = list(rgb.size)
    array = tensor.numpy().astype(np.float32)
    np.save(FIXTURE_DIRECTORY / "preprocessor-input.npy", array)
    (FIXTURE_DIRECTORY / "preprocessor-fixture.json").write_text(
        json.dumps(
            {
                "sourceImage": str(SOURCE_IMAGE.relative_to(REPOSITORY_ROOT)),
                "sourceSize": source_size,
                "tensorShape": list(array.shape),
                "minimum": float(array.min()),
                "maximum": float(array.max()),
                "mean": float(array.mean()),
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    tokenizer, resource = export_tokenizer()
    export_tokenizer_fixtures(tokenizer)
    export_preprocessing_fixture()
    print(f"Tokenizer: {TOKENIZER_OUTPUT} ({resource['vocabularySize']} tokens)")
    print(f"Fixtures: {FIXTURE_DIRECTORY}")


if __name__ == "__main__":
    main()
