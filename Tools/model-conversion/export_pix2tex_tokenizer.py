#!/usr/bin/env python3
"""Export the decoder-only pix2tex vocabulary and Swift parity fixtures."""

import json
import os
from pathlib import Path
import random
import warnings

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\..*")

import torch
from pix2tex.cli import LatexOCR, token2str
from pix2tex.utils import post_process


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_AUTOREGRESSIVE_REPORT = (
    SCRIPT_DIR
    / "artifacts"
    / "fixtures"
    / "decoder-prefix128-context169"
    / "comparison.json"
)
DEFAULT_TOKENIZER_OUTPUT = (
    REPO_ROOT
    / "EqnSnap"
    / "Resources"
    / "Models"
    / "Pix2Tex"
    / "tokenizer.json"
)
DEFAULT_FIXTURE_OUTPUT = (
    REPO_ROOT
    / "EqnSnapTests"
    / "Fixtures"
    / "Tokenizer"
    / "tokenizer-fixtures.json"
)


def decoded_case(name: str, token_ids: list[int], ocr: LatexOCR) -> dict:
    raw = token2str(torch.tensor([token_ids]), ocr.tokenizer)[0]
    return {
        "name": name,
        "tokenIDs": token_ids,
        "tokens": ocr.tokenizer.convert_ids_to_tokens(token_ids),
        "decoded": raw,
        "postProcessed": post_process(raw),
    }


def main() -> None:
    ocr = LatexOCR()
    vocabulary = ocr.tokenizer.get_vocab()
    tokens_by_id: list[str | None] = [None] * len(ocr.tokenizer)
    for token, token_id in vocabulary.items():
        if token_id < len(tokens_by_id):
            tokens_by_id[token_id] = token
    if any(token is None for token in tokens_by_id):
        missing = [
            index for index, token in enumerate(tokens_by_id) if token is None
        ]
        raise RuntimeError(f"Tokenizer IDs are not contiguous: {missing}")

    tokenizer_output = DEFAULT_TOKENIZER_OUTPUT
    tokenizer_output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_output.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "modelID": "pix2tex",
                "vocabularySize": len(tokens_by_id),
                "decoderOutputSize": int(ocr.args.num_tokens),
                "specialTokens": {
                    "pad": int(ocr.args.pad_token),
                    "bos": int(ocr.args.bos_token),
                    "eos": int(ocr.args.eos_token),
                },
                "tokensByID": tokens_by_id,
                "decodeContract": {
                    "joinTokensWithoutSeparator": True,
                    "spaceMarker": "Ġ",
                    "removeTokens": ["[PAD]", "[BOS]", "[EOS]"],
                    "trimWhitespace": True,
                    "applyPix2TexPostProcess": True,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = [
        decoded_case("special-tokens", [1, 30, 2, 0], ocr),
        decoded_case("simple-fraction", [1, 104, 117, 103, 18, 102, 2], ocr),
        decoded_case(
            "mathrm-space-normalization",
            [1, 104, 159, 103, 165, 126, 134, 102, 2],
            ocr,
        ),
    ]
    report_path = DEFAULT_AUTOREGRESSIVE_REPORT
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for record in report["autoregressive_validation"]:
            cases.append(
                decoded_case(
                    f"autoregressive-h{record['target_foreground_height']}",
                    record["reference_token_ids"],
                    ocr,
                )
            )

    randomizer = random.Random(42)
    for index in range(20):
        length = randomizer.randint(1, 24)
        token_ids = [
            randomizer.randrange(len(tokens_by_id)) for _ in range(length)
        ]
        cases.append(decoded_case(f"random-{index:02d}", token_ids, ocr))

    fixture_output = DEFAULT_FIXTURE_OUTPUT
    fixture_output.parent.mkdir(parents=True, exist_ok=True)
    fixture_output.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "tokenizer": tokenizer_output.relative_to(REPO_ROOT).as_posix(),
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Exported {len(tokens_by_id)} tokens to {tokenizer_output}; "
        f"{len(cases)} fixtures to {fixture_output}"
    )


if __name__ == "__main__":
    main()
