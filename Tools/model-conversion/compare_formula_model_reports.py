#!/usr/bin/env python3
"""Compare two formula-recognition reports with model-neutral LaTeX tokens."""

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    SCRIPT_DIR
    / "datasets"
    / "evaluation"
    / "layer1b-complex-formulas"
    / "manifest.jsonl"
)
DEFAULT_LEFT = SCRIPT_DIR / "test-output" / "layer1b-pix2tex-evaluation.json"
DEFAULT_RIGHT = SCRIPT_DIR / "test-output" / "layer1b-unimernet-evaluation.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "test-output" / "layer1b-model-comparison.json"

LATEX_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|[A-Za-z]+|\d+(?:\.\d+)?|[^\s]")
IGNORED_LAYOUT_TOKENS = {
    r"\left",
    r"\right",
    r"\,",
    r"\!",
    r"\;",
    r"\:",
    r"\quad",
    r"\qquad",
}
TOKEN_ALIASES = {r"\rm": r"\mathrm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--left", type=Path, default=DEFAULT_LEFT)
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--left-name", default="pix2tex")
    parser.add_argument("--right-name", default="unimernet_tiny")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def latex_tokens(value: str) -> list[str]:
    tokens = []
    for token in LATEX_TOKEN.findall(value):
        if token in IGNORED_LAYOUT_TOKENS:
            continue
        tokens.append(TOKEN_ALIASES.get(token, token))
    return tokens


def levenshtein(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def token_metrics(expected: str, predicted: str) -> dict[str, Any]:
    expected_tokens = latex_tokens(expected)
    predicted_tokens = latex_tokens(predicted)
    distance = levenshtein(predicted_tokens, expected_tokens)
    denominator = max(len(expected_tokens), len(predicted_tokens))
    similarity = 1.0 if denominator == 0 else 1.0 - distance / denominator
    return {
        "exact": predicted_tokens == expected_tokens,
        "edit_distance": distance,
        "edit_similarity": similarity,
        "expected_token_count": len(expected_tokens),
        "predicted_token_count": len(predicted_tokens),
    }


def load_report(path: Path) -> dict[str, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {record["id"]: record for record in report["samples"]}


def aggregate(records: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    successful = [record for record in records if record[model_name].get("error") is None]
    if not successful:
        return {"records": len(records), "successful": 0}
    similarities = np.array(
        [record[model_name]["token_edit_similarity"] for record in successful],
        dtype=np.float64,
    )
    latencies = np.array(
        [record[model_name]["inference_seconds"] for record in successful],
        dtype=np.float64,
    )
    return {
        "records": len(records),
        "successful": len(successful),
        "token_exact_matches": sum(
            record[model_name]["token_exact_match"] for record in successful
        ),
        "token_exact_match_rate": float(
            np.mean([record[model_name]["token_exact_match"] for record in successful])
        ),
        "mean_token_edit_similarity": float(similarities.mean()),
        "latency_seconds": {
            "mean": float(latencies.mean()),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(latencies.max()),
        },
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    left_path = args.left.expanduser().resolve()
    right_path = args.right.expanduser().resolve()
    for path in (manifest_path, left_path, right_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = {record["id"]: record for record in read_jsonl(manifest_path)}
    left = load_report(left_path)
    right = load_report(right_path)
    common_ids = [record_id for record_id in manifest if record_id in left and record_id in right]
    if not common_ids:
        raise RuntimeError("Reports have no samples in common")

    records = []
    for record_id in common_ids:
        left_record = left[record_id]
        right_record = right[record_id]
        left_expected = left_record.get("expected_latex")
        right_expected = right_record.get("expected_latex")
        if left_expected != right_expected:
            raise RuntimeError(f"Expected LaTeX differs for {record_id}")

        comparison = {
            "id": record_id,
            "complex_formula_category": manifest[record_id].get(
                "complex_formula_category"
            ),
            "expected_latex": left_expected,
        }
        for name, source in (
            (args.left_name, left_record),
            (args.right_name, right_record),
        ):
            model_result = {
                "predicted_latex": source.get("predicted_latex"),
                "inference_seconds": source.get("inference_seconds"),
                "error": source.get("error"),
            }
            if model_result["error"] is None and model_result["predicted_latex"] is not None:
                metrics = token_metrics(left_expected, model_result["predicted_latex"])
                model_result.update(
                    {
                        "token_exact_match": metrics["exact"],
                        "token_edit_distance": metrics["edit_distance"],
                        "token_edit_similarity": metrics["edit_similarity"],
                        "expected_token_count": metrics["expected_token_count"],
                        "predicted_token_count": metrics["predicted_token_count"],
                    }
                )
            comparison[name] = model_result
        records.append(comparison)

    categories = sorted(
        {
            record["complex_formula_category"]
            for record in records
            if record["complex_formula_category"] is not None
        }
    )
    paired = [
        record
        for record in records
        if record[args.left_name].get("error") is None
        and record[args.right_name].get("error") is None
    ]
    epsilon = 1e-12
    wins = {
        args.left_name: sum(
            record[args.left_name]["token_edit_similarity"]
            > record[args.right_name]["token_edit_similarity"] + epsilon
            for record in paired
        ),
        args.right_name: sum(
            record[args.right_name]["token_edit_similarity"]
            > record[args.left_name]["token_edit_similarity"] + epsilon
            for record in paired
        ),
    }
    wins["ties"] = len(paired) - sum(wins.values())

    report = {
        "report_schema_version": 1,
        "manifest": str(manifest_path),
        "reports": {
            args.left_name: str(left_path),
            args.right_name: str(right_path),
        },
        "common_samples": len(records),
        "metrics": {
            args.left_name: {
                "all": aggregate(records, args.left_name),
                "by_complex_formula_category": {
                    category: aggregate(
                        [
                            record
                            for record in records
                            if record["complex_formula_category"] == category
                        ],
                        args.left_name,
                    )
                    for category in categories
                },
            },
            args.right_name: {
                "all": aggregate(records, args.right_name),
                "by_complex_formula_category": {
                    category: aggregate(
                        [
                            record
                            for record in records
                            if record["complex_formula_category"] == category
                        ],
                        args.right_name,
                    )
                    for category in categories
                },
            },
        },
        "paired_similarity_wins": wins,
        "metric_notes": [
            "Both models are scored with the same model-neutral LaTeX lexer.",
            "Whitespace and purely visual spacing commands are ignored.",
            "Legacy \\rm is treated as \\mathrm.",
            "Token similarity is not a proof of rendered or mathematical equivalence.",
        ],
        "samples": records,
    }

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== formula model comparison ===")
    print(f"Common samples: {len(records)}")
    for name in (args.left_name, args.right_name):
        metrics = report["metrics"][name]["all"]
        print(
            f"{name}: token similarity "
            f"{metrics['mean_token_edit_similarity']:.3f}, "
            f"P50 {metrics['latency_seconds']['p50']:.3f}s"
        )
    print(f"Paired wins: {wins}")
    print(f"JSON: {output_path}")


if __name__ == "__main__":
    main()
