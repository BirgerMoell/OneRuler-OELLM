"""Evaluate multilingual HELMET-style prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold().strip()


def compact(text: str) -> str:
    return re.sub(r"\s+", "", normalize(text))


def response_key(row: dict, explicit: str | None) -> str:
    if explicit:
        return explicit
    keys = [key for key in row if key.startswith("response-")]
    if len(keys) != 1:
        raise ValueError(f"Expected exactly one response-* key, found {keys}")
    return keys[0]


def score_exact(expected: list[str], response: str) -> float:
    text = normalize(response)
    compact_text = compact(response)
    return float(
        any(
            normalize(answer) == text
            or normalize(answer) in text
            or compact(answer) in compact_text
            for answer in expected
        )
    )


def score_cite(expected: list[str], response: str) -> float:
    return score_exact(expected, response)


def score_summ(expected: list[str], response: str) -> float:
    text = normalize(response)
    if not expected:
        return 0.0
    return sum(1 for point in expected if normalize(point) in text) / len(expected)


def score_rerank(expected: list[str], response: str) -> float:
    target = re.findall(r"[A-E]", expected[0].upper())
    bracketed = re.findall(r"\[([A-E])\]", response.upper())
    got = bracketed or re.findall(r"\b([A-E])\b", response.upper())
    if not target or not got:
        return 0.0
    got = got[: len(target)]
    return sum(1 for left, right in zip(target, got) if left == right) / len(target)


def score_row(row: dict, response: str) -> float:
    task = row.get("task", "")
    expected = row.get("outputs", [])
    if task == "summ":
        return score_summ(expected, response)
    if task == "rerank":
        return score_rerank(expected, response)
    if task == "cite":
        return score_cite(expected, response)
    return score_exact(expected, response)


def evaluate(path: Path, response_field: str | None) -> dict:
    scores = []
    by_task = {}
    by_lang = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = response_key(row, response_field)
            score = score_row(row, row.get(key) or "")
            scores.append(score)
            by_task.setdefault(row.get("task", "unknown"), []).append(score)
            by_lang.setdefault(row.get("language", "unknown"), []).append(score)
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "file": str(path),
        "num_samples": len(scores),
        "avg_score": avg,
        "by_task": {key: sum(vals) / len(vals) for key, vals in sorted(by_task.items())},
        "by_language": {key: sum(vals) / len(vals) for key, vals in sorted(by_lang.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mHELMET prediction JSONL.")
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--response_field")
    parser.add_argument("--output_path", type=Path)
    args = parser.parse_args()

    result = evaluate(args.input_path, args.response_field)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_path:
        args.output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
