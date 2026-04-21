#!/usr/bin/env python3
"""Run a tiny OpenEuroLLM language smoke test through local Ollama.

This is intentionally not a serious benchmark runner. It exists to verify that
all 38 OpenEuroLLM language codes can produce examples, call a local model, and
flow through OneRuler-style scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "OneRuler"))

from oellm_support import OELLM_LANGUAGES, niah_prompt_dict, synthetic_book_sentences
from eval.evaluate import evaluate_jsonl


DEFAULT_MODEL = "qwen3.6:35b-a3b"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "smoke_results"
TASK = "niah_single"
RESPONSE_KEY = "response-ollama-smoke"


def load_prompt(lang: str) -> dict[str, str]:
    prompt_path = REPO_ROOT / "OneRuler" / "data" / "prompt" / lang / "niah.txt"
    if prompt_path.exists():
        return json.loads(prompt_path.read_text(encoding="utf-8"))
    return niah_prompt_dict(lang)


def build_example(lang: str, index: int, max_context_sentences: int) -> dict[str, object]:
    template_dict = load_prompt(lang)
    context_dict = load_prompt(lang)

    key = f"codex{lang}{index:02d}"
    value = str(7300000 + index)
    needle = context_dict["needle_numbers"].format(key=key, value=value).strip()

    sentences = synthetic_book_sentences(lang, max_context_sentences)
    insert_at = max(1, len(sentences) // 2)
    context_parts = sentences[:insert_at] + [needle] + sentences[insert_at:]
    context = " ".join(context_parts)

    question = (
        template_dict["question_single_numbers"].format(query1=key)
        + " "
        + template_dict["please_list"]
        + " "
        + template_dict["if_no_numbers"]
    )
    answer_template = (
        template_dict["answer_prefix"] + " " + template_dict["answer_numbers"]
    )
    prompt = template_dict["task"].format(context=context) + question + answer_template
    prompt += "\n\nReturn only the answer in the requested <Answer> format."

    return {
        "index": index,
        "input": prompt,
        "outputs": [value],
        "length": len(prompt.split()),
    }


def call_ollama(
    model: str,
    prompt: str,
    url: str,
    num_predict: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body.get("response", "")


def clean_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-")


def fake_response(example: dict[str, object]) -> str:
    return f"<Answer>{example['outputs'][0]}</Answer>"


def run(args: argparse.Namespace) -> int:
    model_label = "oracle" if args.backend == "oracle" else args.model
    output_root = args.output_dir / clean_model_name(model_label)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    started = time.time()

    languages = args.languages or list(OELLM_LANGUAGES)

    for index, lang in enumerate(languages):
        if lang not in OELLM_LANGUAGES:
            raise ValueError(f"Unknown OpenEuroLLM language code: {lang}")
        language_name = OELLM_LANGUAGES[lang]
        print(f"[{index + 1:02d}/{len(languages)}] {lang} ({language_name})")

        example = build_example(
            lang=lang,
            index=index,
            max_context_sentences=args.context_sentences,
        )
        if args.backend == "oracle":
            response_text = fake_response(example)
            status = "ok"
            error = ""
        else:
            try:
                response_text = call_ollama(
                    model=args.model,
                    prompt=str(example["input"]),
                    url=args.ollama_url,
                    num_predict=args.num_predict,
                    timeout=args.timeout,
                )
                status = "ok"
                error = ""
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                response_text = ""
                status = "error"
                error = f"{type(exc).__name__}: {exc}"

        example[RESPONSE_KEY] = response_text
        if error:
            example["error"] = error

        result_path = output_root / lang / str(args.max_seq_length) / TASK / "validation.jsonl"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(example, ensure_ascii=False) + "\n", encoding="utf-8")

        eval_result = evaluate_jsonl(
            file_path=str(result_path),
            task=TASK,
            lang=lang,
            model_name="ollama-smoke",
        )

        summary_rows.append(
            {
                "language": lang,
                "language_name": language_name,
                "status": status,
                "avg_acc": eval_result["avg_acc"],
                "processed": eval_result["processed"],
                "expected": "|".join(example["outputs"]),
                "response_preview": response_text.replace("\n", " ")[:160],
                "path": str(result_path),
                "error": error,
            }
        )

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    elapsed = time.time() - started
    ok_count = sum(row["status"] == "ok" for row in summary_rows)
    acc_count = sum(float(row["avg_acc"]) > 0 for row in summary_rows)
    print(f"Backend: {args.backend}")
    if args.backend == "oracle":
        print("Oracle backend uses expected answer as model output (no real LLM call).")
    print(f"\nWrote {summary_path}")
    print(f"Completed {ok_count}/{len(summary_rows)} generations in {elapsed:.1f}s")
    print(f"Non-zero smoke scores: {acc_count}/{len(summary_rows)}")
    return 0 if ok_count == len(summary_rows) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--context-sentences", type=int, default=10)
    parser.add_argument("--num-predict", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--languages", nargs="+", help="Optional subset of language codes")
    parser.add_argument("--backend", choices=["ollama", "oracle"], default="ollama")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
