#!/usr/bin/env python3
"""Mini OELLM evaluation: 2 NIAH questions per language, saves results to repo.

Runs the Needle-In-A-Haystack task for all 38 OpenEuroLLM languages using a
local Ollama model and writes a summary CSV + markdown table to eval_results/.

Usage:
    python scripts/run_oellm_mini_eval.py                          # qwen2:0.5b, all langs
    python scripts/run_oellm_mini_eval.py --model qwen2:1.5b
    python scripts/run_oellm_mini_eval.py --backend oracle         # pipeline smoke-test
    python scripts/run_oellm_mini_eval.py --languages bg tr cy     # subset
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
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "OneRuler"))

from oellm_support import OELLM_LANGUAGES, niah_prompt_dict, sentence_ending
from eval.evaluate import evaluate_jsonl

DEFAULT_MODEL = "qwen2:0.5b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
QUESTIONS_PER_LANG = 2
CONTEXT_SENTENCES = 20
NUM_PREDICT = 128
TIMEOUT = 120
TASK = "niah_single"

_NOUN_TSV = REPO_ROOT / "OneRuler" / "data" / "vocab" / "100_noun_list_translated.tsv"
_noun_cache: dict[str, list[str]] = {}


def _load_nouns(lang: str) -> list[str]:
    if lang in _noun_cache:
        return _noun_cache[lang]
    if _NOUN_TSV.exists():
        with _NOUN_TSV.open(encoding="utf-8") as f:
            headers = f.readline().rstrip("\n").split("\t")
            if lang in headers:
                col = headers.index(lang)
                nouns = [line.split("\t")[col].strip() for line in f if line.strip()]
                nouns = [n for n in nouns if n]
                if nouns:
                    _noun_cache[lang] = nouns
                    return nouns
    # fallback: load from dictionary CSV
    lang_name = OELLM_LANGUAGES.get(lang, lang).capitalize()
    csv_path = REPO_ROOT / "OneRuler" / "data" / "vocab" / "dictionaries" / lang_name / f"{lang_name}.csv"
    if csv_path.exists():
        import csv as _csv
        with csv_path.open(encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            nouns = [row["words"].strip() for row in reader if row.get("pos") == "noun" and row.get("words", "").strip()]
        if nouns:
            _noun_cache[lang] = nouns
            return nouns
    raise RuntimeError(f"No real vocabulary found for language: {lang}")


def _real_sentences(lang: str, count: int) -> list[str]:
    words = _load_nouns(lang)
    ending = sentence_ending(lang)
    sentences = []
    for i in range(count):
        window = [words[(i + j) % len(words)] for j in range(8)]
        sentences.append(" ".join(window).capitalize() + ending)
    return sentences


def load_prompt(lang: str) -> dict[str, str]:
    prompt_path = REPO_ROOT / "OneRuler" / "data" / "prompt" / lang / "niah.txt"
    if prompt_path.exists():
        return json.loads(prompt_path.read_text(encoding="utf-8"))
    return niah_prompt_dict(lang)


def build_example(lang: str, lang_idx: int, q_idx: int) -> dict[str, object]:
    template = load_prompt(lang)
    unique_idx = lang_idx * 100 + q_idx
    key = f"oellm{lang}{q_idx:02d}"
    value = str(8000000 + unique_idx)

    sentences = _real_sentences(lang, CONTEXT_SENTENCES + 10)
    needle = template["needle_numbers"].format(key=key, value=value).strip()
    insert_at = max(1, (len(sentences) * (q_idx + 1)) // (QUESTIONS_PER_LANG + 1))
    parts = sentences[:insert_at] + [needle] + sentences[insert_at:CONTEXT_SENTENCES]
    context = " ".join(parts)

    question = (
        template["question_single_numbers"].format(query1=key)
        + " "
        + template["please_list"]
        + " "
        + template["if_no_numbers"]
    )
    answer_template = template["answer_prefix"] + " " + template["answer_numbers"]
    prompt = template["task"].format(context=context) + question + answer_template
    prompt += "\n\nReturn only the answer in the requested format."

    return {
        "index": q_idx + 1,
        "input": prompt,
        "outputs": [value],
        "length": len(prompt.split()),
    }


def call_ollama(model: str, prompt: str, url: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "top_p": 1, "num_predict": NUM_PREDICT},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


def clean_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-")


def run(args: argparse.Namespace) -> int:
    model_label = "oracle" if args.backend == "oracle" else args.model
    output_root = args.output_dir / clean_model_name(model_label)
    output_root.mkdir(parents=True, exist_ok=True)

    languages = args.languages or list(OELLM_LANGUAGES)
    summary_rows: list[dict] = []
    started = time.time()

    response_key = f"response-{clean_model_name(model_label)}"

    for lang_idx, lang in enumerate(languages):
        if lang not in OELLM_LANGUAGES:
            raise ValueError(f"Unknown OELLM language code: {lang}")
        lang_name = OELLM_LANGUAGES[lang]
        print(f"[{lang_idx + 1:02d}/{len(languages)}] {lang} ({lang_name})")

        jsonl_path = output_root / lang / TASK / "results.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("")

        lang_accs = []
        lang_errors = 0

        for q_idx in range(QUESTIONS_PER_LANG):
            example = build_example(lang, lang_idx, q_idx)

            if args.backend == "oracle":
                response_text = f"<Answer>{example['outputs'][0]}</Answer>"
                status = "ok"
                error = ""
            else:
                try:
                    response_text = call_ollama(args.model, str(example["input"]), args.ollama_url)
                    status = "ok"
                    error = ""
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    response_text = ""
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                    lang_errors += 1

            record = {**example, response_key: response_text}
            if error:
                record["error"] = error
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        eval_result = evaluate_jsonl(
            file_path=str(jsonl_path),
            task=TASK,
            lang=lang,
            model_name=clean_model_name(model_label),
        )

        avg_acc = eval_result["avg_acc"]
        lang_accs.append(avg_acc)
        status_final = "error" if lang_errors == QUESTIONS_PER_LANG else "ok"

        summary_rows.append(
            {
                "lang": lang,
                "language": lang_name,
                "status": status_final,
                "accuracy": f"{avg_acc:.2f}",
                "questions": QUESTIONS_PER_LANG,
                "errors": lang_errors,
            }
        )
        acc_str = f"{avg_acc:.0%}"
        print(f"        accuracy={acc_str}  errors={lang_errors}")

    elapsed = time.time() - started
    ok = sum(r["status"] == "ok" for r in summary_rows)
    avg_total = sum(float(r["accuracy"]) for r in summary_rows) / len(summary_rows)

    # Write CSV
    csv_path = output_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Write markdown table
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md_lines = [
        f"# OELLM Mini Eval — {model_label}",
        f"",
        f"*{QUESTIONS_PER_LANG} NIAH questions per language · {timestamp} · {elapsed:.0f}s total*",
        f"",
        f"| Lang | Language | Accuracy | Status |",
        f"|------|----------|----------|--------|",
    ]
    for r in summary_rows:
        acc_pct = f"{float(r['accuracy']):.0%}"
        md_lines.append(f"| {r['lang']} | {r['language']} | {acc_pct} | {r['status']} |")
    md_lines += [
        f"",
        f"**Overall average accuracy: {avg_total:.0%}** ({ok}/{len(summary_rows)} languages ok)",
    ]
    md_path = output_root / "results.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nModel: {model_label}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Completed {ok}/{len(summary_rows)} languages in {elapsed:.1f}s")
    print(f"Overall average accuracy: {avg_total:.0%}")
    return 0 if ok == len(summary_rows) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "eval_results" / "mini_eval")
    parser.add_argument("--languages", nargs="+", help="Subset of language codes")
    parser.add_argument("--backend", choices=["ollama", "oracle"], default="ollama")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
