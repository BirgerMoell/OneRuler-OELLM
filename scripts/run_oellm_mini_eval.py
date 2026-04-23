#!/usr/bin/env python3
"""OELLM evaluation: NIAH across all 38 languages.

Harder than a naive setup in three ways:
  1. Real book text is used as haystack for the 26 languages that have one
     (data/books/{lang}/). The remaining 12 OELLM-only languages fall back to
     long synthetic sentences built from the real noun translations.
  2. Distracting 7-digit numbers are seeded throughout the haystack so the
     needle value is not the only number the model sees.
  3. Needle depth is varied uniformly across questions (10%, 30%, 50%, 70%, 90%)
     rather than always placed at the midpoint.

Backends:
  ollama      — local Ollama server (default)
  huggingface — load model directly via transformers (GPU required for large models)
  oracle      — perfect answers, used for smoke-testing the pipeline

Usage:
    python scripts/run_oellm_mini_eval.py                            # qwen2:0.5b via Ollama
    python scripts/run_oellm_mini_eval.py --model gemma4 --questions 5 --num-predict 1024
    python scripts/run_oellm_mini_eval.py --backend oracle --questions 5
    python scripts/run_oellm_mini_eval.py --backend huggingface --model openeurollm/datamix-2b-80-20 --context-words 500
    python scripts/run_oellm_mini_eval.py --languages bg tr cy --questions 3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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
DEFAULT_QUESTIONS = 2
DEFAULT_CONTEXT_WORDS = 2000   # target haystack size in words
NUM_PREDICT = 128
TIMEOUT = 180
TASK = "niah_single"

# Needle is placed at these fractional depths; cycled across questions
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]

_NOUN_TSV = REPO_ROOT / "OneRuler" / "data" / "vocab" / "100_noun_list_translated.tsv"
_BOOKS_DIR = REPO_ROOT / "OneRuler" / "data" / "books"
_noun_cache: dict[str, list[str]] = {}
_haystack_cache: dict[str, list[str]] = {}


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
    lang_name = OELLM_LANGUAGES.get(lang, lang).capitalize()
    csv_path = REPO_ROOT / "OneRuler" / "data" / "vocab" / "dictionaries" / lang_name / f"{lang_name}.csv"
    if csv_path.exists():
        import csv as _csv
        with csv_path.open(encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            nouns = [row["words"].strip() for row in reader
                     if row.get("pos") == "noun" and row.get("words", "").strip()]
        if nouns:
            _noun_cache[lang] = nouns
            return nouns
    raise RuntimeError(f"No vocabulary found for language: {lang}")


def _load_haystack(lang: str) -> list[str]:
    """Return a list of sentences for use as haystack. Prefers real book text."""
    if lang in _haystack_cache:
        return _haystack_cache[lang]

    book_dir = _BOOKS_DIR / lang
    if book_dir.exists():
        texts = []
        for p in sorted(book_dir.glob("*.txt")):
            texts.append(p.read_text(encoding="utf-8", errors="replace"))
        full_text = " ".join(texts)
        full_text = re.sub(r"\s+", " ", full_text)
        # Sentence-split on common terminators
        sentences = re.split(r"(?<=[.!?])\s+", full_text.strip())
        sentences = [s.strip() for s in sentences if len(s.split()) >= 3]
        if sentences:
            _haystack_cache[lang] = sentences
            return sentences

    # Fallback: long synthetic sentences from real nouns
    nouns = _load_nouns(lang)
    ending = sentence_ending(lang)
    sentences = []
    for i in range(3000):
        window = [nouns[(i + j) % len(nouns)] for j in range(10)]
        sentences.append(" ".join(window).capitalize() + ending)
    _haystack_cache[lang] = sentences
    return sentences


def _distractor_sentence(lang: str, rng: random.Random) -> str:
    """A haystack sentence with a distracting 7-digit number embedded."""
    nouns = _load_nouns(lang)
    ending = sentence_ending(lang)
    words = [nouns[rng.randint(0, len(nouns) - 1)] for _ in range(5)]
    number = str(rng.randint(1_000_000, 9_999_999))
    pos = rng.randint(1, len(words) - 1)
    words.insert(pos, number)
    return " ".join(words).capitalize() + ending


def _build_haystack_with_distractors(
    lang: str, target_words: int, n_distractors: int, rng: random.Random
) -> list[str]:
    """Return ~target_words worth of haystack sentences, with n_distractors mixed in."""
    base = _load_haystack(lang)
    # Sample enough sentences to hit the word target
    result: list[str] = []
    word_count = 0
    idx = rng.randint(0, max(0, len(base) - 1))
    while word_count < target_words:
        result.append(base[idx % len(base)])
        word_count += len(base[idx % len(base)].split())
        idx += 1

    # Scatter distracting number sentences throughout
    for _ in range(n_distractors):
        pos = rng.randint(0, len(result))
        result.insert(pos, _distractor_sentence(lang, rng))

    return result


def load_prompt(lang: str) -> dict[str, str]:
    prompt_path = REPO_ROOT / "OneRuler" / "data" / "prompt" / lang / "niah.txt"
    if prompt_path.exists():
        return json.loads(prompt_path.read_text(encoding="utf-8"))
    return niah_prompt_dict(lang)


def build_example(
    lang: str, lang_idx: int, q_idx: int, n_questions: int, context_words: int, rng: random.Random
) -> dict[str, object]:
    template = load_prompt(lang)
    unique_idx = lang_idx * 100 + q_idx
    key = f"oellm{lang}{q_idx:02d}"
    value = str(8_000_000 + unique_idx)

    # Vary needle depth uniformly across questions
    depth = DEPTHS[q_idx % len(DEPTHS)]

    # ~10 distracting numbers per 2000 words of context
    n_distractors = max(5, context_words // 200)
    sentences = _build_haystack_with_distractors(lang, context_words, n_distractors, rng)

    needle = template["needle_numbers"].format(key=key, value=value).strip()
    insert_at = max(1, int(len(sentences) * depth))
    parts = sentences[:insert_at] + [needle] + sentences[insert_at:]
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

    haystack_source = "book" if (_BOOKS_DIR / lang).exists() else "synthetic"

    return {
        "index": q_idx + 1,
        "input": prompt,
        "outputs": [value],
        "length": len(prompt.split()),
        "needle_depth": depth,
        "haystack_source": haystack_source,
        "context_words": len(context.split()),
    }


def call_ollama(model: str, prompt: str, url: str, num_predict: int = NUM_PREDICT) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "top_p": 1, "num_predict": num_predict},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


# Lazily loaded HuggingFace model/tokenizer (shared across all calls in a run)
_hf_model = None
_hf_tokenizer = None
_hf_model_name: str | None = None


def _load_hf_model(model_name: str):
    global _hf_model, _hf_tokenizer, _hf_model_name
    if _hf_model_name == model_name:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  [HF] Loading {model_name} ...", flush=True)
    _hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    _hf_model.eval()
    _hf_model_name = model_name
    print(f"  [HF] Loaded on {next(_hf_model.parameters()).device}", flush=True)


def call_huggingface(model_name: str, prompt: str, num_predict: int = NUM_PREDICT) -> str:
    import torch

    _load_hf_model(model_name)
    tok = _hf_tokenizer
    model = _hf_model

    max_model_len = getattr(model.config, "max_position_embeddings", 2048)
    max_new = min(num_predict, 128)
    max_input = max_model_len - max_new

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )

    new_tokens = out[0][input_len:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def clean_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-")


def run(args: argparse.Namespace) -> int:
    rng = random.Random(42)
    model_label = "oracle" if args.backend == "oracle" else args.model
    output_root = args.output_dir / clean_model_name(model_label)
    output_root.mkdir(parents=True, exist_ok=True)

    languages = args.languages or list(OELLM_LANGUAGES)
    summary_rows: list[dict] = []
    started = time.time()
    response_key = f"response-{clean_model_name(model_label)}"

    total_questions = len(languages) * args.questions
    questions_done = 0

    for lang_idx, lang in enumerate(languages):
        if lang not in OELLM_LANGUAGES:
            raise ValueError(f"Unknown OELLM language code: {lang}")
        lang_name = OELLM_LANGUAGES[lang]
        elapsed_so_far = time.time() - started
        pct = questions_done / total_questions * 100 if total_questions else 0
        eta_str = ""
        if questions_done > 0:
            rate = elapsed_so_far / questions_done
            eta_str = f"  ETA {(total_questions - questions_done) * rate / 60:.1f}m"
        haystack_tag = "book" if (_BOOKS_DIR / lang).exists() else "synthetic"
        print(f"[{lang_idx + 1:02d}/{len(languages)}] {lang} ({lang_name})  "
              f"haystack={haystack_tag}  {pct:.0f}% done{eta_str}", flush=True)

        jsonl_path = output_root / lang / TASK / "results.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("")

        lang_errors = 0

        for q_idx in range(args.questions):
            q_started = time.time()
            example = build_example(lang, lang_idx, q_idx, args.questions,
                                    args.context_words, rng)

            if args.backend == "oracle":
                response_text = f"<Answer>{example['outputs'][0]}</Answer>"
                error = ""
            elif args.backend == "huggingface":
                try:
                    response_text = call_huggingface(
                        args.model, str(example["input"]), args.num_predict
                    )
                    error = ""
                except Exception as exc:
                    response_text = ""
                    error = f"{type(exc).__name__}: {exc}"
                    lang_errors += 1
            else:
                try:
                    response_text = call_ollama(
                        args.model, str(example["input"]), args.ollama_url, args.num_predict
                    )
                    error = ""
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                    response_text = ""
                    error = f"{type(exc).__name__}: {exc}"
                    lang_errors += 1

            correct = bool(response_text) and example["outputs"][0] in response_text
            q_elapsed = time.time() - q_started
            questions_done += 1
            overall_pct = questions_done / total_questions * 100
            depth_pct = f"{example['needle_depth']:.0%}"
            ctx_words = example["context_words"]
            print(f"  q{q_idx + 1:02d}/{args.questions}  {'✓' if correct else '✗'}"
                  f"  depth={depth_pct}  ctx={ctx_words}w"
                  f"  {q_elapsed:.1f}s  [{overall_pct:.0f}% overall]", flush=True)

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
        status_final = "error" if lang_errors == args.questions else "ok"

        summary_rows.append({
            "lang": lang,
            "language": lang_name,
            "haystack": haystack_tag,
            "status": status_final,
            "accuracy": f"{avg_acc:.2f}",
            "questions": args.questions,
            "errors": lang_errors,
        })
        print(f"  → accuracy={avg_acc:.0%}  errors={lang_errors}", flush=True)

    elapsed = time.time() - started
    ok = sum(r["status"] == "ok" for r in summary_rows)
    avg_total = sum(float(r["accuracy"]) for r in summary_rows) / len(summary_rows)

    csv_path = output_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md_lines = [
        f"# OELLM Eval — {model_label}",
        f"",
        f"*{args.questions} NIAH questions per language · ~{args.context_words} word context"
        f" · depths {DEPTHS} · {timestamp} · {elapsed:.0f}s total*",
        f"",
        f"| Lang | Language | Haystack | Accuracy | Status |",
        f"|------|----------|----------|----------|--------|",
    ]
    for r in summary_rows:
        acc_pct = f"{float(r['accuracy']):.0%}"
        md_lines.append(
            f"| {r['lang']} | {r['language']} | {r['haystack']} | {acc_pct} | {r['status']} |"
        )
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
    parser.add_argument("--backend", choices=["ollama", "oracle", "huggingface"], default="ollama")
    parser.add_argument("--num-predict", type=int, default=NUM_PREDICT,
                        help="Max tokens to generate per response")
    parser.add_argument("--questions", type=int, default=DEFAULT_QUESTIONS,
                        help="Questions per language")
    parser.add_argument("--context-words", type=int, default=DEFAULT_CONTEXT_WORDS,
                        help="Target haystack size in words (~1.3x tokens)")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
