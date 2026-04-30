#!/usr/bin/env python3
"""Run a tiny mHELMET experiment through local Ollama.

This is a smoke runner, not a throughput-optimized evaluation harness. It
generates a small mHELMET slice, calls a local Ollama model, writes prediction
JSONL files beside the generated data, and scores each file with the mHELMET
evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MHELMET_DIR = REPO_ROOT / "OneRuler" / "mhelmet"
sys.path.append(str(MHELMET_DIR))

from evaluate import evaluate


DEFAULT_MODEL = "gemma4:e2b"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "smoke_results" / "mhelmet"
RESPONSE_KEY = "response-ollama-smoke"


def clean_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-")


def csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def generate_dataset(args: argparse.Namespace, dataset_dir: Path) -> None:
    command = [
        sys.executable,
        str(MHELMET_DIR / "generate.py"),
        "--save_dir",
        str(dataset_dir),
        "--languages",
        ",".join(args.languages),
        "--tasks",
        ",".join(args.tasks),
        "--lengths",
        str(args.length),
        "--num_samples",
        str(args.num_samples),
        "--tokenizer_path",
        args.tokenizer_path,
        "--tokenizer_type",
        args.tokenizer_type,
        "--random_seed",
        str(args.random_seed),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def call_ollama_http(model: str, prompt: str, url: str, num_predict: int, timeout: int) -> tuple[str, dict]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
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
    return body.get("response", ""), body


def call_ollama_cli(model: str, prompt: str, timeout: int) -> str:
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return strip_ansi(result.stdout).strip()


def strip_ansi(text: str) -> str:
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    return ansi.sub("", text)


def call_ollama(args: argparse.Namespace, prompt: str) -> tuple[str, dict]:
    if args.ollama_transport == "cli":
        return call_ollama_cli(args.model, prompt, args.timeout), {"transport": "cli"}

    response, body = call_ollama_http(
        model=args.model,
        prompt=prompt,
        url=args.ollama_url,
        num_predict=args.num_predict,
        timeout=args.timeout,
    )
    if response.strip() or not args.fallback_cli_on_empty:
        body["transport"] = "http"
        return response, body

    cli_response = call_ollama_cli(args.model, prompt, args.timeout)
    body["transport"] = "http+cli-fallback"
    body["empty_http_response"] = True
    return cli_response, body


def fake_response(row: dict[str, object]) -> str:
    outputs = row.get("outputs", [])
    if not outputs:
        return ""
    if row.get("task") == "summ":
        return "\n".join(f"- {output}" for output in outputs)
    return str(outputs[0])


def predict_file(args: argparse.Namespace, input_path: Path, output_path: Path) -> tuple[str, str]:
    status = "ok"
    error = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            if args.backend == "oracle":
                response = fake_response(row)
            else:
                try:
                    response, metadata = call_ollama(args, str(row["input"]))
                    row["ollama_transport"] = metadata.get("transport")
                    if not response.strip():
                        status = "error"
                        error = "empty model response"
                except (
                    urllib.error.URLError,
                    TimeoutError,
                    json.JSONDecodeError,
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    response = ""
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
            row[RESPONSE_KEY] = response
            if error:
                row["error"] = error
            json.dump(row, dst, ensure_ascii=False)
            dst.write("\n")
    return status, error


def run(args: argparse.Namespace) -> int:
    model_label = "oracle" if args.backend == "oracle" else clean_model_name(args.model)
    run_dir = args.output_dir / model_label
    dataset_dir = run_dir / "dataset"
    prediction_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    generate_dataset(args, dataset_dir)

    rows = []
    for lang in args.languages:
        for task in args.tasks:
            input_path = dataset_dir / lang / str(args.length) / task / "validation.jsonl"
            output_path = prediction_dir / lang / str(args.length) / task / "validation.jsonl"
            print(f"[mHELMET] {lang}/{task}")
            status, error = predict_file(args, input_path, output_path)
            result = evaluate(output_path, RESPONSE_KEY)
            rows.append(
                {
                    "language": lang,
                    "task": task,
                    "status": status,
                    "avg_score": result["avg_score"],
                    "num_samples": result["num_samples"],
                    "prediction_path": str(output_path),
                    "error": error,
                }
            )

    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - started
    ok = sum(row["status"] == "ok" for row in rows)
    scored = sum(float(row["avg_score"]) > 0 for row in rows)
    print(f"\nBackend: {args.backend}")
    print(f"Model: {args.model if args.backend == 'ollama' else 'oracle'}")
    print(f"Wrote {summary_path}")
    print(f"Completed {ok}/{len(rows)} files in {elapsed:.1f}s")
    print(f"Non-zero scores: {scored}/{len(rows)}")
    return 0 if ok == len(rows) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["ollama", "oracle"], default="ollama")
    parser.add_argument("--ollama-transport", choices=["http", "cli"], default="http")
    parser.add_argument("--fallback-cli-on-empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--languages", type=csv_arg, default="en,sv")
    parser.add_argument("--tasks", type=csv_arg, default="recall,rag,cite,longqa")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-predict", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--tokenizer-path", default="whitespace")
    parser.add_argument("--tokenizer-type", default="whitespace", choices=["whitespace", "openai", "hf"])
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
