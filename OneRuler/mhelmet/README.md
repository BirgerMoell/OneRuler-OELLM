# mHELMET

`mHELMET` is a deterministic multilingual HELMET-style benchmark generator for
OneRuler-OELLM. It covers the 38 OpenEuroLLM language codes from
`OneRuler/oellm_support.py`.

## Files

* `generate.py`: creates JSONL benchmark datasets.
* `evaluate.py`: scores JSONL prediction files containing one `response-*`
  field.
* `references.py`: native-language prompts and reference packs for all 38
  OpenEuroLLM languages.
* `templates.py`: legacy/shared task templates.

## Tasks

* `recall`: retrieve an identifier fact from a long context.
* `rag`: answer from retrieved passages with distractors.
* `rerank`: order five candidate passages by relevance score.
* `cite`: answer and cite the supporting passage label.
* `longqa`: answer from a long document.
* `summ`: preserve key facts in a short summary.
* `icl`: infer a mapping from in-context examples.

## Generate Data

Small smoke set:

```bash
python3 OneRuler/mhelmet/generate.py \
  --save_dir dataset/mhelmet-smoke \
  --languages en,sv,fi \
  --tasks recall,rag,rerank,cite,longqa,summ,icl \
  --lengths 1024 \
  --num_samples 2 \
  --tokenizer_type whitespace \
  --tokenizer_path whitespace
```

Default 38-language suite:

```bash
TOKENIZER_TYPE=openai TOKENIZER_PATH=cl100k_base scripts/run_mhelmet_generation.sh
```

Output layout:

```text
dataset/mhelmet/{language}/{length}/{task}/validation.jsonl
```

Each row uses OneRuler-compatible fields:

```json
{
  "index": 0,
  "language": "en",
  "language_name": "English",
  "task": "recall",
  "input": "...",
  "outputs": ["327097"],
  "length": 1024
}
```

## Evaluate Predictions

Prediction rows should contain exactly one `response-*` field, or pass
`--response_field` explicitly.

```bash
python3 OneRuler/mhelmet/evaluate.py \
  --input_path path/to/predictions.jsonl \
  --output_path path/to/predictions.score.json
```

## Local Ollama Smoke

Run a small local model experiment:

```bash
python3 scripts/run_mhelmet_ollama_smoke.py \
  --backend ollama \
  --model gemma4:e2b \
  --languages en,sv \
  --tasks recall,rag,cite,longqa \
  --length 512 \
  --num-samples 1 \
  --num-predict 64
```

The runner:

* generates the requested mHELMET slice,
* calls Ollama,
* writes predictions,
* scores each language/task file,
* writes `summary.csv`.

It sets `think: false` for Ollama HTTP calls, treats empty responses as errors,
and can fall back to `ollama run` if HTTP returns an empty response.

## Scoring Notes

The evaluator is intentionally simple and deterministic.

* `recall`, `rag`, `cite`, `longqa`, and `icl` use normalized exact/substring
  matching. Whitespace-insensitive matching handles compact citations such as
  `11605[S3]`.
* `rerank` parses only candidate labels `A-E`, including bracketed forms such
  as `[C], [A], [E], [B], [D]`.
* `summ` checks whether the generated summary contains the expected factual
  strings. This is strict and may under-score paraphrases.

## Current Caveats

The benchmark is model-generated and deterministic. It is useful for smoke
tests, regression tests, and early multilingual probing, but it should be
reviewed by native speakers and strengthened with real documents before being
used as a publishable benchmark.
