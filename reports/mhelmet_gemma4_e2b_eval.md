# mHELMET Gemma4 e2b Local Evaluation

Date: 2026-04-30

This report summarizes a local smoke-to-medium evaluation of `gemma4:e2b`
through Ollama on the new multilingual HELMET-style benchmark (`mHELMET`).

## Benchmark

`mHELMET` is a deterministic multilingual long-context benchmark inspired by
HELMET. It covers the 38 OpenEuroLLM language codes defined in
`OneRuler/oellm_support.py` and includes native-language reference packs and
prompt translations for every language.

Tasks:

* `recall`: retrieve an identifier fact from a long context.
* `rag`: answer from retrieved passages with distractors.
* `rerank`: order candidate passages by relevance score.
* `cite`: answer and cite the supporting passage label.
* `longqa`: answer from a long document.
* `summ`: preserve key facts in a short summary.
* `icl`: infer a mapping from in-context examples.

## Runner

The local runner is:

```bash
python3 scripts/run_mhelmet_ollama_smoke.py
```

It generates a small mHELMET slice, calls a local Ollama model, writes
prediction JSONL files, and scores each file with `OneRuler/mhelmet/evaluate.py`.

Important Ollama details:

* Default model: `gemma4:e2b`
* HTTP requests set `think: false`.
* Empty model responses are treated as errors.
* The runner can fall back to `ollama run` if the HTTP API returns an empty
  response.
* Prediction rows include `ollama_transport` so the caller can tell whether
  the response came from HTTP or CLI fallback.

## Initial Bugs Found And Fixed

### Empty Gemma Responses

The first Gemma-family run used `gemma4:31b` and produced empty responses. The
small model the user intended was installed as `gemma4:e2b`, not `gemma 2b`.

Fix:

* Changed the runner default model to `gemma4:e2b`.
* Added `think: false` to Ollama HTTP requests.
* Added empty-response detection and CLI fallback.

After this, `gemma4:e2b` returned non-empty HTTP responses.

### Missing Evidence At Short Context Lengths

At short context lengths, generated samples could truncate away the evidence
needed to answer the gold question while keeping the gold output.

Fix:

* Added evidence-preserving context fitting in `OneRuler/mhelmet/generate.py`.
* Required support facts/passages/examples are guaranteed to remain in the final
  prompt.

### RAG/Citation Target Collisions

Some distractors reused the target key with a conflicting answer.

Fix:

* Added target-aware distractor term selection so distractors do not reuse the
  target key.

### Rerank Test Design

The first rerank version was ambiguous:

* It asked "Which passage..." singular while expecting a full ranking.
* Distractors had `[X...]` labels, so models tried to rank distractors.
* Candidate scores were close together.
* The scorer parsed arbitrary capital letters instead of only candidate labels.

Fix:

* Only the true candidates use labels `[A]` through `[E]`.
* Background notes are unlabeled and explicitly ignored.
* The prompt asks to rank exactly five candidates.
* It states that larger numeric relevance score means more relevant.
* Score gaps were widened: `C=100`, `A=80`, `E=60`, `B=20`, `D=0`.
* The scorer now parses only `A-E`, including bracketed forms.

Validation after the fix:

```text
gemma4:e2b
language: en
task: rerank
length: 1024
samples: 4
score: 1.0
responses: [C], [A], [E], [B], [D]
```

## Medium Evaluation

Command:

```bash
python3 scripts/run_mhelmet_ollama_smoke.py \
  --backend ollama \
  --model gemma4:e2b \
  --languages en,sv,de,fr,es,fi \
  --tasks recall,rag,rerank,cite,longqa,summ,icl \
  --length 1024 \
  --num-samples 2 \
  --num-predict 128 \
  --timeout 240 \
  --output-dir /private/tmp/mhelmet-gemma-medium-eval
```

Run summary:

```text
Model: gemma4:e2b
Languages: en, sv, de, fr, es, fi
Tasks: recall, rag, rerank, cite, longqa, summ, icl
Length: 1024
Samples: 2 per language/task
Prediction files: 42
Examples: 84
Runtime: 291.0 seconds
Completed: 42/42 files
Non-zero score files: 29/42
Original file-average score: 0.613
Current scorer-normalized file-average score: 0.649
```

Artifacts:

```text
/private/tmp/mhelmet-gemma-medium-eval/gemma4-e2b/summary.csv
/private/tmp/mhelmet-gemma-medium-eval/gemma4-e2b/predictions/
```

## Medium Evaluation Scores

### By Task

| Task | Score |
| --- | ---: |
| `recall` | 1.000 |
| `rag` | 1.000 |
| `longqa` | 1.000 |
| `cite` | 1.000 |
| `icl` | 0.417 |
| `summ` | 0.111 |
| `rerank` | 0.017 |

The rerank score above is from the pre-fix medium run and should not be used as
the current rerank baseline. After the rerank fixes, an English 4-sample audit
scored 1.0.

The original medium run scored `cite` at 0.750 because compact citation strings
such as `11605[S3]` were treated as different from `11605 [S3]`. The evaluator
now removes whitespace for exact matching, which raises `cite` to 1.000 on the
same predictions.

### By Language

| Language | Score |
| --- | ---: |
| `es` | 0.729 |
| `en` | 0.714 |
| `fr` | 0.714 |
| `sv` | 0.595 |
| `fi` | 0.571 |
| `de` | 0.571 |

## Interpretation

The local `gemma4:e2b` model performs well on direct retrieval tasks at this
small scale: `recall`, `rag`, and `longqa` were perfect in the medium run.

Lower scores came from:

* `summ`: current scoring is exact-fact-string based and penalizes paraphrase,
  formatting changes, and partial factual summaries.
* `icl`: the small model sometimes infers nearby but wrong labels.
* `rerank`: the medium run used the earlier ambiguous test design. The fixed
  rerank task has since been validated separately.

## Recommended Next Steps

* Rerun the medium evaluation after the rerank fix to refresh the headline
  numbers.
* Add a softer summary scorer, for example fact-level keyword matching or an
  optional judge-based scorer, while keeping the strict scorer for regression
  testing.
* Increase samples per language/task once smoke behavior is stable.
* Run a broader all-38-language pass with a small sample count.
