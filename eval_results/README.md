# OELLM Evaluation Results

Evaluation results for the OneRuler benchmark extended with OpenEuroLLM language support.

## Task: Needle-in-a-Haystack (NIAH)

The NIAH task tests a model's ability to retrieve a specific piece of information
("needle") hidden inside a longer passage of text ("haystack").

### How a question is constructed

For each language, each question is built as follows:

1. **Haystack** — a passage of sentences built from real translated nouns for that
   language (sourced from `OneRuler/data/vocab/100_noun_list_translated.tsv`).
   Example haystack sentence for Welsh: *"Afal anifail balŵn llyfr pont gwely car cot."*

2. **Needle** — one sentence injected at a random position in the haystack:
   > *"The special magic number for "oellmbg00" is: 8000000"*
   (translated into the target language using the prompt in `OneRuler/data/prompt/{lang}/niah.txt`)

3. **Question** — after the haystack the model is asked:
   > *"What special magic numbers associated with "oellmbg00" are mentioned in the text?
   > Please list all that apply. If no such numbers exist, please answer "няма"."*
   (again fully translated into the target language)

4. **Expected answer** — the injected number (e.g. `8000000`). The model must
   output it inside `<Answer> ... </Answer>` tags (translated tags are accepted).

### Scoring

A response is marked **correct** if the exact injected number appears anywhere in the
model's output and no "none" word for that language is present. This is evaluated by
`OneRuler/eval/evaluate.py::compare_numbers()`.

### Prompt files

Each language has two prompt files under `OneRuler/data/prompt/{lang}/`:

| File | Format | Purpose |
|------|--------|---------|
| `niah.txt` | JSON (14 keys) | NIAH task — needle templates, question templates, answer format, "none" word |
| `cwe.txt` | Plain text | Common Words Extraction task — word list and question template |

All 38 OELLM languages have real translated prompt files (not English stubs).

---

## Results

| Directory | Model | Questions/lang | Token budget | Notes |
|-----------|-------|---------------|-------------|-------|
| `mini_eval/qwen2-0.5b/` | Qwen2 0.5B | 2 | 128 | baseline |
| `mini_eval/qwen2-1.5b/` | Qwen2 1.5B | 2 | 128 | baseline |
| `mini_eval/gemma4/` | Gemma 4 E4B | 2 | 512 | token budget too small |
| `full_eval/gemma4/` | Gemma 4 E4B | 5 | 1024 | recommended |

See `mini_eval/oellm_eval_results.png` for a visual comparison across all 38 languages.

### Reproducing a run

```bash
# Install Ollama: https://ollama.com
ollama pull gemma4

# Full eval (5 questions per language, ~45 min)
python scripts/run_oellm_mini_eval.py \
  --model gemma4 \
  --questions 5 \
  --num-predict 1024 \
  --output-dir eval_results/full_eval

# Quick smoke-test (2 questions, oracle backend — no model needed)
python scripts/run_oellm_mini_eval.py --backend oracle --questions 2
```

### Key findings

- **Gemma 4 E4B** achieves ~82% average NIAH accuracy across all 38 languages at
  `--num-predict 512`, rising toward 100% at `--num-predict 1024`.
- **All failures are empty responses** (token budget exhausted during reasoning),
  never wrong answers — indicating the model has coverage for all 38 languages.
- **Hungarian** is the most sensitive to token budget: both questions fail at 512
  tokens but pass at 1024.
- **Qwen2 0.5B / 1.5B** score ~20% — these tiny models struggle to follow the
  multilingual format but the pipeline runs correctly end-to-end.
