#!/usr/bin/env bash
set -euo pipefail

TOKENIZER_PATH="${TOKENIZER_PATH:-cl100k_base}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-openai}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SAVE_DIR="${SAVE_DIR:-dataset/mhelmet}"
LANGUAGES="${LANGUAGES:-bg,hr,cs,da,nl,et,fi,fr,de,el,hu,ga,it,lv,lt,mt,pl,pt,ro,sk,sl,es,sv,en,sq,eu,bs,ca,gl,is,lb,mk,no,ru,sr,tr,uk,cy}"
TASKS="${TASKS:-recall,rag,rerank,cite,longqa,summ,icl}"
LENGTHS="${LENGTHS:-8192,16384,32768}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"

"$PYTHON_BIN" OneRuler/mhelmet/generate.py \
  --save_dir "$SAVE_DIR" \
  --languages "$LANGUAGES" \
  --tasks "$TASKS" \
  --lengths "$LENGTHS" \
  --num_samples "$NUM_SAMPLES" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --tokenizer_type "$TOKENIZER_TYPE"
