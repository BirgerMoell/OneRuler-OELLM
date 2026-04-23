#!/bin/bash
# Run NIAH evaluation for all six OpenEuroLLM datamix-2b models.
# Each model is loaded from HuggingFace and run with a 500-word context
# (constrained by their 2048-token positional embedding limit).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODELS=(
    "openeurollm/datamix-2b-en"
    "openeurollm/datamix-2b-50-50"
    "openeurollm/datamix-2b-60-40"
    "openeurollm/datamix-2b-70-30"
    "openeurollm/datamix-2b-80-20"
    "openeurollm/datamix-2b-90-10"
)

for MODEL in "${MODELS[@]}"; do
    echo "========================================================"
    echo "Starting: $MODEL  $(date)"
    echo "========================================================"
    uv run --with transformers --with accelerate \
        python3 scripts/run_oellm_mini_eval.py \
            --backend huggingface \
            --model "$MODEL" \
            --context-words 500 \
            --questions 2 \
            --num-predict 64 \
            --output-dir eval_results/full_eval
    echo "Finished: $MODEL  $(date)"
done

echo ""
echo "All datamix models complete."
