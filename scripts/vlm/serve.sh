#!/bin/bash
# Convert trained VLM to LLaVA format and launch vLLM server.
# Usage: bash scripts/vlm/serve.sh [--port 8000] [--model-path checkpoints/vlm_stage2/final]

set -euo pipefail

PORT=8000
MODEL_PATH="checkpoints/vlm_stage2/final"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

HF_DIR="${MODEL_PATH}/hf_llava"

echo "=== Step 1: Convert to HuggingFace LLaVA format ==="
python eval/convert_to_llava.py --model-path "$MODEL_PATH" --output-path "$HF_DIR"

echo ""
echo "=== Step 2: Launch vLLM server on port $PORT ==="
echo "Model: $HF_DIR"
echo ""

# pip install vllm  # uncomment if needed
vllm serve "$HF_DIR" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    "${EXTRA_ARGS[@]}"
