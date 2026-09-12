#!/usr/bin/env bash
# Step 2 toy run, end to end. Run from the repo root.
#   1. disk check      2. build toy dataset      3. generate BEFORE
#   4. train LoRA      5. generate AFTER          6. keep the logs
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
RUN=train/runs/toy
mkdir -p "$RUN"

PROMPT='The KMP 1500P is an autonomous mobile platform. Before starting the vehicle, the operator must'

bin/disk-check.sh
$PY data/make_toy_set.py ../kuka-mcp/knowledge | tee "$RUN/dataset.log"

echo "== BEFORE (base model) ==" | tee "$RUN/before.txt"
$PY -m mlx_lm generate --model mlx-community/Qwen3-0.6B-bf16 \
    --prompt "$PROMPT" --max-tokens 120 --temp 0 2>/dev/null | tee -a "$RUN/before.txt"

echo "== TRAIN =="
$PY -m mlx_lm lora -c train/toy/lora.yaml 2>&1 | tee "$RUN/train.log"

echo "== AFTER (base + adapter) ==" | tee "$RUN/after.txt"
$PY -m mlx_lm generate --model mlx-community/Qwen3-0.6B-bf16 --adapter-path "$RUN/adapters" \
    --prompt "$PROMPT" --max-tokens 120 --temp 0 2>/dev/null | tee -a "$RUN/after.txt"

echo "done; logs in $RUN/"
