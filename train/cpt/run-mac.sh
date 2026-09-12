#!/usr/bin/env bash
# Step 4 rehearsal on Apple Silicon. Run from the repo root; takes ~2-3 h.
#   1. disk check   2. BEFORE generations   3. train   4. AFTER generations
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
CFG=${CFG:-train/cpt/lora-mac.yaml}
RUN=train/runs/cpt-mac
MODEL=$(grep '^model:' $CFG | awk '{print $2}')
mkdir -p "$RUN"

bin/disk-check.sh
test -f data/out/cpt/train.jsonl || { echo "run data/build_cpt_dataset.py first" >&2; exit 1; }

gen() {  # $1 = output file, $2.. = extra args
  local out=$1; shift
  : > "$out"
  while IFS= read -r prompt; do
    [ -z "$prompt" ] && continue
    { echo "### $prompt"; $PY -m mlx_lm generate --model "$MODEL" "$@" \
        --ignore-chat-template --prompt "$prompt" --max-tokens 100 --temp 0 2>/dev/null | sed -n '/^==========$/,/^==========$/p' | grep -v '^==========$'; echo; } >> "$out"
  done < train/cpt/prompts.txt
}

if [ -s "$RUN/before.txt" ]; then echo "== BEFORE (reusing $RUN/before.txt) =="; else echo "== BEFORE =="; gen "$RUN/before.txt"; fi
echo "== TRAIN =="; $PY -m mlx_lm lora -c $CFG 2>&1 | tee "$RUN/train.log"
echo "== AFTER =="; gen "$RUN/after.txt" --adapter-path "$RUN/adapters"
echo "done; logs in $RUN/"
