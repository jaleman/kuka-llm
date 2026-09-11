#!/usr/bin/env bash
# Warn when free disk space approaches the 20 GB floor needed for model downloads.
# Run before every step; exits 1 when under the warning threshold so CI/scripts can stop.
set -euo pipefail

WARN_GB="${WARN_GB:-25}"     # warn when free space drops below this
HARD_GB="${HARD_GB:-20}"     # the floor we agreed on
CACHE="${HF_HOME:-$HOME/Projects/kuka-llm/.hf-cache}"

free_gb=$(df -g / | awk 'NR==2 {print $4}')
cache_size=$(du -sh "$CACHE" 2>/dev/null | cut -f1 || echo "0")

echo "free disk:  ${free_gb} GB"
echo "HF cache:   ${cache_size}  (${CACHE})"

if (( free_gb < HARD_GB )); then
  echo "STOP: under the ${HARD_GB} GB floor. Free space before downloading more models." >&2
  osascript -e "display notification \"Only ${free_gb} GB free, under the ${HARD_GB} GB floor\" with title \"kuka-llm disk check\"" 2>/dev/null || true
  exit 1
elif (( free_gb < WARN_GB )); then
  echo "WARNING: within 5 GB of the ${HARD_GB} GB floor." >&2
  osascript -e "display notification \"${free_gb} GB free, approaching the ${HARD_GB} GB floor\" with title \"kuka-llm disk check\"" 2>/dev/null || true
  exit 1
fi
echo "ok"
