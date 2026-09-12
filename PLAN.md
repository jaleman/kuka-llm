# Plan: kuka-llm

Goal: an open-weights model that answers KUKA AMR questions from knowledge
baked into its weights, trained on the corpus produced by `../kuka-mcp`,
runnable on an NVIDIA DGX Spark.

## Status dashboard

> **This table is the single source of truth for progress.** Update it when
> starting or finishing a step and add a dated entry to the Progress Log.
> Statuses: `not started` | `in progress` | `complete` | `blocked`.
>
> **Do not start a step without the user's explicit approval.** Finishing one
> step is not authorization to start the next.
>
> **Every step gets a lesson** in `lessons/` in the same side-by-side style
> as `../kuka-mcp/lessons/` (the reader knows Java and some Rust; Python and
> ML concepts are explained by comparison).
>
> **Git workflow: one PR per step**, branch `step-N-<name>`. The user reviews
> and merges. Never commit datasets, adapters, or weights.
>
> **Disk rule (Mac mini phase).** Run `bin/disk-check.sh` before every step
> and before any model download. Warn the user when free space is within
> 5 GB of the 20 GB floor; stop at the floor. `HF_HOME` points at
> `.hf-cache/` in this repo so downloads are visible and gitignored.

| Step | Name | Status | Notes |
|------|------|--------|-------|
| 1 | Token count and corpus profile | complete | `data/count_tokens.py` |
| 2 | Foundations toy run (tiny model, watch loss fall) | complete | Mac mini, mlx_lm, Qwen3-0.6B LoRA: val loss 2.11→1.26, test ppl 4.3; lesson 3 |
| 3 | Build continued-pretraining dataset (raw text + 10–20% general mix) | complete | Done: 2.53M raw → 868k KUKA + 131k general tokens, unique-text 94%. `data/build_cpt_dataset.py`, `data/cpt-results.md`, lesson 4. |
| 4 | Continued pretraining run (LoRA, 8B base, 3–5 epochs) | not started | |
| 5 | Synthetic Q&A generation (5–10 pairs per chunk via teacher model) | not started | |
| 6 | Supervised fine-tuning on Q&A | not started | |
| 7 | Evaluation harness (held-out questions, LLM judge, 3-way comparison) | not started | base vs tuned vs base+MCP |
| 8 | Merge, quantize, serve (vLLM or llama.cpp) | not started | |
| 9 | Point kuka-mcp at the tuned model (hybrid) | not started | |

## Decisions

- **Base model**: Qwen3-8B (default) — strong on technical text, permissive
  license, tokenizer available without gating. Llama 3.1 8B is the fallback.
- **Method**: LoRA for continued pretraining and SFT; full fine-tune only if
  LoRA plateaus.
- **Language**: Python for data/train/eval. Rust (candle, mistral.rs) is an
  optional stretch for serving.

## Progress log

- 2026-09-11 — Repo scaffolded. Step 1 started: token-counting script.
- 2026-09-11 — Step 1 script runs against `../kuka-mcp/knowledge` with the
  Qwen3-8B tokenizer: 108 documents, 2,012 chunks, 1.21M words, **2.61M
  tokens** (2.17 tokens/word — tables and part numbers tokenize expensively).
  Chunk median 1,269 tokens, one chunk over 4,096. Found near-duplicate
  documents (fleet manual v2.14/2.15/2.16, several KMP 600P variants) —
  Step 3 must dedupe or down-weight these. Per-chunk CSV in `data/out/`.
- 2026-09-11 — Step 1 lesson written: `lessons/llm-01-token-counting.html`
  (Python-vs-Java side by side, findings table, dedupe note). Step 1 flips to
  `complete`: committed as the initial commit on master.
- 2026-09-11 — Redundancy investigation at user request (`data/find_duplicates.py`,
  lesson 2). 40% of corpus tokens are repeated text: three fleet-manual
  releases (547k tokens redundant), a byte-identical duplicate PDF
  (`BA_KMP_600P-U-D_series_en.pdf` == `BA_KMP_600P_series_en_V1.pdf`, 81k),
  and YAML front matter on every chunk (461k, 17.6%). Rules recorded in the
  step 3 row. Upstream fix for kuka-mcp: remove the duplicate PDF.
- 2026-09-11 — Step 2 started on branch `step-2-toy-run`: LoRA on
  Qwen3-0.6B-bf16 via mlx_lm, 300 iters, 270 train chunks. Lesson 0
  (concepts primer) added as required reading.
- 2026-09-11 — Step 2 complete on branch `step-2-toy-run`. Qwen3-0.6B-bf16
  LoRA (0.48% params), 300 iters, 20 min, peak 7.1 GB. Val loss 2.108 → 1.261,
  test perplexity 4.32. Style transferred, facts did not (model guessed "GPS"),
  train/val gap opened after iter 200. Results in `train/toy/results.md`,
  lesson 3 written. Lessons 0–3 now exist. Awaiting user review/merge;
  no GitHub remote exists yet for this repo.
- 2026-09-11 — Step 3 complete on branch `step-3-cpt-dataset` (stacked on
  step 2). `data/fetch_general.py` streams FineWeb-Edu (score ≥ 3) for the
  15% general mix; `data/build_cpt_dataset.py` drops superseded releases
  (fleet 2.11/2.14/2.15, QR-code v2.00), strips front matter, cleans pdftotext
  layout and both footer shapes, near-dedupes at paragraph level (8-word
  shingles, ≥ 80% seen → drop; 5,245 paragraphs), re-chunks to ≤ 2,048
  tokens, and writes provenance. Result: 868k KUKA + 131k general tokens,
  unique-text 60% → 94%. Lesson 4 written. Step 4 (real CPT run) needs the
  Spark; a Mac rehearsal on Qwen3-1.7B is possible if wanted.

