# Step 3 — continued-pretraining dataset (2026-09-11)

Built by `data/build_cpt_dataset.py` from `../kuka-mcp/knowledge` (after the
duplicate 600P PDF was removed upstream), Qwen3-8B tokenizer, seed 0.

| Stage | Rows | Tokens | vs raw |
|---|---:|---:|---:|
| raw corpus | 1,951 | 2,531,814 | 100.0% |
| drop excluded docs (fleet 2.11/2.14/2.15, QR-code v2.00, reflector dup) | 1,305 | 1,762,604 | 69.6% |
| strip YAML front matter | 1,305 | 1,456,579 | 57.5% |
| layout cleaning (footers, indentation, hyphen breaks) | 1,305 | 1,159,931 | 45.8% |
| exact dedupe | 1,305 | 1,159,931 | 45.8% |
| paragraph near-dedupe (≥ 80% of shingles seen; 5,245 paragraphs) | 1,303 | 869,192 | 34.3% |
| re-chunk to ≤ 2,048 tokens | 1,258 | 867,938 | 34.3% |
| + general text (FineWeb-Edu, score ≥ 3, 15% target) | 191 | 130,836 | |

KUKA unique-text estimate after cleaning: **94%** (60% before).
101 documents kept. KUKA rows: median 661 tokens, max 2,079 (one
paragraph over the cap; the trainer truncates 31 tokens).

| Split | Rows | Tokens |
|---|---:|---:|
| train | 1,377 | 946,145 |
| valid | 36 | 29,858 |
| test | 36 | 22,771 |

Residue check on train: 0 rows with front matter, 0 footer lines, 3 in-body
mentions of www.kuka.com (legitimate references).

Outputs in `data/out/cpt/` (gitignored): `train/valid/test.jsonl`,
`provenance.jsonl` (row id → parent doc, pages, tokens), `stats.json`.
Regenerate with:

    python data/fetch_general.py --target-tokens 260000
    python data/build_cpt_dataset.py ../kuka-mcp/knowledge
