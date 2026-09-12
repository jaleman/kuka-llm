#!/usr/bin/env python3
"""Step 2: build a small training set from the KUKA corpus for the toy run.

Samples a fixed number of chunks (deterministic seed), drops any that are too
long for the training window, and writes the three files mlx_lm expects:

    data/out/toy/train.jsonl   {"text": "..."} one chunk per line
    data/out/toy/valid.jsonl   held out, used to report validation loss
    data/out/toy/test.jsonl    held out, scored once at the end

Why "text" and not question/answer: this is continued pretraining in
miniature. The model is shown raw manual text and asked to predict the next
token; there is no instruction yet. Step 5/6 add the Q&A layer.

Usage:
  python data/make_toy_set.py [CORPUS_DIR] [--n 300] [--max-chars 10000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", nargs="?", default="../kuka-mcp/knowledge")
    ap.add_argument("--n", type=int, default=300, help="chunks to sample in total")
    ap.add_argument("--max-chars", type=int, default=8_500,
                    help="skip chunks longer than this (~1,600 tokens; keeps every chunk inside the 2k window)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/out/toy")
    args = ap.parse_args()

    files = sorted(Path(args.corpus).glob("*.md"))
    if not files:
        print(f"error: no *.md under {args.corpus}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(files)

    texts: list[str] = []
    skipped = 0
    for f in files:
        t = f.read_text(encoding="utf-8", errors="replace").strip()
        if len(t) > args.max_chars or len(t) < 200:
            skipped += 1
            continue
        texts.append(t)
        if len(texts) == args.n:
            break

    # 90 / 5 / 5 split
    n_valid = max(1, len(texts) // 20)
    n_test = max(1, len(texts) // 20)
    splits = {
        "test": texts[:n_test],
        "valid": texts[n_test:n_test + n_valid],
        "train": texts[n_test + n_valid:],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for t in rows:
                fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        print(f"{name:<5} {len(rows):>4} chunks  {sum(len(t) for t in rows):>9,} chars")
    print(f"skipped {skipped} chunks outside [200, {args.max_chars}] chars; wrote {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
