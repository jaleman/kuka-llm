#!/usr/bin/env python3
"""Step 3: fetch a slice of general English text to mix into the KUKA corpus.

Why: continued pretraining on domain text alone makes the model forget general
English and reasoning ("catastrophic forgetting", lesson 0). Mixing 10-20% of
ordinary text in keeps those abilities intact. FineWeb-Edu is a filtered web
crawl scored for educational quality (ODC-By licence), streamed so nothing is
downloaded beyond what we keep.

Writes data/out/general.jsonl, one {"text": ...} per line, until the target
token count is reached. Documents longer than --max-tokens are cut at a
paragraph boundary.

Usage:
  python data/fetch_general.py [--target-tokens 250000] [--min-score 3.0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-tokens", type=int, default=250_000)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--min-score", type=float, default=3.0, help="FineWeb-Edu educational score floor (0-5)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="data/out/general.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(args.tokenizer)
    count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = total = seen = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in ds:
            seen += 1
            if row.get("score", 0) < args.min_score:
                continue
            text = row["text"].strip()
            # cut at a paragraph boundary if too long
            if count(text) > args.max_tokens:
                paras, acc = text.split("\n\n"), []
                for p in paras:
                    if count("\n\n".join(acc + [p])) > args.max_tokens:
                        break
                    acc.append(p)
                text = "\n\n".join(acc)
                if not text:
                    continue
            n = count(text)
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            kept += 1
            total += n
            if kept % 100 == 0:
                print(f"  ... {kept} docs, {total:,} tokens", file=sys.stderr)
            if total >= args.target_tokens:
                break
    print(f"kept {kept} of {seen} streamed docs, {total:,} tokens -> {out}")
    sys.stdout.flush()
    # pyarrow's streaming generator crashes at interpreter shutdown on Python 3.14;
    # everything is written and flushed, so exit without running finalizers.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
