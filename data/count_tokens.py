#!/usr/bin/env python3
"""Step 1: profile the KUKA corpus in tokens.

Reads every *.md under a corpus directory (default: ../kuka-mcp/knowledge),
tokenizes each chunk with a real model tokenizer, and reports:

  * total files, words, tokens, and tokens-per-word ratio
  * per-chunk size distribution (min / median / p90 / max)
  * the largest source documents (chunks regrouped by document)
  * a CSV with one row per chunk for later steps to consume

Why tokens and not words: training cost, context limits, and epoch math are
all in tokens. The ratio varies by tokenizer and by content (tables and
part numbers tokenize badly), so we measure instead of guessing.

Usage:
  python data/count_tokens.py [CORPUS_DIR] [--tokenizer Qwen/Qwen3-8B]
                              [--top 15] [--out data/out/tokens.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Chunk filenames look like  <doc>-p001-007.md ; strip the page suffix to
# regroup chunks into their source document.
PAGE_SUFFIX = re.compile(r"-p\d{3}-\d{3}$")


def load_tokenizer(name: str):
    """Return (count_fn, description). Falls back gracefully when offline."""
    try:
        from tokenizers import Tokenizer  # type: ignore

        tok = Tokenizer.from_pretrained(name)
        return (lambda s: len(tok.encode(s, add_special_tokens=False).ids), name)
    except Exception as exc:  # network down, gated repo, missing package
        print(f"warning: could not load tokenizer '{name}' ({exc.__class__.__name__}: {exc})",
              file=sys.stderr)
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda s: len(enc.encode(s, disallowed_special=())), "tiktoken cl100k_base (fallback)")
    except Exception:
        pass
    print("warning: falling back to chars/4 heuristic; install 'tokenizers' and go online for real numbers",
          file=sys.stderr)
    return (lambda s: len(s) // 4, "chars/4 heuristic (fallback)")


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    k = max(0, min(len(values) - 1, round(p * (len(values) - 1))))
    return values[k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", nargs="?", default="../kuka-mcp/knowledge", help="directory of *.md chunks")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="Hugging Face tokenizer repo id")
    ap.add_argument("--top", type=int, default=15, help="how many largest documents to list")
    ap.add_argument("--out", default="data/out/tokens.csv", help="per-chunk CSV output path")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    files = sorted(corpus.glob("*.md"))
    if not files:
        print(f"error: no *.md files under {corpus}", file=sys.stderr)
        return 1

    count, tok_desc = load_tokenizer(args.tokenizer)
    print(f"tokenizer: {tok_desc}")
    print(f"corpus:    {corpus.resolve()}  ({len(files)} chunks)")

    rows: list[tuple[str, str, int, int, int]] = []  # file, doc, chars, words, tokens
    per_doc: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(files, 1):
        text = f.read_text(encoding="utf-8", errors="replace")
        doc = PAGE_SUFFIX.sub("", f.stem)
        n_tok = count(text)
        rows.append((f.name, doc, len(text), len(text.split()), n_tok))
        per_doc[doc].append(n_tok)
        if i % 200 == 0:
            print(f"  ... {i}/{len(files)}", file=sys.stderr)

    total_chars = sum(r[2] for r in rows)
    total_words = sum(r[3] for r in rows)
    total_tokens = sum(r[4] for r in rows)
    chunk_tokens = [r[4] for r in rows]

    print()
    print("== Totals ==")
    print(f"documents:        {len(per_doc):>12,}")
    print(f"chunks:           {len(rows):>12,}")
    print(f"characters:       {total_chars:>12,}")
    print(f"words:            {total_words:>12,}")
    print(f"tokens:           {total_tokens:>12,}")
    print(f"tokens per word:  {total_tokens / max(total_words, 1):>12.2f}")
    print(f"chars per token:  {total_chars / max(total_tokens, 1):>12.2f}")

    print()
    print("== Per-chunk token distribution ==")
    print(f"min:    {min(chunk_tokens):>8,}")
    print(f"median: {int(statistics.median(chunk_tokens)):>8,}")
    print(f"p90:    {percentile(chunk_tokens, 0.90):>8,}")
    print(f"max:    {max(chunk_tokens):>8,}")
    over_4k = sum(1 for t in chunk_tokens if t > 4096)
    print(f"chunks over 4,096 tokens: {over_4k} (these need re-splitting for a 4k training window)")

    print()
    print(f"== Largest {args.top} documents by tokens ==")
    ranked = sorted(per_doc.items(), key=lambda kv: sum(kv[1]), reverse=True)[: args.top]
    width = max(len(d) for d, _ in ranked)
    print(f"{'document':<{width}}  {'chunks':>6}  {'tokens':>10}  {'share':>6}")
    for doc, toks in ranked:
        s = sum(toks)
        print(f"{doc:<{width}}  {len(toks):>6}  {s:>10,}  {100 * s / total_tokens:>5.1f}%")

    print()
    print("== Epoch math (for planning) ==")
    for epochs in (1, 3, 5):
        print(f"{epochs} epoch(s): {total_tokens * epochs:>12,} tokens seen")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "document", "chars", "words", "tokens"])
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
