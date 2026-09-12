#!/usr/bin/env python3
"""Find redundancy in the KUKA corpus before it inflates training.

Three kinds of repetition matter for training and are invisible to search:

  1. Exact duplicate chunks     - same bytes under two file names
  2. Near-duplicate documents   - manual v2.15 vs v2.16, or KMP 600P vs 600P-EU
  3. Boilerplate lines          - page headers, footers, copyright notices
                                  repeated on every page of every manual

Near-duplicates are measured with word shingles: every run of K consecutive
words is hashed, each document becomes a set of hashes, and two documents are
compared by Jaccard similarity |A & B| / |A | B|. It is the same idea as a
diff at paragraph granularity, robust to small edits and reorderings.

The greedy "unique tokens" pass keeps the largest document of each near-dup
group first and counts only shingles not already seen, giving a floor on how
much of the corpus is genuinely new text.

Usage:
  python data/find_duplicates.py [CORPUS_DIR] [--k 8] [--threshold 0.5]
                                 [--tokens-csv data/out/tokens.csv]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PAGE_SUFFIX = re.compile(r"-p\d{3}-\d{3}$")
WORD = re.compile(r"\S+")


def shingles(text: str, k: int) -> set[int]:
    words = WORD.findall(text.lower())
    return {hash(" ".join(words[i:i + k])) for i in range(max(0, len(words) - k + 1))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", nargs="?", default="../kuka-mcp/knowledge")
    ap.add_argument("--k", type=int, default=8, help="words per shingle")
    ap.add_argument("--threshold", type=float, default=0.5, help="Jaccard above which two docs are near-duplicates")
    ap.add_argument("--tokens-csv", default="data/out/tokens.csv", help="per-chunk token counts from step 1")
    ap.add_argument("--top-lines", type=int, default=12)
    args = ap.parse_args()

    files = sorted(Path(args.corpus).glob("*.md"))
    if not files:
        print(f"error: no *.md under {args.corpus}", file=sys.stderr)
        return 1

    # Token counts per chunk from step 1, if available, else chars/5.5
    tokens_by_file: dict[str, int] = {}
    p = Path(args.tokens_csv)
    if p.exists():
        with p.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                tokens_by_file[row["file"]] = int(row["tokens"])

    # ---- pass over files -------------------------------------------------
    exact: dict[str, list[str]] = defaultdict(list)
    doc_text: dict[str, list[str]] = defaultdict(list)
    doc_tokens: Counter[str] = Counter()
    line_counts: Counter[str] = Counter()
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        doc = PAGE_SUFFIX.sub("", f.stem)
        exact[hashlib.sha256(text.strip().encode()).hexdigest()].append(f.name)
        doc_text[doc].append(text)
        doc_tokens[doc] += tokens_by_file.get(f.name, len(text) // 5)
        for line in text.splitlines():
            s = line.strip()
            if 12 <= len(s) <= 120 and not s.startswith("#"):
                line_counts[s] += 1
    total_tokens = sum(doc_tokens.values())

    # ---- 1. exact duplicate chunks ---------------------------------------
    dup_groups = [names for names in exact.values() if len(names) > 1]
    dup_tokens = sum(tokens_by_file.get(n, 0) for g in dup_groups for n in g[1:])
    print("== 1. Exact duplicate chunks ==")
    print(f"groups: {len(dup_groups)}   redundant chunks: {sum(len(g) - 1 for g in dup_groups)}"
          f"   redundant tokens: {dup_tokens:,} ({100 * dup_tokens / total_tokens:.1f}%)")
    for g in sorted(dup_groups, key=len, reverse=True)[:5]:
        print("  " + "  ==  ".join(g[:3]) + ("  ..." if len(g) > 3 else ""))

    # ---- 2. near-duplicate documents ---------------------------------------
    doc_sh = {d: shingles("\n".join(t), args.k) for d, t in doc_text.items()}
    docs = sorted(doc_sh, key=lambda d: doc_tokens[d], reverse=True)
    pairs = []
    for i, a in enumerate(docs):
        for b in docs[i + 1:]:
            sa, sb = doc_sh[a], doc_sh[b]
            if not sa or not sb:
                continue
            inter = len(sa & sb)
            j = inter / len(sa | sb)
            if j >= args.threshold:
                pairs.append((j, a, b, inter / len(sb)))  # fraction of the smaller doc contained in the larger
    pairs.sort(reverse=True)
    print()
    print(f"== 2. Near-duplicate documents (Jaccard >= {args.threshold}, {args.k}-word shingles) ==")
    print(f"{'jaccard':>7}  {'contained':>9}  pair")
    for j, a, b, cont in pairs:
        print(f"{j:>7.2f}  {cont:>8.0%}  {a}  ~  {b}   ({doc_tokens[a]:,} / {doc_tokens[b]:,} tokens)")

    # union-find the pairs into groups
    parent = {d: d for d in docs}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for _, a, b, _ in pairs:
        parent[find(a)] = find(b)
    groups: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        groups[find(d)].append(d)
    near_groups = [g for g in groups.values() if len(g) > 1]
    keep_one = sum(sum(doc_tokens[d] for d in g) - max(doc_tokens[d] for d in g) for g in near_groups)
    print()
    print(f"groups: {len(near_groups)}   docs involved: {sum(len(g) for g in near_groups)}"
          f"   tokens saved if one per group kept: {keep_one:,} ({100 * keep_one / total_tokens:.1f}%)")
    for g in sorted(near_groups, key=lambda g: -sum(doc_tokens[d] for d in g)):
        print("  " + "  |  ".join(sorted(g, key=lambda d: -doc_tokens[d])))

    # ---- 3. greedy unique-shingle floor ------------------------------------
    seen: set[int] = set()
    unique_tokens = 0.0
    for d in docs:
        sh = doc_sh[d]
        if not sh:
            continue
        new = len(sh - seen)
        unique_tokens += doc_tokens[d] * new / len(sh)
        seen |= sh
    print()
    print("== 3. Genuinely new text (greedy, largest documents first) ==")
    print(f"total tokens: {total_tokens:,}   unique-text estimate: {unique_tokens:,.0f}"
          f" ({100 * unique_tokens / total_tokens:.0f}%)   repeated: {100 - 100 * unique_tokens / total_tokens:.0f}%")

    # ---- 4. boilerplate lines ------------------------------------------------
    print()
    print(f"== 4. Most repeated lines (headers, footers, notices) ==")
    boiler = 0
    for line, n in line_counts.most_common(args.top_lines):
        boiler += n
        print(f"{n:>6}  {line[:90]}")
    print(f"(top {args.top_lines} lines alone appear {boiler:,} times)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
