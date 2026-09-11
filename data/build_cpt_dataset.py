#!/usr/bin/env python3
"""Step 3: build the continued-pretraining dataset from the KUKA corpus.

Applies the rules from lesson 2, in order, and reports tokens after each:

  1. drop excluded documents   (older fleet releases, duplicate PDFs/notes)
  2. strip YAML front matter   (the --- block at the top of every chunk)
  3. clean layout              (page footers, indentation, hyphenated breaks)
  4. drop exact-duplicate texts, then paragraph-level near-duplicates:
     documents are visited largest-first and a paragraph is dropped when
     --para-dedupe of its 8-word shingles were already seen in an earlier
     paragraph. This is what removes the safety/transport/maintenance
     chapters that every KMP platform manual repeats, while keeping each
     manual's model-specific tables and values.
  5. re-chunk to --max-tokens at paragraph boundaries, drop tiny pieces
  6. mix in general text       (data/out/general.jsonl, --mix fraction)
  7. split train / valid / test

Outputs data/out/cpt/{train,valid,test}.jsonl with {"text": ...} rows, a
provenance.jsonl mapping each KUKA row back to its parent document and
pages, and stats.json with the per-stage token table.

Usage:
  python data/build_cpt_dataset.py [CORPUS_DIR] [--mix 0.15] [--max-tokens 2048]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from find_duplicates import shingles  # noqa: E402

PAGE_SUFFIX = re.compile(r"-p(\d{3})-(\d{3})$")
FRONT = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
FOOTER = re.compile(r"^.*\|\s*\d+\s*/\s*\d+\s*$", re.MULTILINE)          # "KUKA.AMR Fleet 2.16 V1 | Issued: ... | 77/755"
FOOTER2 = re.compile(r"^(?:KUKA Deutschland GmbH|www\.kuka\.com|Issued:).*$", re.MULTILINE)
FOOTER3 = re.compile(r"^(?=.*www\.kuka\.com)(?=.*\|).*$|^\d+\s*/\s*\d+\s*\|.*$", re.MULTILINE)  # any footer line with the domain and a pipe, either order
HYPHEN_BREAK = re.compile(r"([a-z])-\n[ \t]*([a-z])")
LEADING_WS = re.compile(r"^[ \t]+", re.MULTILINE)
TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
MANY_BLANKS = re.compile(r"\n{3,}")

# Rules from lesson 2. Substrings matched against the document stem.
DEFAULT_EXCLUDE = [
    "kuka_amr_fleet_211",              # superseded by 2.16
    "kuka_amr_fleet_214",              # superseded by 2.16
    "kuka_amr_fleet_215",              # superseded by 2.16
    "kuka-technical-note--qr-code-deployment--v200",  # superseded by kukaamr---qr-code-deployment-210
    "ba_kmp_600p-u-d_series_en",       # byte-identical to ba_kmp_600p_series_en_v1
    "kukaamr---reflector-deployment",  # duplicate of kuka-technical-note--reflector-deployment
]


def clean(text: str) -> str:
    text = FRONT.sub("", text, count=1)
    text = FOOTER.sub("", text)
    text = FOOTER2.sub("", text)
    text = FOOTER3.sub("", text)
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    text = LEADING_WS.sub("", text)
    text = TRAILING_WS.sub("", text)
    text = MANY_BLANKS.sub("\n\n", text)
    return text.strip()


def rechunk(text: str, count, max_tokens: int) -> list[str]:
    """Split on blank lines and greedily pack paragraphs up to max_tokens."""
    if count(text) <= max_tokens:
        return [text]
    pieces, acc, acc_tok = [], [], 0
    for para in text.split("\n\n"):
        n = count(para)
        if acc and acc_tok + n > max_tokens:
            pieces.append("\n\n".join(acc))
            acc, acc_tok = [], 0
        acc.append(para)
        acc_tok += n
    if acc:
        pieces.append("\n\n".join(acc))
    return pieces


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", nargs="?", default="../kuka-mcp/knowledge")
    ap.add_argument("--general", default="data/out/general.jsonl")
    ap.add_argument("--mix", type=float, default=0.15, help="general text as a fraction of KUKA tokens")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--min-tokens", type=int, default=48)
    ap.add_argument("--para-dedupe", type=float, default=0.8,
                    help="drop a paragraph when this fraction of its shingles was already seen (0 disables)")
    ap.add_argument("--exclude", action="append", default=None, help="doc-stem substring to drop (repeatable)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/out/cpt")
    args = ap.parse_args()
    exclude = args.exclude if args.exclude is not None else DEFAULT_EXCLUDE

    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(args.tokenizer)
    count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)

    files = sorted(Path(args.corpus).glob("*.md"))
    if not files:
        print(f"error: no *.md under {args.corpus}", file=sys.stderr)
        return 1

    stage: list[tuple[str, int, int]] = []   # (name, rows, tokens)

    # 0. raw
    raw = [(f, f.read_text(encoding="utf-8", errors="replace")) for f in files]
    stage.append(("raw corpus", len(raw), sum(count(t) for _, t in raw)))

    # 1. exclude documents
    kept = [(f, t) for f, t in raw if not any(x in f.stem for x in exclude)]
    stage.append(("after dropping excluded docs", len(kept), sum(count(t) for _, t in kept)))

    # 2+3. front matter + layout cleaning
    cleaned = [(f, clean(t)) for f, t in kept]
    no_front = sum(count(FRONT.sub("", t, count=1)) for _, t in kept)
    stage.append(("after stripping front matter", len(cleaned), no_front))
    stage.append(("after layout cleaning", len(cleaned), sum(count(t) for _, t in cleaned)))

    # 4. exact dupes after cleaning
    seen_hash, uniq = set(), []
    for f, t in cleaned:
        h = hashlib.sha256(t.encode()).hexdigest()
        if h in seen_hash or not t:
            continue
        seen_hash.add(h)
        uniq.append((f, t))
    stage.append(("after exact-dedupe", len(uniq), sum(count(t) for _, t in uniq)))

    # 4b. paragraph-level near-dedupe, largest documents first so the fullest
    # manual keeps its text and the variants keep only what differs.
    if args.para_dedupe > 0:
        doc_of = lambda f: PAGE_SUFFIX.sub("", f.stem)
        doc_size = Counter()
        for f, t in uniq:
            doc_size[doc_of(f)] += len(t)
        order = sorted(uniq, key=lambda ft: (-doc_size[doc_of(ft[0])], ft[0].name))
        seen_sh: set[int] = set()
        deduped, dropped_paras = [], 0
        for f, t in order:
            keep_paras = []
            for para in t.split("\n\n"):
                sh = shingles(para, 8)
                if sh:
                    contained = len(sh & seen_sh) / len(sh)
                    if contained >= args.para_dedupe:
                        dropped_paras += 1
                        continue
                    seen_sh |= sh
                keep_paras.append(para)
            body = "\n\n".join(keep_paras).strip()
            if body:
                deduped.append((f, body))
        uniq = sorted(deduped, key=lambda ft: ft[0].name)
        stage.append((f"after paragraph near-dedupe (>= {args.para_dedupe:.0%} seen; {dropped_paras} paras)",
                      len(uniq), sum(count(t) for _, t in uniq)))

    # 5. re-chunk
    rows, prov = [], []
    for f, t in uniq:
        m = PAGE_SUFFIX.search(f.stem)
        parent = PAGE_SUFFIX.sub("", f.stem)
        pages = f"{int(m.group(1))}-{int(m.group(2))}" if m else ""
        for i, piece in enumerate(rechunk(t, count, args.max_tokens)):
            n = count(piece)
            if n < args.min_tokens:
                continue
            rid = f"{f.stem}#{i}"
            rows.append((rid, piece, n))
            prov.append({"id": rid, "file": f.name, "parent": parent, "pages": pages, "tokens": n})
    kuka_tokens = sum(n for _, _, n in rows)
    stage.append((f"after re-chunking to <= {args.max_tokens} tokens", len(rows), kuka_tokens))

    # unique-text estimate on the cleaned KUKA rows
    by_doc: dict[str, list[str]] = {}
    for rid, piece, _ in rows:
        by_doc.setdefault(PAGE_SUFFIX.sub("", rid.split("#")[0]), []).append(piece)
    doc_tok = {d: sum(count(p) for p in ps) for d, ps in by_doc.items()}
    seen_sh: set[int] = set()
    unique = 0.0
    for d in sorted(by_doc, key=lambda d: -doc_tok[d]):
        sh = shingles("\n".join(by_doc[d]), 8)
        if sh:
            unique += doc_tok[d] * len(sh - seen_sh) / len(sh)
            seen_sh |= sh

    # 6. general mix
    general = []
    gp = Path(args.general)
    target = int(kuka_tokens * args.mix)
    if gp.exists() and args.mix > 0:
        acc = 0
        with gp.open(encoding="utf-8") as fh:
            for line in fh:
                t = json.loads(line)["text"]
                n = count(t)
                general.append((f"general#{len(general)}", t, n))
                acc += n
                if acc >= target:
                    break
        stage.append((f"general text mixed in ({args.mix:.0%} target)", len(general), acc))
    else:
        print(f"warning: no general text at {gp}; run data/fetch_general.py first", file=sys.stderr)

    # 7. split. KUKA and general are split separately so each set keeps the mix ratio.
    rng = random.Random(args.seed)
    def split(items):
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        n_hold = max(1, round(n * 0.025))
        return {"test": items[:n_hold], "valid": items[n_hold:2 * n_hold], "train": items[2 * n_hold:]}
    k, g = split(rows), split(general) if general else {"test": [], "valid": [], "train": []}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    totals = {}
    for name in ("train", "valid", "test"):
        items = k[name] + g[name]
        rng.shuffle(items)
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for _, t, _ in items:
                fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        totals[name] = (len(items), sum(n for _, _, n in items))
    with (out / "provenance.jsonl").open("w", encoding="utf-8") as fh:
        for p in prov:
            fh.write(json.dumps(p) + "\n")

    # report
    print(f"{'stage':<45} {'rows':>6} {'tokens':>11}  {'vs raw':>7}")
    base = stage[0][2]
    for name, n, t in stage:
        print(f"{name:<45} {n:>6} {t:>11,}  {100 * t / base:>6.1f}%")
    print()
    print(f"KUKA unique-text estimate after cleaning: {100 * unique / max(kuka_tokens, 1):.0f}%  (was 60% before)")
    print(f"documents kept: {len(by_doc)}   excluded patterns: {exclude}")
    print()
    for name, (n, t) in totals.items():
        print(f"{name:<6} {n:>6} rows  {t:>11,} tokens")
    print(f"\nwrote {out}/  (train/valid/test.jsonl, provenance.jsonl, stats.json)")
    (out / "stats.json").write_text(json.dumps({
        "stages": [{"stage": s, "rows": n, "tokens": t} for s, n, t in stage],
        "unique_text_fraction": unique / max(kuka_tokens, 1),
        "splits": {k_: {"rows": n, "tokens": t} for k_, (n, t) in totals.items()},
        "exclude": exclude, "max_tokens": args.max_tokens, "mix": args.mix, "seed": args.seed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
