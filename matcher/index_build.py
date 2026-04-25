"""One-shot index builder.

Reads places.json, builds the lexical, phonetic, and semantic indexes, and
writes them to INDEX_DIR. Run with: `python -m matcher.index_build`.

Pass --skip-semantic to build only the lexical+phonetic indexes (much faster,
no heavy deps required). Recall will be lower without the semantic index.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import (
    LEXICAL_INDEX_FILE,
    PHONETIC_INDEX_FILE,
    settings,
)
from .retrievers.lexical import LexicalIndex
from .retrievers.phonetic import PhoneticIndex


def _load_places(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cleaned: list[dict] = []
    skipped = 0
    for entry in raw:
        if "id" not in entry:
            skipped += 1
            continue
        if not entry.get("canonicalName"):
            skipped += 1
            continue
        cleaned.append(entry)
    if skipped:
        print(
            f"[index_build] warning: skipped {skipped} entry/entries missing id or canonicalName",
            file=sys.stderr,
        )
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(prog="matcher.index_build")
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="skip the BGE-M3 + FAISS index (faster, no heavy deps)",
    )
    args = parser.parse_args()

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    print(f"[index_build] reading {settings.places_path}")
    places = _load_places(settings.places_path)
    print(f"[index_build] loaded {len(places)} places")

    print("[index_build] building lexical index...")
    lex = LexicalIndex.build(places)
    lex.save(settings.index_dir / LEXICAL_INDEX_FILE)
    print(f"  -> {len(lex.entries)} (string, id) entries")

    print("[index_build] building phonetic index...")
    pho = PhoneticIndex.build(places)
    pho.save(settings.index_dir / PHONETIC_INDEX_FILE)
    print(f"  -> {len(pho.entries)} entries")

    if args.skip_semantic:
        print("[index_build] --skip-semantic set; not building FAISS index.")
        print("[index_build] done.")
        return

    print("[index_build] building semantic index (BGE-M3, this can take a while)...")
    # Lazy import keeps lexical+phonetic-only environments lighter.
    from .retrievers.semantic import SemanticIndex

    sem = SemanticIndex.build(places)
    sem.save(settings.index_dir)
    print(f"  -> {len(sem._strings)} embeddings written to {settings.index_dir}")

    print("[index_build] done.")


if __name__ == "__main__":
    main()
