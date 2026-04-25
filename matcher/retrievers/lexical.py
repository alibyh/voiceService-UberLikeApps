"""Stage 3a — lexical retriever using RapidFuzz over normalized strings."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

from ..normalize import normalize_forms


@dataclass
class LexicalCandidate:
    place_id: int
    score: float  # 0..1
    matched_string: str


class LexicalIndex:
    """Flat list of (normalized_variant, place_id). Pickled at build time."""

    def __init__(self, entries: list[tuple[str, int]]) -> None:
        self.entries = entries
        # Cache the parallel arrays RapidFuzz wants.
        self._strings = [s for s, _ in entries]
        self._ids = [i for _, i in entries]

    @classmethod
    def build(cls, places: list[dict]) -> "LexicalIndex":
        entries: list[tuple[str, int]] = []
        for place in places:
            pid = place["id"]
            seen: set[str] = set()
            for v in [place["canonicalName"], *place.get("variants", [])]:
                for form in normalize_forms(v):
                    if form and form not in seen:
                        entries.append((form, pid))
                        seen.add(form)
        return cls(entries)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "LexicalIndex":
        with open(path, "rb") as f:
            entries = pickle.load(f)
        return cls(entries)

    def search(self, query: str, top_k: int = 10) -> list[LexicalCandidate]:
        forms = normalize_forms(query)
        # Score every form, then fold to best per place_id.
        best: dict[int, LexicalCandidate] = {}
        for form in forms:
            if not form:
                continue
            # Pull a generous pool — we'll fold by place_id below.
            results = process.extract(
                form,
                self._strings,
                scorer=fuzz.token_set_ratio,
                limit=top_k * 4,
            )
            for matched_string, score, idx in results:
                pid = self._ids[idx]
                norm_score = score / 100.0
                prev = best.get(pid)
                if prev is None or norm_score > prev.score:
                    best[pid] = LexicalCandidate(
                        place_id=pid,
                        score=norm_score,
                        matched_string=matched_string,
                    )
        return sorted(best.values(), key=lambda c: c.score, reverse=True)[:top_k]
