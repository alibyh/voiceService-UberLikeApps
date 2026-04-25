"""Stage 3b — phonetic retriever.

Both query and variants are projected through arabic_to_latin_phonetic + metaphone
into a shared sound-space, then fuzzy-matched.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

from ..normalize import normalize_forms
from ..transliterate import phonetic_key


@dataclass
class PhoneticCandidate:
    place_id: int
    score: float
    matched_string: str


class PhoneticIndex:
    def __init__(self, entries: list[tuple[str, int]]) -> None:
        # entries are (phonetic_key, place_id)
        self.entries = entries
        self._strings = [s for s, _ in entries]
        self._ids = [i for _, i in entries]

    @classmethod
    def build(cls, places: list[dict]) -> "PhoneticIndex":
        entries: list[tuple[str, int]] = []
        for place in places:
            pid = place["id"]
            seen: set[str] = set()
            for v in [place["canonicalName"], *place.get("variants", [])]:
                for norm in normalize_forms(v):
                    key = phonetic_key(norm)
                    if key and key not in seen:
                        entries.append((key, pid))
                        seen.add(key)
        return cls(entries)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "PhoneticIndex":
        with open(path, "rb") as f:
            entries = pickle.load(f)
        return cls(entries)

    def search(self, query: str, top_k: int = 10) -> list[PhoneticCandidate]:
        best: dict[int, PhoneticCandidate] = {}
        for form in normalize_forms(query):
            qkey = phonetic_key(form)
            if not qkey:
                continue
            results = process.extract(
                qkey,
                self._strings,
                scorer=fuzz.token_set_ratio,
                limit=top_k * 4,
            )
            for matched_string, score, idx in results:
                pid = self._ids[idx]
                norm = score / 100.0
                prev = best.get(pid)
                if prev is None or norm > prev.score:
                    best[pid] = PhoneticCandidate(
                        place_id=pid,
                        score=norm,
                        matched_string=matched_string,
                    )
        return sorted(best.values(), key=lambda c: c.score, reverse=True)[:top_k]
