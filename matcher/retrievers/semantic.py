"""Stage 3c — semantic retriever (BGE-M3 + FAISS).

The model is heavy (~1GB). It is loaded lazily on first use and reused
across requests via a module-level cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config import (
    EMBEDDINGS_FILE,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    settings,
)
from ..normalize import normalize_forms

if TYPE_CHECKING:  # avoid heavy imports at module load
    import faiss
    from sentence_transformers import SentenceTransformer


@dataclass
class SemanticCandidate:
    place_id: int
    score: float
    matched_string: str


_model_cache: "SentenceTransformer | None" = None


def _get_model() -> "SentenceTransformer":
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer

        _model_cache = SentenceTransformer(settings.embedding_model)
    return _model_cache


def _embed(texts: list[str]) -> np.ndarray:
    model = _get_model()
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
    )
    return embs.astype("float32")


class SemanticIndex:
    """FAISS IndexFlatIP over normalized BGE-M3 embeddings."""

    def __init__(
        self,
        index: "faiss.Index",
        ids: list[int],
        strings: list[str],
        embeddings: np.ndarray | None = None,
    ) -> None:
        self._index = index
        self._ids = ids
        self._strings = strings
        self._embeddings = embeddings  # only kept after build, dropped after save+load

    @classmethod
    def build(cls, places: list[dict]) -> "SemanticIndex":
        import faiss

        # One row per (variant_form, place_id).
        rows: list[tuple[str, int]] = []
        for place in places:
            pid = place["id"]
            seen: set[str] = set()
            for v in [place["canonicalName"], *place.get("variants", [])]:
                for norm in normalize_forms(v):
                    if norm and norm not in seen:
                        rows.append((norm, pid))
                        seen.add(norm)

        strings = [s for s, _ in rows]
        ids = [i for _, i in rows]

        embeddings = _embed(strings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        return cls(index, ids, strings, embeddings)

    def save(self, dir_path: Path) -> None:
        import faiss

        if self._embeddings is None:
            raise RuntimeError("Embeddings only available right after build().")
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dir_path / FAISS_INDEX_FILE))
        np.save(dir_path / EMBEDDINGS_FILE, self._embeddings)
        with open(dir_path / METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"ids": self._ids, "strings": self._strings},
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, dir_path: Path) -> "SemanticIndex":
        import faiss

        index = faiss.read_index(str(dir_path / FAISS_INDEX_FILE))
        with open(dir_path / METADATA_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(index, meta["ids"], meta["strings"])

    def search(self, query: str, top_k: int = 10) -> list[SemanticCandidate]:
        forms = [f for f in normalize_forms(query) if f]
        if not forms:
            return []

        # Embed all forms together; take the max similarity per place.
        query_embs = _embed(forms)
        # FAISS top_k * 4 to get a good pool before folding by place_id.
        k = min(top_k * 4, len(self._ids))
        scores, idxs = self._index.search(query_embs, k)

        best: dict[int, SemanticCandidate] = {}
        for row in range(query_embs.shape[0]):
            for s, i in zip(scores[row], idxs[row]):
                if i < 0:
                    continue
                pid = self._ids[i]
                norm_score = float((s + 1.0) / 2.0)  # cosine [-1,1] → [0,1]
                prev = best.get(pid)
                if prev is None or norm_score > prev.score:
                    best[pid] = SemanticCandidate(
                        place_id=pid,
                        score=norm_score,
                        matched_string=self._strings[i],
                    )
        return sorted(best.values(), key=lambda c: c.score, reverse=True)[:top_k]
