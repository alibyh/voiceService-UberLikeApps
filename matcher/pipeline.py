"""Stages 1→5 orchestrator."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import BinaryIO

from .config import (
    LEXICAL_INDEX_FILE,
    PHONETIC_INDEX_FILE,
    settings,
)
from .rerank import (
    HeuristicReranker,
    RerankCandidate,
    haversine_meters,
)
from .retrievers.lexical import LexicalIndex
from .retrievers.phonetic import PhoneticIndex


@dataclass
class Match:
    id: int
    canonicalName: str
    lat: float
    lon: float
    confidence: float
    matchedVariant: str
    distanceMeters: float | None = None
    reason: str | None = None


@dataclass
class ResolveResponse:
    matches: list[Match]
    needsConfirmation: bool
    confirmationPrompt: str | None = None
    debug: dict = field(default_factory=dict)


def _load_places_dict() -> dict[int, dict]:
    with open(settings.places_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {p["id"]: p for p in raw if "id" in p}


class Pipeline:
    """Holds the loaded indexes + reranker. Construct once at startup."""

    def __init__(
        self,
        lexical: LexicalIndex,
        phonetic: PhoneticIndex,
        semantic=None,  # SemanticIndex or None if not built
        reranker=None,
        asr_backend=None,
        bias_prompt: str | None = None,
    ) -> None:
        self.lexical = lexical
        self.phonetic = phonetic
        self.semantic = semantic
        self.reranker = reranker or HeuristicReranker()
        self.asr_backend = asr_backend
        self.bias_prompt = bias_prompt
        self.places = _load_places_dict()
        self._executor = ThreadPoolExecutor(max_workers=3)

    @classmethod
    def load(cls, *, with_semantic: bool = True, with_reranker: bool = True) -> "Pipeline":
        lex = LexicalIndex.load(settings.index_dir / LEXICAL_INDEX_FILE)
        pho = PhoneticIndex.load(settings.index_dir / PHONETIC_INDEX_FILE)
        sem = None
        if with_semantic:
            try:
                from .retrievers.semantic import SemanticIndex

                sem = SemanticIndex.load(settings.index_dir)
            except (FileNotFoundError, ImportError, RuntimeError) as exc:
                print(f"[pipeline] semantic index unavailable, skipping: {exc}")
                sem = None

        reranker = None
        if with_reranker:
            from .rerank import default_reranker

            reranker = default_reranker()

        return cls(
            lexical=lex,
            phonetic=pho,
            semantic=sem,
            reranker=reranker,
        )

    # --------------------------------------------------------------------
    # Stage 1 — ASR

    def transcribe(self, audio: BinaryIO, filename: str | None = None) -> list[str]:
        if self.asr_backend is None:
            from .asr import default_backend

            self.asr_backend = default_backend()
        return self.asr_backend.transcribe(
            audio, prompt=self.bias_prompt, filename=filename
        )

    # --------------------------------------------------------------------
    # Stage 3 — fan-out retrieval (parallel)

    async def _retrieve(self, query: str, per_retriever: int = 10) -> dict[int, dict[str, float]]:
        """Run all retrievers in parallel and return {place_id: {retriever: score}}."""
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(self._executor, self.lexical.search, query, per_retriever),
            loop.run_in_executor(self._executor, self.phonetic.search, query, per_retriever),
        ]
        if self.semantic is not None:
            tasks.append(
                loop.run_in_executor(self._executor, self.semantic.search, query, per_retriever)
            )

        results = await asyncio.gather(*tasks)
        retriever_names = ["lexical", "phonetic"]
        if self.semantic is not None:
            retriever_names.append("semantic")

        merged: dict[int, dict[str, float]] = {}
        matched_strings: dict[int, str] = {}
        for name, hits in zip(retriever_names, results):
            for c in hits:
                merged.setdefault(c.place_id, {})[name] = c.score
                # Keep the matched string from the first retriever that hit.
                matched_strings.setdefault(c.place_id, c.matched_string)

        # Stash matched strings on the dict — pipeline.resolve uses them later.
        merged["__matched_strings__"] = matched_strings  # type: ignore[assignment]
        return merged

    # --------------------------------------------------------------------
    # End-to-end resolve

    async def resolve(
        self,
        query: str,
        *,
        user_lat: float | None = None,
        user_lon: float | None = None,
        top_k: int = 3,
        extra_queries: list[str] | None = None,
    ) -> ResolveResponse:
        t0 = time.time()
        queries = [query, *(extra_queries or [])]

        # Run retrieval for every query string and union by place_id.
        union: dict[int, dict[str, float]] = {}
        union_strings: dict[int, str] = {}
        for q in queries:
            merged = await self._retrieve(q)
            mstrings = merged.pop("__matched_strings__")  # type: ignore[arg-type]
            for pid, scores in merged.items():
                slot = union.setdefault(pid, {})
                for k, v in scores.items():
                    if v > slot.get(k, 0.0):
                        slot[k] = v
            for pid, s in mstrings.items():
                union_strings.setdefault(pid, s)

        # Fuse + cap at 15 — order by max retriever score.
        fused = sorted(
            union.items(),
            key=lambda kv: max(kv[1].values()) if kv[1] else 0.0,
            reverse=True,
        )[:15]

        # Build RerankCandidate list (lookup canonical name, variants, distance).
        rerank_inputs: list[RerankCandidate] = []
        for pid, scores in fused:
            place = self.places.get(pid)
            if not place:
                continue
            distance = None
            if user_lat is not None and user_lon is not None:
                distance = haversine_meters(user_lat, user_lon, place["lat"], place["lon"])
            rerank_inputs.append(
                RerankCandidate(
                    place_id=pid,
                    canonical_name=place["canonicalName"],
                    variants=place.get("variants", []),
                    distance_meters=distance,
                    retriever_scores=scores,
                )
            )

        ranked = self.reranker.rerank(query, rerank_inputs, user_lat=user_lat, user_lon=user_lon)
        # Map back to Match objects.
        rerank_by_id = {r.place_id: r for r in ranked}
        ordered_matches: list[Match] = []
        for r in ranked[:top_k]:
            place = self.places.get(r.place_id)
            if not place:
                continue
            distance = None
            if user_lat is not None and user_lon is not None:
                distance = haversine_meters(user_lat, user_lon, place["lat"], place["lon"])
            ordered_matches.append(
                Match(
                    id=r.place_id,
                    canonicalName=place["canonicalName"],
                    lat=place["lat"],
                    lon=place["lon"],
                    confidence=round(r.confidence, 4),
                    matchedVariant=union_strings.get(r.place_id, place["canonicalName"]),
                    distanceMeters=round(distance, 1) if distance is not None else None,
                    reason=r.reason or None,
                )
            )

        # Stage 5 — confidence gate.
        needs_confirm = False
        prompt = None
        if len(ordered_matches) >= 2:
            top, second = ordered_matches[0], ordered_matches[1]
            if (
                top.confidence < settings.confidence_threshold
                or top.confidence - second.confidence < settings.confidence_margin
            ):
                needs_confirm = True
                prompt = f"هل تقصد {top.canonicalName} أم {second.canonicalName}؟"
        elif len(ordered_matches) == 1:
            if ordered_matches[0].confidence < settings.confidence_threshold:
                needs_confirm = True
                prompt = f"هل تقصد {ordered_matches[0].canonicalName}؟"

        elapsed_ms = int((time.time() - t0) * 1000)
        response = ResolveResponse(
            matches=ordered_matches,
            needsConfirmation=needs_confirm,
            confirmationPrompt=prompt,
            debug={
                "elapsedMs": elapsed_ms,
                "candidatesConsidered": len(rerank_inputs),
                "queries": queries,
            },
        )
        self._log(query, queries, response)
        return response

    # --------------------------------------------------------------------
    # Logging — every resolution lands in JSONL for future fine-tuning.

    def _log(self, query: str, queries: list[str], response: ResolveResponse) -> None:
        try:
            settings.log_file.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": time.time(),
                "query_hash": hashlib.sha1(query.encode("utf-8")).hexdigest()[:12],
                "transcripts": queries,
                "matches": [asdict(m) for m in response.matches],
                "needsConfirmation": response.needsConfirmation,
            }
            with open(settings.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            # Never let logging take down a request.
            pass


# A simple sync wrapper for callers that don't want asyncio.
def resolve_sync(pipeline: Pipeline, query: str, **kwargs) -> ResolveResponse:
    return asyncio.run(pipeline.resolve(query, **kwargs))
