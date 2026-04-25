"""Stage 4 — LLM reranker (Claude Haiku).

Receives the top ~15 fused candidates and returns scored, ordered IDs.
Uses tool-call mode for guaranteed JSON output.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass

from .config import settings


logger = logging.getLogger("matcher.rerank")


@dataclass
class RerankCandidate:
    place_id: int
    canonical_name: str
    variants: list[str]
    distance_meters: float | None
    retriever_scores: dict[str, float]  # {"lexical": 0.83, "phonetic": 0.91, ...}


@dataclass
class RerankResult:
    place_id: int
    confidence: float
    reason: str


_RERANK_TOOL = {
    "name": "rank_candidates",
    "description": "Return the candidates ranked by how well they match the user's spoken query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ranked": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "0=clearly wrong, 1=certain match",
                        },
                        "reason": {"type": "string", "maxLength": 200},
                    },
                    "required": ["id", "confidence", "reason"],
                },
            }
        },
        "required": ["ranked"],
    },
}


def _format_candidate(idx: int, c: RerankCandidate) -> str:
    variants_preview = " | ".join(c.variants[:3]) if c.variants else "(none)"
    dist = f"{c.distance_meters:.0f}m" if c.distance_meters is not None else "unknown"
    score_str = ", ".join(f"{k}={v:.2f}" for k, v in c.retriever_scores.items())
    return (
        f"{idx}. id={c.place_id} | canonical=\"{c.canonical_name}\" | "
        f"variants=[{variants_preview}] | distance={dist} | retrievers=[{score_str}]"
    )


def build_prompt(
    query: str,
    candidates: list[RerankCandidate],
    user_lat: float | None,
    user_lon: float | None,
) -> str:
    loc_line = (
        f"User's current location: {user_lat:.5f}, {user_lon:.5f} (use distance as a tiebreaker)"
        if user_lat is not None and user_lon is not None
        else "User's current location: unknown"
    )
    cand_block = "\n".join(_format_candidate(i + 1, c) for i, c in enumerate(candidates))
    return (
        "You are matching a noisy voice transcription of a place name in Mauritania\n"
        "to a database. The user's speech may be in Hassaniya Arabic, French, or\n"
        "English, and the transcription may contain ASR errors.\n\n"
        f'User said: "{query}"\n'
        f"{loc_line}\n\n"
        "Candidates:\n"
        f"{cand_block}\n\n"
        "Score every candidate. The top candidate's confidence should reflect how\n"
        "certain you are that it is the place the user meant; lower confidences\n"
        "for candidates you are confident are wrong. Use the retriever scores and\n"
        "distance only as supporting signal — the canonical name and variants are\n"
        "the primary evidence."
    )


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class HaikuReranker:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.reranker_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            if not settings.anthropic_api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Add it to .env to use the reranker."
                )
            self._client = Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        user_lat: float | None = None,
        user_lon: float | None = None,
    ) -> list[RerankResult]:
        if not candidates:
            return []
        client = self._get_client()
        prompt = build_prompt(query, candidates, user_lat, user_lon)
        try:
            resp = client.messages.create(
                model=self.model,
                max_tokens=512,
                temperature=0,
                tools=[_RERANK_TOOL],
                tool_choice={"type": "tool", "name": "rank_candidates"},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            logger.error("Haiku rerank API call failed: %s", exc, exc_info=True)
            raise

        # Walk the response and pull the tool_use payload.
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "tool_use" and getattr(block, "name", None) == "rank_candidates":
                payload = block.input
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        logger.error("rerank tool_use payload was non-JSON string: %r", payload)
                        return []
                if not isinstance(payload, dict):
                    logger.error("rerank tool_use payload not a dict: %r", payload)
                    return []
                ranked_raw = payload.get("ranked", [])
                if not ranked_raw:
                    logger.warning("rerank returned empty ranked list. payload=%r", payload)
                    return []
                return [
                    RerankResult(
                        place_id=int(item["id"]),
                        confidence=float(item.get("confidence", 0.0)),
                        reason=str(item.get("reason", "")),
                    )
                    for item in ranked_raw
                ]

        # No matching tool_use block found — log what Claude actually returned
        # so we can diagnose without another round trip.
        try:
            debug_blocks = [
                {
                    "type": getattr(b, "type", None),
                    "name": getattr(b, "name", None),
                    "text": (getattr(b, "text", None) or "")[:300],
                }
                for b in resp.content
            ]
        except Exception:
            debug_blocks = []
        logger.error(
            "Haiku rerank: no rank_candidates tool_use block. stop_reason=%s blocks=%s",
            getattr(resp, "stop_reason", None),
            debug_blocks,
        )
        return []


class HeuristicReranker:
    """Fallback reranker that runs without an API key.

    Combines the per-retriever scores into one confidence and shrinks slightly
    for far-away places. Use this in tests and when ANTHROPIC_API_KEY is unset.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {"lexical": 0.35, "phonetic": 0.4, "semantic": 0.25}

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        user_lat: float | None = None,
        user_lon: float | None = None,
    ) -> list[RerankResult]:
        out: list[RerankResult] = []
        for c in candidates:
            wsum = sum(self.weights.values())
            score = sum(
                self.weights.get(k, 0.0) * v for k, v in c.retriever_scores.items()
            ) / wsum if wsum else 0.0
            # Mild distance penalty: -0.05 for >5km, -0.1 for >20km.
            if c.distance_meters is not None:
                if c.distance_meters > 20000:
                    score -= 0.10
                elif c.distance_meters > 5000:
                    score -= 0.05
            score = max(0.0, min(1.0, score))
            out.append(
                RerankResult(
                    place_id=c.place_id,
                    confidence=score,
                    reason="heuristic: weighted retriever fusion",
                )
            )
        out.sort(key=lambda r: r.confidence, reverse=True)
        return out


def default_reranker():
    """Pick Haiku if ANTHROPIC_API_KEY is set, otherwise the heuristic fallback."""
    if settings.anthropic_api_key:
        return HaikuReranker()
    return HeuristicReranker()
