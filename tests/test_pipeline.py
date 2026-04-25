"""End-to-end pipeline test using the heuristic reranker.

The strict-pass-rate (correct id == top 1) and loose-pass-rate (correct id in
top 3) are the two key quality metrics. The heuristic reranker without LLM
will not hit production-quality numbers — those targets only apply when
ANTHROPIC_API_KEY is set and the Haiku reranker is in use. We assert a
loose-pass floor here so this catches regressions without flapping.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from matcher.pipeline import Pipeline
from matcher.retrievers.lexical import LexicalIndex
from matcher.retrievers.phonetic import PhoneticIndex


GOLDEN = Path(__file__).parent / "golden_cases.json"
PLACES = Path(__file__).resolve().parent.parent / "matcher" / "data" / "places.json"


@pytest.fixture(scope="module")
def pipeline():
    places = [p for p in json.loads(PLACES.read_text(encoding="utf-8")) if "id" in p]
    lex = LexicalIndex.build(places)
    pho = PhoneticIndex.build(places)
    return Pipeline(lexical=lex, phonetic=pho, semantic=None)


def _resolve_ids(pipeline: Pipeline, query: str) -> list[int]:
    resp = asyncio.run(pipeline.resolve(query, top_k=3))
    return [m.id for m in resp.matches]


def test_loose_pass_rate_meets_floor(pipeline):
    """At least 70% of golden cases should have the correct id somewhere in top-3.

    Heuristic reranker only — the LLM reranker is the one that hits production
    quality. This floor is for regression detection.
    """
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    passes = 0
    failures: list[str] = []
    for case in cases:
        ids = _resolve_ids(pipeline, case["query"])
        if case["expected_id"] in ids:
            passes += 1
        else:
            failures.append(f"  {case['query']!r} -> {ids} (expected {case['expected_id']})")
    rate = passes / len(cases)
    msg = f"loose pass rate {rate:.2%} ({passes}/{len(cases)})"
    if failures:
        msg += "\nFailures:\n" + "\n".join(failures)
    assert rate >= 0.70, msg
