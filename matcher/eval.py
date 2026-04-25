"""`make eval` entry point: runs golden_cases.json and reports pass-rate.

Strict-pass = expected id is the top-1 result.
Loose-pass  = expected id is in top-3.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .pipeline import Pipeline


GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden_cases.json"


async def _run() -> None:
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    pipeline = Pipeline.load()

    strict, loose = 0, 0
    failures: list[dict] = []
    started = time.time()
    for case in cases:
        resp = await pipeline.resolve(case["query"], top_k=3)
        ids = [m.id for m in resp.matches]
        is_strict = bool(ids) and ids[0] == case["expected_id"]
        is_loose = case["expected_id"] in ids
        if is_strict:
            strict += 1
        if is_loose:
            loose += 1
        else:
            failures.append(
                {
                    "query": case["query"],
                    "expected_id": case["expected_id"],
                    "got_ids": ids,
                    "note": case.get("note", ""),
                }
            )
    elapsed = time.time() - started

    n = len(cases)
    print(f"Cases:  {n}")
    print(f"Strict: {strict}/{n} = {strict / n:.2%}  (expected_id == top-1)")
    print(f"Loose:  {loose}/{n} = {loose / n:.2%}  (expected_id in top-3)")
    print(f"Time:   {elapsed:.1f}s ({elapsed / n * 1000:.0f}ms/case)")

    if failures:
        print("\nFailures:")
        for f in failures:
            print(
                f"  {f['query']!r:40} expected={f['expected_id']:>5}  got={f['got_ids']}  {f['note']}"
            )


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
