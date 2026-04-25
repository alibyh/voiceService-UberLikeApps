"""CLI: python -m matcher.cli "بتروديس الشارة" [--lat 18.10 --lon -15.96]"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict

from .pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="matcher.cli")
    parser.add_argument("query", help="text query in any script")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="skip the BGE-M3 retriever (faster, lower recall)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="skip the LLM reranker; use the heuristic fallback instead",
    )
    args = parser.parse_args()

    pipeline = Pipeline.load(
        with_semantic=not args.no_semantic,
        with_reranker=not args.no_rerank,
    )

    resp = asyncio.run(
        pipeline.resolve(
            args.query,
            user_lat=args.lat,
            user_lon=args.lon,
            top_k=args.top_k,
        )
    )
    print(
        json.dumps(
            {
                "matches": [asdict(m) for m in resp.matches],
                "needsConfirmation": resp.needsConfirmation,
                "confirmationPrompt": resp.confirmationPrompt,
                "debug": resp.debug,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
