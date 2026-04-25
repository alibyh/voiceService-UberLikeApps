# Hassaniya Voice Place-Matching Service

Voice-driven place lookup for ride-hailing in Mauritania. A user speaks a place name in Hassaniya Arabic (often code-switching with French/English); the service transcribes, matches against a curated catalog despite ASR errors and dialectal variation, and returns the canonical name + GPS coordinates.

## Pipeline

```
audio ──► [1] ASR (Whisper, with biasing + N-best)
            │
            ▼
         transcripts[]
            │
            ▼
         [2] Normalizer ──► normalized query
            │
            ▼
         [3] Hybrid retrieval (parallel)
              ├─ Lexical    (RapidFuzz, token_set_ratio)
              ├─ Phonetic   (Hassaniya AR→Latin transliterator + Metaphone)
              └─ Semantic   (BGE-M3 embeddings + FAISS)
            │
            ▼
         candidates[] (top ~15, deduplicated)
            │
            ▼
         [4] LLM reranker (Claude Haiku, JSON tool-call)
            │
            ▼
         [5] Confidence gate ──► single match OR Arabic confirmation prompt
```

The **whole point** of the pipeline is symmetric normalization — every transformation applied to the query is also applied to every variant at index-build time. If you change `matcher/normalize.py` or `matcher/transliterate.py`, you must rebuild the indexes (`make index`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in ANTHROPIC_API_KEY and OPENAI_API_KEY in .env
```

### Where to put the API keys

Open `.env` (which you copied from `.env.example`) and fill in:

```
ANTHROPIC_API_KEY=sk-ant-...   # for the Claude Haiku reranker (Stage 4)
OPENAI_API_KEY=sk-...          # for Whisper ASR (Stage 1)
```

The service auto-loads `.env` at import time via `python-dotenv`. **Do not commit `.env`** — it's already in `.gitignore`.

If `ANTHROPIC_API_KEY` is unset, the pipeline falls back to a heuristic reranker (weighted retriever fusion). It works for sanity-checking but does not disambiguate well when several candidates share tokens. If `OPENAI_API_KEY` is unset, the audio branch of `/resolve` will return an error — text-mode `/resolve` and the CLI still work fully.

## Build the indexes

The service needs prebuilt indexes — they are not regenerated on each request.

```bash
make index                              # builds all three (lex + phonetic + semantic)
python -m matcher.index_build --skip-semantic  # lex + phonetic only (fast, no FAISS deps)
```

The semantic build downloads `BAAI/bge-m3` (~1GB) on first run.

Outputs land in `matcher/data/index/`:
- `lexical.pkl` — pickled (string, id) tuples
- `phonetic.pkl` — same, in phonetic space
- `faiss.idx` + `embeddings.npy` + `metadata.json` — semantic index

## Run

```bash
make dev          # uvicorn matcher.api:app on :8000
```

```bash
# CLI
python -m matcher.cli "بتروديس الشارة"
python -m matcher.cli "Petrodis Chara" --lat 18.10 --lon -15.96
python -m matcher.cli "بتروديس الشارة" --no-semantic --no-rerank   # fast dev mode
```

```bash
# HTTP — text mode
curl -X POST http://localhost:8000/resolve \
  -H "Content-Type: application/json" \
  -d '{"query":"بتروديس الشارة","user_lat":18.10,"user_lon":-15.96,"top_k":3}'

# HTTP — audio mode
curl -X POST http://localhost:8000/resolve \
  -F "audio=@sample.m4a" \
  -F "user_lat=18.10" -F "user_lon=-15.96" -F "top_k=3"
```

Response:

```json
{
  "matches": [
    {
      "id": 1844,
      "canonicalName": "بيترو ديس طريق الشارة",
      "lat": 18.1309611,
      "lon": -15.9617428,
      "confidence": 0.94,
      "matchedVariant": "بتروديس الشاره",
      "distanceMeters": 3420
    }
  ],
  "needsConfirmation": false,
  "confirmationPrompt": null
}
```

If the top-2 candidates are within `CONFIDENCE_MARGIN` (default 0.1) of each other, the service sets `needsConfirmation: true` and returns an Arabic prompt like `"هل تقصد X أم Y؟"`.

## Test and evaluate

```bash
make test         # unit + integration tests
make eval         # runs tests/golden_cases.json end-to-end and prints pass-rates
```

`make eval` reports:
- **Strict pass-rate**: expected id is top-1
- **Loose pass-rate**: expected id is in top-3

Iterate on these by adding cases to `tests/golden_cases.json` and tracking the rates over time. Without the LLM reranker, the heuristic baseline is around 70–75%; with the Haiku reranker enabled (`ANTHROPIC_API_KEY` set), it should be substantially higher.

## Adding new places

1. Edit `matcher/data/places.json`. Each entry needs:
   ```json
   {
     "id": <unique int>,
     "canonicalName": "...",
     "variants": ["...", "...", "..."],
     "lat": 18.x, "lon": -15.x
   }
   ```
2. Re-run `make index` to rebuild all retrievers.
3. The service must be restarted to pick up the new indexes (or hit `make dev` again — uvicorn `--reload` only reloads source code, not the indexes).

The more variants per entry, the better — include both Arabic and Latin spellings, common ASR mangles, and the version with and without prefixes like `محطة` / `Station`.

## Module layout

```
matcher/
├── config.py              env vars, model names, thresholds
├── normalize.py           Stage 2 — symmetric Arabic+Latin normalization
├── transliterate.py       Arabic → Latin phonetic projection (Hassaniya-tuned)
├── retrievers/
│   ├── lexical.py         Stage 3a — RapidFuzz token_set_ratio
│   ├── phonetic.py        Stage 3b — phonetic-space fuzzy match
│   └── semantic.py        Stage 3c — BGE-M3 + FAISS
├── asr.py                 Stage 1 — ASRBackend protocol + Whisper impl
├── rerank.py              Stage 4 — Claude Haiku (+ heuristic fallback)
├── pipeline.py            Stages 1→5 orchestrator
├── api.py                 FastAPI: POST /resolve
├── cli.py                 python -m matcher.cli "..."
├── index_build.py         python -m matcher.index_build
├── eval.py                python -m matcher.eval
└── data/
    ├── places.json
    └── index/             generated by index_build, gitignored

tests/
├── test_normalize.py
├── test_transliterate.py
├── test_retrievers.py
├── test_pipeline.py       end-to-end with golden cases
└── golden_cases.json      ~30 (query, expected_id) pairs
```

## Configuration

Environment variables (all overridable in `.env`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | _(unset)_ | Required for the LLM reranker |
| `OPENAI_API_KEY` | _(unset)_ | Required for Whisper ASR |
| `RERANKER_MODEL` | `claude-haiku-4-5` | Anthropic model id |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | sentence-transformers model id |
| `ASR_MODEL` | `whisper-1` | OpenAI Whisper model id |
| `PLACES_PATH` | `matcher/data/places.json` | Catalog source |
| `INDEX_DIR` | `matcher/data/index` | Where indexes are written/loaded |
| `CONFIDENCE_THRESHOLD` | `0.85` | Top-1 confidence required to skip confirmation |
| `CONFIDENCE_MARGIN` | `0.1` | Required gap between top-1 and top-2 |
| `LOG_FILE` | `logs/resolutions.jsonl` | JSONL of every resolution (for fine-tuning later) |

## Resolution log

Every resolution is appended to `logs/resolutions.jsonl` as a JSON line:

```json
{"ts": 1761478800.1, "query_hash": "a3f...", "transcripts": ["..."], "matches": [...], "needsConfirmation": false}
```

This is gold for:
- Fine-tuning the reranker.
- Auto-expanding the `variants` list for places that get matched via fuzzy paths.
- Spotting ASR error patterns to add to the bias prompt.

## Swapping the ASR backend

`matcher/asr.py` defines an `ASRBackend` protocol. To switch from Whisper to Deepgram/AssemblyAI/Soniox (which have first-class keyword boosting), implement a class with `transcribe(audio, prompt) -> list[str]` and pass it to `Pipeline(asr_backend=...)` (or set it on `_state["pipeline"]` in `api.py`).
