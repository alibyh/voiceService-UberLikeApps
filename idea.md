# Hassaniya Voice Place-Matching Service — Build Spec

## Context

We are building a voice-driven place lookup service for ride-hailing apps (Uber-like) in Mauritania. A user speaks the name of a place in **Hassaniya Arabic** (the Mauritanian dialect, with frequent French/English code-switching). The system transcribes the audio, matches it against our place database despite ASR errors and dialectal variation, and returns the canonical place name + GPS coordinates.

The hard part is **not** speech-to-text. It is robust matching of a noisy, dialectal, multi-script transcription against a curated place catalog. ASR will mangle Hassaniya pronunciations; users will mix Arabic and Latin script ("Petrodis Chara"); the same place has many spellings ("بيترو ديس" / "بتروديس" / "Petrodis"). We solve this with a **hybrid retrieval + LLM rerank** pipeline.

## Data we already have

A JSON file `places.json` containing entries shaped like:

```json
{
  "id": 1844,
  "canonicalName": "بيترو ديس طريق الشارة",
  "variants": [
    "بيترو ديس طريق الشارة",
    "بتروديس طريق الشارة",
    "Petrodis Route Al Shara",
    "Petrodis Chara",
    "محطة بيتروديس طريق الشارة",
    "Station Petrodis Route Al Chara",
    "ستاصيون بتروديس طريق الشارة"
  ],
  "lat": 18.1309611,
  "lon": -15.9617428
}
```

Assume thousands of places, each with 5–30 variants in mixed Arabic/Latin script.

## What to build

A Python service exposing two interfaces:

1. A **FastAPI HTTP endpoint** `POST /resolve` that accepts either an audio file (multipart) or a text transcription (JSON), plus optional user GPS coordinates, and returns the best-matching place(s).
2. A **CLI** `python -m matcher.cli "بتروديس الشارة"` for testing without audio.

### Request / response contract

**Request (text mode):**
```json
{
  "query": "بتروديس الشارة",
  "user_lat": 18.10,
  "user_lon": -15.96,
  "top_k": 3
}
```

**Response:**
```json
{
  "matches": [
    {
      "id": 1844,
      "canonicalName": "بيترو ديس طريق الشارة",
      "lat": 18.1309611,
      "lon": -15.9617428,
      "confidence": 0.94,
      "matchedVariant": "بتروديس الشارة",
      "distanceMeters": 3420
    }
  ],
  "needsConfirmation": false,
  "confirmationPrompt": null
}
```

If the top-2 candidates are within 0.1 confidence of each other, set `needsConfirmation: true` and produce an Arabic prompt like `"هل تقصد X أم Y؟"`.

## Architecture (5-stage pipeline)

```
audio ──► [1] ASR (with biasing + N-best)
            │
            ▼
         transcripts[]
            │
            ▼
         [2] Normalizer ──► normalized query
            │
            ▼
         [3] Hybrid retrieval (parallel)
              ├─ Lexical (RapidFuzz)
              ├─ Phonetic (custom AR→Latin transliterator + Double Metaphone)
              └─ Semantic (BGE-M3 embeddings + FAISS)
            │
            ▼
         candidates[] (top ~15, deduplicated)
            │
            ▼
         [4] LLM reranker (Claude Haiku) ──► scored & ordered
            │
            ▼
         [5] Confidence gate ──► answer OR confirmation prompt
```

## Module / file layout

```
matcher/
├── __init__.py
├── config.py              # env vars, model names, thresholds
├── data/
│   └── places.json        # input catalog
├── normalize.py           # Stage 2
├── transliterate.py       # Arabic → Latin phonetic
├── retrievers/
│   ├── __init__.py
│   ├── lexical.py         # RapidFuzz over normalized strings
│   ├── phonetic.py        # phonetic-space fuzzy
│   └── semantic.py        # BGE-M3 + FAISS
├── rerank.py              # Stage 4 — Claude Haiku
├── asr.py                 # Stage 1 — Whisper or Deepgram wrapper
├── pipeline.py            # orchestrates 1→5
├── api.py                 # FastAPI app
├── cli.py                 # CLI entry point
└── index_build.py         # one-shot script: build FAISS index + phonetic index from places.json

tests/
├── test_normalize.py
├── test_transliterate.py
├── test_retrievers.py
├── test_pipeline.py       # end-to-end with golden cases
└── golden_cases.json      # ~30 hand-written (query, expected_id) pairs
```

## Stage-by-stage implementation notes

### Stage 1 — ASR

- Default to **OpenAI Whisper** (`whisper-large-v3`) via the `openai` Python SDK, language hint `"ar"`.
- Pass `prompt=` containing a comma-separated sample of the most common ~50 place names from the catalog (read from `places.json` at startup) to bias the decoder toward our vocabulary.
- Request **N-best transcriptions** (top 3) when the provider supports it; pass each one through the rest of the pipeline and union the candidates before reranking.
- Make the ASR backend swappable behind an `ASRBackend` protocol so we can later switch to Deepgram, AssemblyAI, or Soniox (which have first-class keyword boosting).

### Stage 2 — Normalizer

Build `normalize(text: str) -> str` that applies the following in order. **The exact same normalizer must be applied to both the query AND every variant at index time** — symmetry is the whole point.

Arabic side (use `pyarabic`):
- Strip diacritics (tashkeel) and tatweel `ـ`.
- Unify alef variants: `أ إ آ` → `ا`.
- `ى` → `ي`, `ة` → `ه`.
- Remove Arabic punctuation, normalize whitespace.

Latin side:
- Lowercase, NFKD-normalize, strip combining accents.
- Collapse whitespace.

Cross-script:
- Maintain a list of "stop-prefixes" that add no discriminative value: `محطة`, `ستاصيون`, `Station`, `Gas Station`, `Route`, `طريق`. **Do not delete them**; instead emit two normalized forms per string — one with, one without — and index both. Deletion loses information when a place's canonical name actually contains "Station".

### Stage 3a — Lexical retriever

- Use `rapidfuzz`, specifically `process.extract` with `scorer=fuzz.token_set_ratio`. Token-set handles word reorderings ("Petrodis Al Shara" vs "Al Shara Petrodis") and partial overlaps cleanly.
- Index: a flat list of `(normalized_variant, place_id)` tuples.
- Return top 10 with their scores.

### Stage 3b — Phonetic retriever

This is the most important retriever for dialectal mismatch and the one Claude Code is most likely to under-build. **Do not skip it.**

Build `arabic_to_latin_phonetic(text: str) -> str` — a deterministic transliterator that maps both Arabic-script and Latin-script inputs into a single shared phonetic space. The goal is that "الشارة", "Chara", "Shara", and "Sharah" all collapse to the same or near-same string.

Minimum mapping table for Arabic letters (Hassaniya pronunciation, not MSA):
```
ا → a       ب → b       ت → t       ث → s       ج → j (Hassaniya: often g)
ح → h       خ → kh      د → d       ذ → z       ر → r
ز → z       س → s       ش → sh      ص → s       ض → d
ط → t       ظ → z       ع → '       غ → gh      ف → f
ق → g       ك → k       ل → l       م → m       ن → n
ه → h       و → w/u     ي → y/i     ة → h       ء → '
```

For Latin input, also normalize French/English digraphs into the same space:
- `ch` → `sh`, `ou` → `u`, `ai` → `e`, `ph` → `f`, double letters collapse.
- Strip leading `al-` / `el-` / `ال` (the definite article) since it's noise.

Then pipe the result through `jellyfish.metaphone` (or Double Metaphone via `doublemetaphone`) for one final phonetic squeeze.

Match in this space using RapidFuzz again.

### Stage 3c — Semantic retriever

- Embed every variant at index-build time with **`BAAI/bge-m3`** (multilingual, handles Arabic + French + English well, ~1GB model).
- Store vectors in **FAISS** (`IndexFlatIP` with normalized vectors is fine for this scale).
- At query time, embed the query, take top 10 by cosine similarity.
- Cache embeddings to disk so we don't re-embed on every restart.

### Stage 3d — Candidate fusion

Take the union of top-10 from each retriever (~30 raw, dedupe by `place_id`, cap at 15). Pass to the reranker.

### Stage 4 — LLM reranker

Use **Claude Haiku** via the Anthropic API (cheap, fast, ~200ms). Prompt template:

```
You are matching a noisy voice transcription of a place name in Mauritania
to a database. The user's speech may be in Hassaniya Arabic, French, or
English, and the transcription may contain ASR errors.

User said: "{query}"
User's current location: {user_lat}, {user_lon}  (use distance as a tiebreaker)

Candidates:
1. id={id} | canonical="{canonicalName}" | variants=[{top_3_variants}] | distance={meters}m
2. ...

Return JSON only:
{
  "ranked": [
    {"id": <int>, "confidence": <float 0-1>, "reason": "<short>"},
    ...
  ]
}
```

Use the API in JSON mode / with a tool definition to guarantee parseable output. Set `max_tokens=512`, `temperature=0`.

### Stage 5 — Confidence gate

- If `top.confidence >= 0.85` and `top.confidence - second.confidence >= 0.1` → return single match, `needsConfirmation: false`.
- Otherwise → return top 2–3 with `needsConfirmation: true` and an Arabic confirmation prompt.

## Key decisions / non-negotiables

- **Symmetry of normalization.** Whatever transformation is applied to the query must be applied identically to every variant at index time. Most matching bugs come from violating this.
- **Don't delete information, add it.** When in doubt about whether to strip a token, index both forms.
- **Indexes are built once.** `index_build.py` is a separate script that produces `index/lexical.pkl`, `index/phonetic.pkl`, `index/faiss.idx`, `index/embeddings.npy`, `index/metadata.json`. The API loads these at startup. Do not rebuild on every request.
- **All four retrievers run in parallel** using `asyncio.gather` or a thread pool — they have very different latency profiles (FAISS ~5ms, RapidFuzz ~20ms, embedding the query ~50ms).
- **Log everything** — every `(audio_hash, transcripts, candidates, chosen_id, user_correction)` tuple to a JSONL file. This dataset is gold for future fine-tuning and for auto-expanding the variants list.

## Tech stack

- Python 3.11+
- `fastapi`, `uvicorn`
- `rapidfuzz`
- `pyarabic`
- `jellyfish` (metaphone)
- `sentence-transformers` (for BGE-M3) + `faiss-cpu`
- `anthropic` (reranker)
- `openai` (Whisper) — keep behind a protocol so it's swappable
- `pytest` for tests
- `python-dotenv` for config

## Configuration (`.env`)

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
RERANKER_MODEL=claude-haiku-4-5
EMBEDDING_MODEL=BAAI/bge-m3
ASR_MODEL=whisper-large-v3
PLACES_PATH=matcher/data/places.json
INDEX_DIR=matcher/data/index
CONFIDENCE_THRESHOLD=0.85
CONFIDENCE_MARGIN=0.1
```

## Testing

In `tests/golden_cases.json`, write ~30 hand-crafted cases covering the failure modes we care about:

```json
[
  {"query": "بتروديس الشارة", "expected_id": 1844, "note": "missing yaa"},
  {"query": "Petrodis Chara", "expected_id": 1844, "note": "French sh→ch"},
  {"query": "ستاسيون بترودس", "expected_id": 1844, "note": "ASR drops final word + sad→sin"},
  {"query": "petro dis route shara", "expected_id": 1844, "note": "split + English"},
  ...
]
```

`tests/test_pipeline.py` runs every case end-to-end and asserts that the expected ID is in the top 1 (strict) or top 3 (loose). Track strict-pass-rate and loose-pass-rate as your two key quality metrics.

Add a `make eval` target that prints these rates plus the full failure list — this is what we'll iterate on.

## What NOT to do

- Do not try to write a single mega-regex that "handles all the cases." The pipeline above is the whole point.
- Do not skip the phonetic retriever because "the embedding model probably handles it." It does not — embeddings cluster by meaning, not by sound, and ASR errors are sound-shaped.
- Do not call the reranker on raw retrieval-candidate dumps without scores or distance — it needs that context to make good decisions.
- Do not normalize variants at query time. Build the index once.
- Do not couple the pipeline to a specific ASR vendor; hide it behind a protocol.

## Deliverables

1. Complete repo with the structure above.
2. `index_build.py` runnable as `python -m matcher.index_build`.
3. `make dev` to start the FastAPI server.
4. `make eval` to run the golden test suite and print pass-rate.
5. A short `README.md` covering setup, the architecture, and how to add new places.