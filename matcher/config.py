"""Centralized configuration. Loads .env once at import time."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(env_key: str, default: str) -> Path:
    raw = os.getenv(env_key, default)
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    openai_api_key: str | None
    deepgram_api_key: str | None

    reranker_model: str
    embedding_model: str
    asr_model: str
    asr_backend: str  # "whisper" | "deepgram"
    deepgram_model: str

    places_path: Path
    index_dir: Path
    log_file: Path

    confidence_threshold: float
    confidence_margin: float


settings = Settings(
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
    openai_api_key=os.getenv("OPENAI_API_KEY") or None,
    deepgram_api_key=os.getenv("DEEPGRAM_API_KEY") or None,
    reranker_model=os.getenv("RERANKER_MODEL", "claude-haiku-4-5"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
    asr_model=os.getenv("ASR_MODEL", "whisper-1"),
    asr_backend=os.getenv("ASR_BACKEND", "whisper").lower(),
    deepgram_model=os.getenv("DEEPGRAM_MODEL", "nova-2-general"),
    places_path=_path("PLACES_PATH", "matcher/data/places.json"),
    index_dir=_path("INDEX_DIR", "matcher/data/index"),
    log_file=_path("LOG_FILE", "logs/resolutions.jsonl"),
    confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.85")),
    confidence_margin=float(os.getenv("CONFIDENCE_MARGIN", "0.1")),
)


# Index file names — kept here so the builder and loaders agree.
LEXICAL_INDEX_FILE = "lexical.pkl"
PHONETIC_INDEX_FILE = "phonetic.pkl"
FAISS_INDEX_FILE = "faiss.idx"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"
