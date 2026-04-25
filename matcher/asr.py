"""Stage 1 — ASR.

ASRBackend is a Protocol so we can swap Whisper for Deepgram, AssemblyAI, or
Soniox without touching the rest of the pipeline. Each backend takes audio
bytes + an optional bias-prompt and returns a list of N-best transcripts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Protocol

from .config import settings


class ASRBackend(Protocol):
    def transcribe(
        self,
        audio: BinaryIO,
        prompt: str | None = None,
        filename: str | None = None,
    ) -> list[str]:
        """Return up to N best transcripts, most likely first."""
        ...


def build_bias_prompt(places_path: Path, top_k: int = 50) -> str:
    """Build a comma-separated string of the most common canonical names.

    "Most common" here is by length-truncation of the catalog — we just take
    the first `top_k` canonicals. A frequency signal would require usage logs
    we don't have yet.
    """
    with open(places_path, "r", encoding="utf-8") as f:
        places = json.load(f)
    names: list[str] = []
    seen: set[str] = set()
    for p in places:
        name = p.get("canonicalName")
        if name and name not in seen:
            names.append(name)
            seen.add(name)
        if len(names) >= top_k:
            break
    return ", ".join(names)


class WhisperBackend:
    """OpenAI Whisper via the openai SDK."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.asr_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            if not settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to .env to use the Whisper backend."
                )
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def transcribe(
        self,
        audio: BinaryIO,
        prompt: str | None = None,
        filename: str | None = None,
    ) -> list[str]:
        client = self._get_client()
        # OpenAI's SDK uses the filename's extension to detect the format.
        # Fall back to .m4a since that's what Expo's recorder produces by default.
        name = filename or "audio.m4a"
        audio_bytes = audio.read()
        audio.seek(0)

        # OpenAI Whisper returns one transcript per call. Until first-class
        # N-best lands in the API we approximate by sampling at one or two temps.
        candidates: list[str] = []
        for temp in (0.0, 0.4):
            resp = client.audio.transcriptions.create(
                model=self.model,
                file=(name, audio_bytes),
                language="ar",
                prompt=prompt or "",
                temperature=temp,
            )
            text = getattr(resp, "text", None) or str(resp)
            if text and text not in candidates:
                candidates.append(text)
        # Order by frequency (a transcript that came back twice is more likely).
        counts = Counter(candidates)
        return [t for t, _ in counts.most_common()]


class StaticBackend:
    """Test/CLI backend that returns whatever transcripts it was given."""

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = transcripts

    def transcribe(
        self,
        audio: BinaryIO,
        prompt: str | None = None,
        filename: str | None = None,
    ) -> list[str]:
        return list(self._transcripts)


def default_backend() -> ASRBackend:
    return WhisperBackend()
