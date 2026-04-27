"""Stage 1 — ASR.

ASRBackend is a Protocol so we can swap Whisper for Deepgram, AssemblyAI, or
Soniox without touching the rest of the pipeline. Each backend takes audio
bytes + an optional bias-prompt and returns a list of N-best transcripts.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Protocol

from .config import settings


# Whisper emits these phrases when given silent/unclear audio — they are
# hallucinations from YouTube training data, not transcriptions. Drop them.
_WHISPER_HALLUCINATIONS = {
    "اشتركوا في القناة",
    "شكرا على المشاهدة",
    "شكرا لكم على المشاهدة",
    "شكرا لمشاهدتكم",
    "ترجمة نانسي قنقر",
    "subscribe to the channel",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "merci d'avoir regardé",
    "abonnez-vous",
}


def _is_hallucination(text: str) -> bool:
    folded = text.strip().lower()
    return any(h in folded for h in _WHISPER_HALLUCINATIONS)


def _collapse_repetition(text: str) -> str:
    """Whisper loops sometimes emit 'X, X, X, X, ...' for noisy audio.

    If a comma-delimited transcript repeats the same fragment 3+ times in a
    row, keep only the first occurrence — that's the most likely real signal.
    """
    parts = [p.strip() for p in re.split(r"[,،]", text) if p.strip()]
    if len(parts) < 3:
        return text
    counts = Counter(parts)
    most_common, n = counts.most_common(1)[0]
    if n >= 3 and n / len(parts) >= 0.5:
        return most_common
    return text


class ASRBackend(Protocol):
    def transcribe(
        self,
        audio: BinaryIO,
        prompt: str | None = None,
        filename: str | None = None,
    ) -> list[str]:
        """Return up to N best transcripts, most likely first."""
        ...


def build_bias_prompt(places_path: Path, top_k: int = 80) -> str:
    """Build a context+vocabulary prompt to bias Whisper.

    Whisper's `prompt=` is capped at ~244 tokens. We use that budget to:
    1. Set dialect context ("Hassaniya Arabic in Mauritania, places in Nouakchott").
    2. List the most common canonical names so the decoder has them in scope.

    This is soft conditioning — it nudges Whisper toward our vocabulary but
    doesn't force it. For hard keyword boosting, switch to Deepgram or Soniox.
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
    preamble = (
        "أسماء أماكن في نواكشوط، موريتانيا. قد يخلط المتحدث الحسانية والعربية والفرنسية. "
        "Place names in Nouakchott, Mauritania. The speaker may mix Hassaniya Arabic, French, and English."
    )
    return preamble + " " + ", ".join(names)


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
        # We use response_format="verbose_json" to access per-segment metadata
        # (no_speech_prob, avg_logprob) so we can drop low-confidence guesses.
        candidates: list[str] = []
        for temp in (0.0, 0.4):
            resp = client.audio.transcriptions.create(
                model=self.model,
                file=(name, audio_bytes),
                language="ar",
                prompt=prompt or "",
                temperature=temp,
                response_format="verbose_json",
            )
            text = getattr(resp, "text", None)
            if not text:
                continue
            # Reject transcripts where the decoder itself is unsure. Thresholds
            # are tuned to drop hallucinations while keeping noisy-but-real speech.
            segments = getattr(resp, "segments", None) or []

            def _seg_field(seg, key, default):
                if isinstance(seg, dict):
                    return seg.get(key, default)
                return getattr(seg, key, default)

            if segments:
                no_speech = max(
                    (_seg_field(s, "no_speech_prob", 0.0) for s in segments), default=0.0
                )
                avg_logprob = min(
                    (_seg_field(s, "avg_logprob", 0.0) for s in segments), default=0.0
                )
                if no_speech > 0.6 or avg_logprob < -1.0:
                    continue
            text = _collapse_repetition(text)
            if _is_hallucination(text):
                continue
            if text not in candidates:
                candidates.append(text)
        # Order by frequency (a transcript that came back twice is more likely).
        counts = Counter(candidates)
        return [t for t, _ in counts.most_common()]


def build_keyword_list(places_path: Path, max_keywords: int = 200) -> list[str]:
    """Build a `keywords` list for Deepgram from the catalog.

    Deepgram's keyword boosting is first-class — each keyword can be passed
    with a `:boost` suffix (1–10). We use 3 by default, which is enough to
    nudge the decoder toward our vocabulary without over-fitting.

    Cap the list at Deepgram's documented max of 200. If the catalog is
    bigger, sample evenly across the full id range so late-added places
    aren't systematically excluded — taking only the first N would silently
    drop everything past id ~200.
    """
    with open(places_path, "r", encoding="utf-8") as f:
        places = json.load(f)

    canonicals: list[str] = []
    seen: set[str] = set()
    for p in places:
        name = (p.get("canonicalName") or "").strip()
        if name and name not in seen:
            seen.add(name)
            canonicals.append(name)

    if len(canonicals) <= max_keywords:
        sampled = canonicals
    else:
        # Even stride sampling — preserves the first and last entries and
        # spreads coverage across the catalog.
        stride = len(canonicals) / max_keywords
        sampled = [canonicals[int(i * stride)] for i in range(max_keywords)]

    return [f"{name}:3" for name in sampled]


class DeepgramBackend:
    """Deepgram Nova with first-class keyword boosting.

    This is the structurally correct ASR for our problem: keyword boosting
    is hard conditioning (decoder-level), not soft prompting like Whisper.
    Pass each place name as a boosted keyword and the model is much more
    likely to hear "بتروديس" instead of falling back to a more common phrase.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.deepgram_model
        self._client = None
        self._keywords: list[str] | None = None

    def _get_client(self):
        if self._client is None:
            from deepgram import DeepgramClient

            if not settings.deepgram_api_key:
                raise RuntimeError(
                    "DEEPGRAM_API_KEY is not set. Add it to your env to use the Deepgram backend."
                )
            # Newer deepgram-sdk versions require api_key as a keyword arg.
            self._client = DeepgramClient(api_key=settings.deepgram_api_key)
        return self._client

    def _get_keywords(self) -> list[str]:
        if self._keywords is None:
            self._keywords = build_keyword_list(settings.places_path)
        return self._keywords

    def transcribe(
        self,
        audio: BinaryIO,
        prompt: str | None = None,
        filename: str | None = None,
    ) -> list[str]:
        client = self._get_client()
        audio_bytes = audio.read()
        audio.seek(0)

        # Use a plain dict for options instead of the typed PrerecordedOptions
        # class — the class lives at different import paths across SDK versions
        # (deepgram-sdk v3 vs v4). The dict form is stable.
        options = {
            "model": self.model,
            "language": "ar",
            "keywords": self._get_keywords(),
            "smart_format": True,
            "punctuate": True,
            "alternatives": 3,  # native N-best — no temperature loop needed
        }

        payload = {"buffer": audio_bytes}
        response = client.listen.rest.v("1").transcribe_file(payload, options)

        # Walk results.channels[0].alternatives → list of {transcript, confidence, ...}
        try:
            channels = response.results.channels
            alts = channels[0].alternatives if channels else []
        except (AttributeError, IndexError):
            return []

        candidates: list[str] = []
        for alt in alts:
            text = (getattr(alt, "transcript", "") or "").strip()
            if not text:
                continue
            confidence = float(getattr(alt, "confidence", 1.0) or 0.0)
            if confidence < 0.3:
                continue
            text = _collapse_repetition(text)
            if _is_hallucination(text):
                continue
            if text not in candidates:
                candidates.append(text)
        return candidates


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
    """Pick the ASR backend based on ASR_BACKEND env var.

    Falls back to Whisper if "deepgram" is requested but no key is set.
    """
    choice = settings.asr_backend
    if choice == "deepgram":
        if not settings.deepgram_api_key:
            print("[asr] ASR_BACKEND=deepgram but DEEPGRAM_API_KEY unset; falling back to Whisper")
            return WhisperBackend()
        return DeepgramBackend()
    return WhisperBackend()
