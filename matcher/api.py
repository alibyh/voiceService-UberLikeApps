"""FastAPI entry point.

- POST /resolve         — JSON body with a text query
- POST /resolve/audio   — multipart upload with an audio file

Splitting the two avoids a FastAPI quirk: when a single endpoint mixes Form()
parameters with a Pydantic body, the framework treats every request as
multipart and silently drops JSON payloads.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .asr import build_bias_prompt
from .config import settings
from .pipeline import Pipeline


class TextResolveRequest(BaseModel):
    query: str
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None
    top_k: int = 3


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = Pipeline.load()
    pipeline.bias_prompt = build_bias_prompt(settings.places_path)
    _state["pipeline"] = pipeline
    yield
    _state.clear()


app = FastAPI(title="Hassaniya Voice Place Matcher", lifespan=lifespan)


def _serialize(resp) -> dict:
    return {
        "matches": [asdict(m) for m in resp.matches],
        "needsConfirmation": resp.needsConfirmation,
        "confirmationPrompt": resp.confirmationPrompt,
        "debug": resp.debug,
    }


def _get_pipeline() -> Pipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    return pipeline


@app.get("/health")
def health() -> dict:
    pipeline = _state.get("pipeline")
    return {
        "ok": pipeline is not None,
        "places": len(pipeline.places) if pipeline else 0,
        "semantic": pipeline.semantic is not None if pipeline else False,
    }


@app.post("/resolve")
async def resolve_text(payload: TextResolveRequest) -> dict:
    pipeline = _get_pipeline()
    resp = await pipeline.resolve(
        payload.query,
        user_lat=payload.user_lat,
        user_lon=payload.user_lon,
        top_k=payload.top_k,
    )
    return _serialize(resp)


@app.post("/resolve/audio")
async def resolve_audio(
    audio: UploadFile = File(...),
    user_lat: Optional[float] = Form(default=None),
    user_lon: Optional[float] = Form(default=None),
    top_k: int = Form(default=3),
) -> dict:
    pipeline = _get_pipeline()
    audio_bytes = await audio.read()
    transcripts = pipeline.transcribe(io.BytesIO(audio_bytes))
    if not transcripts:
        raise HTTPException(status_code=422, detail="no transcripts produced")
    primary, *extras = transcripts
    resp = await pipeline.resolve(
        primary,
        user_lat=user_lat,
        user_lon=user_lon,
        top_k=top_k,
        extra_queries=extras,
    )
    return _serialize(resp)
