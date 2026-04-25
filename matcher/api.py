"""FastAPI entry point.

- POST /resolve         — JSON body with a text query
- POST /resolve/audio   — multipart upload with an audio file

Splitting the two avoids a FastAPI quirk: when a single endpoint mixes Form()
parameters with a Pydantic body, the framework treats every request as
multipart and silently drops JSON payloads.
"""

from __future__ import annotations

import io
import logging
import traceback
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .asr import build_bias_prompt
from .config import settings
from .pipeline import Pipeline


logger = logging.getLogger("matcher.api")
logging.basicConfig(level=logging.INFO)


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


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Surface real errors instead of bare 500s.

    The traceback is written to the container log so it shows up in Railway,
    and the response carries the error class + message so the mobile client
    can display something useful while debugging.
    """
    logger.error("unhandled exception on %s %s\n%s", request.method, request.url.path, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "detail": str(exc),
            "path": request.url.path,
        },
    )


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
    logger.info(
        "resolve_audio: filename=%s content_type=%s bytes=%d top_k=%s gps=(%s,%s)",
        audio.filename,
        audio.content_type,
        len(audio_bytes),
        top_k,
        user_lat,
        user_lon,
    )
    try:
        transcripts = pipeline.transcribe(
            io.BytesIO(audio_bytes), filename=audio.filename
        )
    except Exception as exc:  # surface ASR failures with detail
        logger.error("ASR failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"ASR failed: {exc}") from exc

    logger.info("resolve_audio: transcripts=%s", transcripts)
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
