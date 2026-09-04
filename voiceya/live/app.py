"""FastAPI app for the live worker: ``/api/live`` WebSocket + healthz.

Runs in its own process (see ``voiceya/live/__main__.py``).  Loads Engine A
at startup and warms Whisper in the background so the first sentence does
not pay model-load time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import voiceya  # noqa: F401  — config load + submodule patches happen on import
from voiceya.config import CFG
from voiceya.live.realtime import PHONE_MODEL, RealtimeSession, ensure_phone_model
from voiceya.live.session import LiveSession
from voiceya.live.wav import pcm16_to_wav
from voiceya.services.audio_analyser.seg import load_seg

logger = logging.getLogger("voiceya.live")

_ALLOWED_LANGS = ("zh-CN", "en-US", "fr-FR", "ko-KR")
_WARM_LANGS = {"en-US"}


async def _warm_whisper() -> None:
    if not CFG.engine_c_enabled:
        return
    try:
        from voiceya.services.audio_analyser.engine_c_asr_en import transcribe_en  # noqa: PLC0415

        silence = pcm16_to_wav(np.zeros(16000, dtype=np.int16))
        await transcribe_en(silence)
        logger.info("live: Whisper (en) warm")
    except Exception as exc:  # pragma: no cover — warmup is best-effort
        logger.warning("live: Whisper warmup skipped: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await load_seg()
    warm = asyncio.create_task(_warm_whisper())
    # Experimental frame-level mode: load the phone classifier in the
    # background; sessions that ask for realtime before it is ready wait.
    phone = asyncio.create_task(ensure_phone_model())
    yield
    warm.cancel()
    phone.cancel()


app = FastAPI(title=f"{CFG.app_name} live", lifespan=lifespan)


@app.get("/api/live/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "engine_c": CFG.engine_c_enabled,
        "languages": list(_ALLOWED_LANGS),
        "realtime": {
            "phone_model": PHONE_MODEL.model is not None,
            "error": PHONE_MODEL.error,
            "languages": ["en-US"],
        },
    }


@app.websocket("/api/live")
async def live(ws: WebSocket) -> None:
    await ws.accept()
    session: LiveSession | RealtimeSession | None = None
    closed = False

    async def send(payload: dict[str, Any]) -> None:
        nonlocal closed
        if closed:
            return
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            closed = True

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is not None:
                if session is not None:
                    await session.feed(data)
                continue
            text = msg.get("text")
            if not text:
                continue
            try:
                cmd = json.loads(text)
            except json.JSONDecodeError:
                await send({"type": "error", "msg": "bad json"})
                continue
            kind = cmd.get("type")
            if kind == "start":
                if session is not None:
                    await send({"type": "error", "msg": "session already started"})
                    continue
                language = cmd.get("language") if cmd.get("language") in _ALLOWED_LANGS else "zh-CN"
                mode = "script" if cmd.get("mode") == "script" else "free"
                script = (cmd.get("script") or "").strip() or None
                if cmd.get("realtime"):
                    await ensure_phone_model()
                    session = RealtimeSession(
                        language=language, mode=mode, script=script, send=send
                    )
                    logger.info(
                        "realtime session start lang=%s phones=%s", language, session.phones_enabled
                    )
                    await send(
                        {
                            "type": "ready",
                            "language": language,
                            "mode": mode,
                            "realtime": True,
                            "phones_enabled": session.phones_enabled,
                            "engine_c": CFG.engine_c_enabled,
                        }
                    )
                    continue
                session = LiveSession(language=language, mode=mode, script=script, send=send)
                logger.info("live session start lang=%s mode=%s", language, session.mode)
                await send(
                    {
                        "type": "ready",
                        "language": language,
                        "mode": session.mode,
                        "engine_c": CFG.engine_c_enabled,
                    }
                )
            elif kind == "stop":
                if session is not None:
                    await session.stop()
                    n = len(getattr(session, "sentences", None) or getattr(session, "phones", []))
                    await send({"type": "done", "sentences": n})
                    logger.info("live session done items=%d", n)
                    session = None
                break
    except WebSocketDisconnect:
        pass
    finally:
        closed = True
        if session is not None:
            session.abort()
        try:
            await ws.close()
        except Exception:
            pass
