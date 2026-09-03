"""Tiny WAV writer for int16 mono PCM (no ffmpeg round-trip)."""

from __future__ import annotations

import io
import wave

import numpy as np


def pcm16_to_wav(samples: np.ndarray, sr: int = 16000) -> bytes:
    if samples.dtype != np.int16:
        samples = samples.astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return buf.getvalue()
