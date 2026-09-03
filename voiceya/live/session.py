"""One live session: PCM in, batch-shaped summaries out after every sentence.

The session owns a ``SentenceSegmenter`` and a single ordered worker task.
Each finished sentence goes through the same building blocks as the batch
path (Engine A, Whisper, sidecar, pyin) but on its own audio, and the
results are accumulated so ``_build_result()`` can hand the frontend the
exact shape ``/analyze-voice`` would have produced for everything heard so
far.  The frontend therefore renders live output through the code it already
has (``renderFromSummary`` + ``PhoneTimeline.setData``).
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import numpy as np

from voiceya.config import CFG
from voiceya.live.script_cursor import ScriptCursor
from voiceya.live.vad import SentenceSegmenter
from voiceya.live.wav import pcm16_to_wav
from voiceya.services.audio_analyser import f0_panel as f0mod
from voiceya.services.audio_analyser import resonance_calibration, statics
from voiceya.services.audio_analyser.advice_v2 import compute_advice
from voiceya.services.audio_analyser.engine_c import (
    _aggregate_per_vowel,
    _alignment_confidence,
    _median,
    _normalize_lang,
    analyze_with_sidecar,
)
from voiceya.services.audio_analyser.seg_analyser import AnalyseResultItem

logger = logging.getLogger(__name__)

SendFn = Callable[[dict[str, Any]], Awaitable[None]]
Language = Literal["zh-CN", "en-US", "fr-FR", "ko-KR"]
Mode = Literal["free", "script"]

MAX_SESSION_SEC = 180.0
# Per-sentence sidecar budget.  A healthy sentence takes ~1 s; anything past
# this is a stuck aligner and must not hold up the sentences behind it.
SENTENCE_SIDECAR_TIMEOUT_SEC = 15.0


class LiveSession:
    def __init__(
        self,
        *,
        language: Language,
        mode: Mode,
        script: str | None,
        send: SendFn,
        sr: int = 16000,
    ) -> None:
        self.language = language
        self.lang_short = _normalize_lang(language)
        self.mode: Mode = mode if (mode == "script" and script) else "free"
        self.script = script if self.mode == "script" else None
        self.send = send
        self.sr = sr
        self.seg = SentenceSegmenter(sr=sr)
        self.cursor = ScriptCursor(script or "", language) if self.mode == "script" else None

        self._queue: asyncio.Queue[tuple | None] = asyncio.Queue()
        self._worker = asyncio.create_task(self._run())
        self._n_queued = 0
        self._stopped = False

        # Accumulators (absolute session time).
        self.analyse_results: list[AnalyseResultItem] = []
        self.phones: list[dict[str, Any]] = []
        self.transcripts: list[str] = []
        self.sentences: list[dict[str, Any]] = []
        self.voiced_f0 = np.zeros(0, dtype=np.float64)
        self.ceilings: Counter[int] = Counter()
        self.word_count = 0

    # ── input ───────────────────────────────────────────────────────

    async def feed(self, data: bytes) -> None:
        if self._stopped:
            return
        if self.seg.elapsed_sec >= MAX_SESSION_SEC:
            await self.send(
                {"type": "error", "msg": "max session length reached", "msg_key": "live.maxLength"}
            )
            await self.stop()
            return
        pcm = np.frombuffer(data, dtype="<i2")
        for ev in self.seg.feed(pcm):
            await self._on_event(ev)

    async def stop(self) -> None:
        """Flush the segmenter, wait for queued sentences, then return."""
        if self._stopped:
            return
        self._stopped = True
        for ev in self.seg.flush():
            await self._on_event(ev)
        await self._queue.put(None)
        await self._worker

    def abort(self) -> None:
        """Client vanished: drop pending work without sending anything."""
        self._stopped = True
        self._worker.cancel()

    # ── events ──────────────────────────────────────────────────────

    async def _on_event(self, ev: tuple) -> None:
        if ev[0] == "speaking":
            await self.send({"type": "vad", "speaking": bool(ev[1]), "t": round(float(ev[2]), 2)})
            return
        _, start, end, samples = ev
        index = self._n_queued
        self._n_queued += 1
        await self.send(
            {
                "type": "sentence_queued",
                "index": index,
                "start": round(float(start), 2),
                "end": round(float(end), 2),
                "pending": self._queue.qsize() + 1,
            }
        )
        await self._queue.put((index, float(start), float(end), samples, time.monotonic()))

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            index, start, end, samples, t_queued = item
            try:
                sentence = await self._process(index, start, end, samples)
                sentence["latency_ms"] = int((time.monotonic() - t_queued) * 1000)
                self.sentences.append(sentence)
                result = self._build_result()
                result["summary"]["live"] = {
                    "sentences": len(self.sentences),
                    "elapsed_sec": round(self.seg.elapsed_sec, 2),
                    "pending": self._queue.qsize(),
                    "script_progress": self.cursor.progress if self.cursor else None,
                    "last": sentence,
                }
                await self.send(
                    {"type": "summary", "index": index, "sentence": sentence, "result": result}
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("live sentence %d failed", index)
                await self.send(
                    {"type": "sentence_failed", "index": index, "reason": str(exc)[:200]}
                )

    # ── per-sentence pipeline ───────────────────────────────────────

    async def _process(
        self, index: int, start: float, end: float, samples: np.ndarray
    ) -> dict[str, Any]:
        wav = pcm16_to_wav(samples, self.sr)
        y = samples.astype(np.float32) / 32768.0
        timings: dict[str, int] = {}
        t_stage = time.monotonic()

        asr_text, seg_items, voiced = await asyncio.gather(
            self._transcribe(wav),
            self._engine_a(wav, start),
            asyncio.to_thread(f0mod.voiced_f0_values, y, self.sr),
        )
        timings["asr_engine_a_pyin_ms"] = int((time.monotonic() - t_stage) * 1000)
        self.analyse_results.extend(seg_items)
        if voiced.size:
            self.voiced_f0 = np.concatenate([self.voiced_f0, voiced])

        sentence: dict[str, Any] = {
            "index": index,
            "start": round(start, 2),
            "end": round(end, 2),
            "asr_text": asr_text,
            "transcript": asr_text,
            "matched": None,
            "engine_c": False,
            "timings_ms": timings,
        }

        transcript = asr_text.strip()
        if self.cursor is not None and transcript:
            m = self.cursor.consume(transcript)
            transcript = m.text
            sentence["matched"] = m.matched
        sentence["transcript"] = transcript

        if not transcript or not CFG.engine_c_enabled:
            return sentence

        t_stage = time.monotonic()
        ec = await analyze_with_sidecar(
            wav,
            transcript,
            language=self.language,
            mode=self.mode,
            script=None,
            word_timestamps=None,
            total_audio_sec=end - start,
            fast_only=True,
            timeout_sec=SENTENCE_SIDECAR_TIMEOUT_SEC,
        )
        timings["sidecar_ms"] = int((time.monotonic() - t_stage) * 1000)
        if not ec:
            return sentence

        for p in ec["phones"]:
            p["start"] = round(p["start"] + start, 3)
            p["end"] = round(p["end"] + start, 3)
        self.phones.extend(ec["phones"])
        self.transcripts.append(transcript)
        self.word_count += int(ec.get("word_count") or 0)
        if ec.get("formant_ceiling_hz"):
            self.ceilings[int(ec["formant_ceiling_hz"])] += 1
        sentence["engine_c"] = True
        sentence["phone_count"] = len(ec["phones"])
        return sentence

    async def _transcribe(self, wav: bytes) -> str:
        if not CFG.engine_c_enabled:
            return ""
        try:
            if self.lang_short == "en":
                from voiceya.services.audio_analyser.engine_c_asr_en import (
                    transcribe_en,  # noqa: PLC0415
                )

                text, _ = await transcribe_en(wav)
            elif self.lang_short == "fr":
                from voiceya.services.audio_analyser.engine_c_asr_fr import (
                    transcribe_fr,  # noqa: PLC0415
                )

                text, _ = await transcribe_fr(wav)
            elif self.lang_short == "ko":
                from voiceya.services.audio_analyser.engine_c_asr_ko import (
                    transcribe_ko,  # noqa: PLC0415
                )

                text, _ = await transcribe_ko(wav)
            else:
                from voiceya.services.audio_analyser.engine_c_asr import (
                    transcribe_zh,  # noqa: PLC0415
                )

                text = await transcribe_zh(wav)
        except Exception as exc:
            logger.warning("live ASR failed: %s", exc)
            return ""
        return (text or "").strip()

    async def _engine_a(self, wav: bytes, offset: float) -> list[AnalyseResultItem]:
        from voiceya.services.audio_analyser import engine_a  # noqa: PLC0415

        try:
            raw = await engine_a.do_segmentation(io.BytesIO(wav))
        except Exception as exc:
            logger.warning("live Engine A failed: %s", exc)
            return []
        out: list[AnalyseResultItem] = []
        for item in raw:
            label, s, e = item[0], float(item[1]), float(item[2])
            r = AnalyseResultItem(
                label=label,
                start_time=round(s + offset, 2),
                end_time=round(e + offset, 2),
                duration=round(e - s, 2),
                confidence=round(item[3], 4) if len(item) > 3 and item[3] is not None else None,
                confidence_frames=item[4] if len(item) > 4 else None,
            )
            if r.label not in ("female", "male") and r.duration < 0.5:
                continue
            out.append(r)
        return out

    # ── aggregate → batch-shaped result ─────────────────────────────

    def _engine_c_summary(self) -> dict[str, Any] | None:
        if not self.phones:
            return None
        phones = self.phones
        lang = self.lang_short
        pitches = [float(p["pitch"]) for p in phones if p.get("pitch") is not None]
        resos = [float(p["resonance"]) for p in phones if p.get("resonance") is not None]
        per_vowel = _aggregate_per_vowel(phones, lang)
        median_res = _median(resos)
        qualifying = [
            float(v["resonance_med"])
            for v in per_vowel
            if (v.get("n") or 0) >= 5 and isinstance(v.get("resonance_med"), (int, float))
        ]
        zone_median = _median(qualifying) if qualifying else median_res

        joiner = "" if lang == "zh" else " "
        transcript = joiner.join(self.transcripts)
        speech_sec = sum(s["end"] - s["start"] for s in self.sentences if s.get("engine_c"))

        silence_ranges = []
        done = [s for s in self.sentences if s.get("engine_c")]
        for a, b in zip(done, done[1:], strict=False):
            if b["start"] > a["end"]:
                silence_ranges.append({"start": a["end"], "end": b["start"]})

        def _mean(v: list[float]) -> float | None:
            return float(np.mean(v)) if v else None

        def _std(v: list[float]) -> float | None:
            return float(np.std(v)) if len(v) > 1 else None

        return {
            "mean_pitch_hz": _mean(pitches),
            "median_pitch_hz": _median(pitches),
            "stdev_pitch_hz": _std(pitches),
            "mean_resonance": _mean(resos),
            "median_resonance": median_res,
            "stdev_resonance": _std(resos),
            "phone_count": len(phones),
            "word_count": self.word_count,
            "transcript": transcript,
            "phones": phones,
            "silence_ranges": silence_ranges,
            "mode": self.mode,
            "script": self.script,
            "language": self.language,
            "alignment_confidence": _alignment_confidence(phones, transcript, speech_sec, lang),
            "formant_ceiling_hz": self.ceilings.most_common(1)[0][0] if self.ceilings else None,
            "resonance_zone_key": resonance_calibration.classify_zone(zone_median, self.language),
            "resonance_per_vowel": per_vowel,
        }

    def _build_result(self) -> dict[str, Any]:
        duration = float(self.seg.elapsed_sec)
        f0_panel = f0mod.panel_from_voiced_f0(self.voiced_f0, self.sr, duration)
        engine_c = self._engine_c_summary()
        if engine_c:
            f0mod.prefer_praat_median(f0_panel, engine_c.get("median_pitch_hz"))

        result = statics.do_statics(self.analyse_results, f0_median_hz=f0_panel.get("median_hz"))
        summary = result["summary"]
        summary["engine_c"] = engine_c
        summary["advice"] = compute_advice(
            np.zeros(0, dtype=np.float32),
            self.sr,
            self.analyse_results,
            duration,
            summary.get("dominant_label"),
            weighted_margin=summary.get("dominant_confidence", 0.0),
            f0_panel=f0_panel,
            engine_c=engine_c,
        )
        result["filename"] = "live"
        return result
