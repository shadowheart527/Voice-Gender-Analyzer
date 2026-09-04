"""Frame-level real-time analysis (experimental).

Instead of waiting for a pause, this session analyses audio as it arrives:

* every ``ACOUSTIC_HOP_S`` seconds Praat (via parselmouth, the same Burg
  formant tracker and autocorrelation pitch the batch sidecar's Praat script
  uses) produces 10 ms frames of F0 / F1 / F2 / F3 with ~50 ms of lag;
* every ``PHONE_HOP_S`` seconds a wav2vec2 frame classifier (charsiu
  ``en_w2v2_fc_10ms``, ARPABET, 10 ms frames) labels the audio; frames older
  than ``PHONE_LOOKAHEAD_S`` are committed, so labels arrive ~250-350 ms after
  the sound;
* consecutive frames with the same label become a phone; its formants are
  scored against the calibrated ``stats.json`` / ``weights.json`` exactly like
  the batch path's ``compute_resonance`` (z-scores of F1..F3, weighted sum,
  clamped to 0..1);
* every ``STATS_HOP_S`` the session emits rolling medians, zone and the
  Engine A tendency over the last few seconds.

The phone model is English-only; other languages get acoustic frames and F0
statistics but no phone labels or resonance.  See
docs/plans/realtime_experimental.md.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from voiceya.config import BASE_DIR
from voiceya.live.wav import pcm16_to_wav
from voiceya.services.audio_analyser import resonance_calibration
from voiceya.services.audio_analyser.engine_c import (
    _EN_VOWELS,
    _aggregate_per_vowel,
    _median,
    _normalize_lang,
)

logger = logging.getLogger(__name__)

SendFn = Callable[[dict[str, Any]], Awaitable[None]]

SR = 16000
FRAME_S = 0.01
ACOUSTIC_HOP_S = 0.05
ACOUSTIC_WINDOW_S = 0.40
PHONE_HOP_S = 0.05
PHONE_WINDOW_S = 0.80
PHONE_LOOKAHEAD_S = 0.12
STATS_HOP_S = 0.25
NN_HOP_S = 2.0
NN_WINDOW_S = 3.0
MIN_PHONE_S = 0.03
MAX_SESSION_SEC = 180.0
FORMANT_CEILING_HZ = 5500.0

_VENDOR = BASE_DIR / "sidecars" / "visualizer-backend"
_STRESS_RE = re.compile(r"[012]$")
_NON_PHONES = {"[SIL]", "[UNK]", "[PAD]"}

# charsiu/tokenizer_en_cmu vocab (the model repo ships no vocab.json).
CHARSIU_VOCAB: dict[int, str] = {
    v: k
    for k, v in {
        "[SIL]": 0, "NG": 1, "F": 2, "M": 3, "AE": 4, "R": 5, "UW": 6, "N": 7, "IY": 8,
        "AW": 9, "V": 10, "UH": 11, "OW": 12, "AA": 13, "ER": 14, "HH": 15, "Z": 16,
        "K": 17, "CH": 18, "W": 19, "EY": 20, "ZH": 21, "T": 22, "EH": 23, "Y": 24,
        "AH": 25, "B": 26, "P": 27, "TH": 28, "DH": 29, "AO": 30, "G": 31, "L": 32,
        "JH": 33, "OY": 34, "SH": 35, "D": 36, "AY": 37, "S": 38, "IH": 39,
        "[UNK]": 40, "[PAD]": 41,
    }.items()
}  # fmt: skip


# ── calibration tables ───────────────────────────────────────────


def load_stats(lang_short: str = "en") -> tuple[dict[str, list[dict]], list[float]]:
    """stats_*.json keyed by stress-stripped phone + weights for ``lang_short``.

    stats.json (en) is keyed on ARPABET *with* stress digits; the frame
    classifier has none, so we pick one entry per base phone with the
    precedence primary stress → unstressed → secondary → bare.
    """
    stats_file = {"zh": "stats_zh.json", "fr": "stats_fr.json", "ko": "stats_ko.json"}.get(
        lang_short, "stats.json"
    )
    weights_file = {"zh": "weights_zh.json", "fr": "weights_fr.json", "ko": "weights_ko.json"}.get(
        lang_short, "weights.json"
    )
    raw = json.loads((_VENDOR / stats_file).read_text(encoding="utf-8"))
    weights = json.loads((_VENDOR / weights_file).read_text(encoding="utf-8"))
    rank = {"1": 0, "0": 1, "2": 2, "": 3}
    best: dict[str, tuple[int, list[dict]]] = {}
    for key, entry in raw.items():
        base = _STRESS_RE.sub("", key)
        stress = key[len(base) :]
        r = rank.get(stress, 4)
        if base not in best or r < best[base][0]:
            best[base] = (r, entry)
    return {k: v for k, (_r, v) in best.items()}, [float(w) for w in weights]


def score_phone(
    phone: str,
    f1: float | None,
    f2: float | None,
    f3: float | None,
    stats: dict[str, list[dict]],
    weights: list[float],
) -> dict[str, float | None]:
    """z-scores and resonance for one phone, mirroring resonance.compute_resonance."""
    out: dict[str, float | None] = {"z_F1": None, "z_F2": None, "z_F3": None, "resonance": None}
    entry = stats.get(phone)
    if not entry or len(entry) < 4:
        return out
    zs: list[float | None] = []
    for i, f in enumerate((f1, f2, f3), start=1):
        st = entry[i]
        sd = float(st.get("stdev") or 0.0)
        if f is None or sd <= 0:
            zs.append(None)
        else:
            zs.append((float(f) - float(st["mean"])) / sd)
    out["z_F1"], out["z_F2"], out["z_F3"] = zs
    total = 0.5
    for w, z in zip(weights, zs, strict=False):
        if w == 0:
            continue
        if z is None:
            return out
        total += w * z
    out["resonance"] = max(0.0, min(1.0, total))
    return out


# ── phone segmentation (pure) ────────────────────────────────────


class PhoneSegmenter:
    """Turn committed (t, label) frames into phone dicts with scored formants.

    ``acoustic`` maps frame time (rounded to 10 ms) → (f0, f1, f2, f3).
    """

    def __init__(self, stats: dict[str, list[dict]], weights: list[float]) -> None:
        self.stats = stats
        self.weights = weights
        self._cur_label: str | None = None
        self._cur_start: float = 0.0
        self._cur_times: list[float] = []

    def commit(self, t: float, label: str, acoustic: dict[int, tuple]) -> dict[str, Any] | None:
        closed: dict[str, Any] | None = None
        if label != self._cur_label:
            closed = self._close(t, acoustic)
            self._cur_label = label
            self._cur_start = t
            self._cur_times = []
        self._cur_times.append(t)
        return closed

    def flush(self, t: float, acoustic: dict[int, tuple]) -> dict[str, Any] | None:
        closed = self._close(t, acoustic)
        self._cur_label = None
        self._cur_times = []
        return closed

    def _close(self, end: float, acoustic: dict[int, tuple]) -> dict[str, Any] | None:
        label = self._cur_label
        if label is None or label in _NON_PHONES or not self._cur_times:
            return None
        start = self._cur_start
        if end - start < MIN_PHONE_S:
            return None
        # Middle 50 % of the phone: steadier formants than the transitions.
        n = len(self._cur_times)
        lo, hi = n // 4, max(n // 4 + 1, n - n // 4)
        rows = [acoustic.get(int(round(tt / FRAME_S))) for tt in self._cur_times[lo:hi]]
        rows = [r for r in rows if r is not None]
        cols = list(zip(*rows, strict=False)) if rows else [(), (), (), ()]

        def med(vals) -> float | None:
            v = [float(x) for x in vals if x is not None and not np.isnan(x)]
            return _median(v) if v else None

        f0, f1, f2, f3 = (med(c) for c in cols)
        scored = score_phone(label, f1, f2, f3, self.stats, self.weights)
        return {
            "start": round(start, 3),
            "end": round(end, 3),
            "char": "",
            "phone": label,
            "is_vowel": label in _EN_VOWELS,
            "pitch": f0,
            "F1": f1,
            "F2": f2,
            "F3": f3,
            **scored,
        }


# ── models ───────────────────────────────────────────────────────


class _PhoneModel:
    """charsiu wav2vec2 frame classifier, loaded once per process."""

    def __init__(self) -> None:
        self.model = None
        self.lock = threading.Lock()
        self.error: str | None = None

    def load(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import Wav2Vec2ForCTC  # noqa: PLC0415

        torch.set_num_threads(6)
        t0 = time.perf_counter()
        m = Wav2Vec2ForCTC.from_pretrained("charsiu/en_w2v2_fc_10ms")
        m.eval()
        self.model = m
        logger.info("realtime: phone model ready in %.1f s", time.perf_counter() - t0)

    def infer(self, x: np.ndarray) -> np.ndarray:
        """float32 mono 16 kHz → int label ids, one per 10 ms frame."""
        import torch  # noqa: PLC0415

        x = (x - x.mean()) / (x.std() + 1e-7)
        with self.lock, torch.inference_mode():
            logits = self.model(torch.from_numpy(x.astype(np.float32))[None]).logits[0]
        return logits.argmax(-1).cpu().numpy()


PHONE_MODEL = _PhoneModel()


async def ensure_phone_model() -> None:
    if PHONE_MODEL.model is not None or PHONE_MODEL.error:
        return
    try:
        await asyncio.to_thread(PHONE_MODEL.load)
    except Exception as exc:  # pragma: no cover — network / disk
        PHONE_MODEL.error = str(exc)[:200]
        logger.warning("realtime: phone model unavailable: %s", exc)


def praat_frames(
    x: np.ndarray, sr: int, t0: float
) -> list[tuple[float, float | None, float | None, float | None, float | None]]:
    """10 ms frames of (t, f0, f1, f2, f3) for ``x`` starting at time ``t0``."""
    import parselmouth  # noqa: PLC0415

    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    formant = snd.to_formant_burg(
        time_step=FRAME_S,
        max_number_of_formants=5,
        maximum_formant=FORMANT_CEILING_HZ,
        window_length=0.025,
        pre_emphasis_from=50,
    )
    pitch = snd.to_pitch_ac(time_step=FRAME_S, pitch_floor=75, pitch_ceiling=600)
    out = []
    for tt in formant.ts():
        vals = []
        for k in (1, 2, 3):
            v = formant.get_value_at_time(k, tt)
            vals.append(None if v is None or np.isnan(v) else float(v))
        f0 = pitch.get_value_at_time(tt)
        f0 = None if f0 is None or np.isnan(f0) or f0 <= 0 else float(f0)
        out.append((t0 + float(tt), f0, vals[0], vals[1], vals[2]))
    return out


# ── the session ──────────────────────────────────────────────────


class RealtimeSession:
    def __init__(
        self, *, language: str, mode: str, script: str | None, send: SendFn, sr: int = SR
    ) -> None:
        self.language = language
        self.lang_short = _normalize_lang(language)
        self.mode = mode
        self.script = script
        self.send = send
        self.sr = sr
        self.phones_enabled = self.lang_short == "en" and PHONE_MODEL.model is not None
        self.stats, self.weights = load_stats(self.lang_short)
        self.segmenter = PhoneSegmenter(self.stats, self.weights)

        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0  # absolute sample index of _buf[0]
        self._n_total = 0
        self._stopped = False

        self._acoustic_emit_t = 0.0  # frames with center <= this were emitted
        self._acoustic: dict[int, tuple] = {}  # frame idx → (f0, f1, f2, f3)
        self._f0_recent: deque[tuple[float, float]] = deque()  # (t, f0)
        self._phone_committed_t = 0.0
        self._phone_last_run = 0
        self._phone_task: asyncio.Task | None = None
        self._nn_last_run = 0
        self._nn_task: asyncio.Task | None = None
        self._nn: dict[str, Any] | None = None
        self._stats_last_t = 0.0
        self.phones: list[dict[str, Any]] = []
        self._t_start = time.monotonic()

    # ── input ───────────────────────────────────────────────────

    @property
    def elapsed_sec(self) -> float:
        return self._n_total / self.sr

    async def feed(self, data: bytes) -> None:
        if self._stopped:
            return
        if self.elapsed_sec >= MAX_SESSION_SEC:
            await self.send(
                {"type": "error", "msg": "max session length reached", "msg_key": "live.maxLength"}
            )
            await self.stop()
            return
        pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, pcm])
        self._n_total += pcm.size
        keep = int((PHONE_WINDOW_S + NN_WINDOW_S) * self.sr)
        if self._buf.size > keep:
            drop = self._buf.size - keep
            self._buf = self._buf[drop:]
            self._buf_start += drop
        await self._acoustic_step()
        self._maybe_phone_step()
        self._maybe_nn_step()
        await self._maybe_stats()

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for task in (self._phone_task, self._nn_task):
            if task is not None:
                try:
                    await task
                except Exception:
                    pass
        closed = self.segmenter.flush(self.elapsed_sec, self._acoustic)
        if closed:
            self.phones.append(closed)
            await self.send({"type": "rt_phones", "phones": [closed]})
        await self._emit_stats(force=True)

    def abort(self) -> None:
        self._stopped = True
        for task in (self._phone_task, self._nn_task):
            if task is not None:
                task.cancel()

    # ── acoustic frames (Praat) ─────────────────────────────────

    def _window(self, seconds: float) -> tuple[np.ndarray, float]:
        n = min(self._buf.size, int(seconds * self.sr))
        x = self._buf[self._buf.size - n :]
        t0 = (self._n_total - n) / self.sr
        return x, t0

    async def _acoustic_step(self) -> None:
        t_end = self.elapsed_sec
        if t_end - self._acoustic_emit_t < ACOUSTIC_HOP_S:
            return
        x, t0 = self._window(ACOUSTIC_WINDOW_S)
        if x.size < int(0.06 * self.sr):
            return
        try:
            frames = praat_frames(x, self.sr, t0)
        except Exception as exc:
            logger.warning("realtime: praat failed: %s", exc)
            return
        fresh = [f for f in frames if self._acoustic_emit_t < f[0] <= t_end - 0.02]
        if not fresh:
            return
        for t, f0, f1, f2, f3 in fresh:
            self._acoustic[int(round(t / FRAME_S))] = (f0, f1, f2, f3)
            if f0 is not None:
                self._f0_recent.append((t, f0))
        self._acoustic_emit_t = fresh[-1][0]
        # Trim stores to what phone scoring / rolling stats still need.
        cutoff = int(round((t_end - 8.0) / FRAME_S))
        for k in [k for k in self._acoustic if k < cutoff]:
            del self._acoustic[k]
        while self._f0_recent and self._f0_recent[0][0] < t_end - 8.0:
            self._f0_recent.popleft()
        await self.send(
            {
                "type": "rt_frames",
                "t": round(t_end, 3),
                "frames": [
                    [round(t, 3), _r(f0), _r(f1), _r(f2), _r(f3)] for t, f0, f1, f2, f3 in fresh
                ],
            }
        )

    # ── phone labels (wav2vec2) ─────────────────────────────────

    def _maybe_phone_step(self) -> None:
        if not self.phones_enabled or self._stopped:
            return
        if self._n_total - self._phone_last_run < int(PHONE_HOP_S * self.sr):
            return
        if self._phone_task is not None and not self._phone_task.done():
            return  # still busy: this hop is skipped, next one covers the gap
        self._phone_last_run = self._n_total
        x, t0 = self._window(PHONE_WINDOW_S)
        if x.size < int(0.3 * self.sr):
            return
        self._phone_task = asyncio.create_task(self._phone_run(x.copy(), t0))

    async def _phone_run(self, x: np.ndarray, t0: float) -> None:
        t_end = t0 + x.size / self.sr
        try:
            ids = await asyncio.to_thread(PHONE_MODEL.infer, x)
        except Exception as exc:
            logger.warning("realtime: phone model failed: %s", exc)
            return
        if self._stopped:
            return
        n = len(ids)
        if n == 0:
            return
        step = (x.size / self.sr) / n
        commit_until = t_end - PHONE_LOOKAHEAD_S
        closed: list[dict[str, Any]] = []
        for i, idx in enumerate(ids):
            t = t0 + (i + 0.5) * step
            if t <= self._phone_committed_t or t > commit_until:
                continue
            label = CHARSIU_VOCAB.get(int(idx), "[UNK]")
            done = self.segmenter.commit(t, label, self._acoustic)
            if done:
                closed.append(done)
            self._phone_committed_t = t
        if closed:
            self.phones.extend(closed)
            await self.send(
                {
                    "type": "rt_phones",
                    "phones": closed,
                    "lag_ms": int((self.elapsed_sec - closed[-1]["end"]) * 1000),
                }
            )

    # ── Engine A tendency (rolling) ─────────────────────────────

    def _maybe_nn_step(self) -> None:
        if self._stopped or self._n_total - self._nn_last_run < int(NN_HOP_S * self.sr):
            return
        if self._nn_task is not None and not self._nn_task.done():
            return
        self._nn_last_run = self._n_total
        x, _t0 = self._window(NN_WINDOW_S)
        if x.size < int(1.0 * self.sr):
            return
        self._nn_task = asyncio.create_task(self._nn_run(x.copy()))

    async def _nn_run(self, x: np.ndarray) -> None:
        from voiceya.services.audio_analyser import seg as _seg  # noqa: PLC0415

        if _seg.SEG is None:
            return
        wav = pcm16_to_wav((x * 32767).astype(np.int16), self.sr)
        try:
            raw = await asyncio.to_thread(_seg.SEG, io.BytesIO(wav))
        except Exception as exc:
            logger.debug("realtime: engine A failed: %s", exc)
            return
        dur = {"female": 0.0, "male": 0.0}
        conf = {"female": 0.0, "male": 0.0}
        for item in raw:
            label = item[0]
            if label not in dur:
                continue
            d = float(item[2]) - float(item[1])
            dur[label] += d
            if len(item) > 3 and item[3] is not None:
                conf[label] += d * float(item[3])
        speech = dur["female"] + dur["male"]
        if speech <= 0:
            self._nn = {"label": None, "confidence": 0.0, "female_ratio": None}
            return
        label = "female" if dur["female"] >= dur["male"] else "male"
        self._nn = {
            "label": label,
            "confidence": round(conf[label] / dur[label], 3) if dur[label] else 0.0,
            "female_ratio": round(dur["female"] / speech, 3),
        }

    # ── rolling stats ───────────────────────────────────────────

    async def _maybe_stats(self) -> None:
        if self.elapsed_sec - self._stats_last_t >= STATS_HOP_S:
            await self._emit_stats()

    async def _emit_stats(self, force: bool = False) -> None:
        t = self.elapsed_sec
        self._stats_last_t = t
        f0_3s = [f for tt, f in self._f0_recent if tt >= t - 3.0]
        recent = [p for p in self.phones if p["end"] >= t - 4.0 and p.get("resonance") is not None]
        recent_v = [p["resonance"] for p in recent if p["is_vowel"]]
        all_v = [
            p["resonance"] for p in self.phones if p["is_vowel"] and p.get("resonance") is not None
        ]
        per_vowel = _aggregate_per_vowel(self.phones, self.lang_short) if self.phones else []
        qualifying = [
            v["resonance_med"] for v in per_vowel if v["is_vowel"] and (v.get("n") or 0) >= 5
        ]
        zone_med = (
            _median([float(q) for q in qualifying])
            if qualifying
            else (_median(all_v) if all_v else None)
        )
        await self.send(
            {
                "type": "rt_stats",
                "t": round(t, 2),
                "f0_med_3s": _r(_median(f0_3s)) if f0_3s else None,
                "res_med_4s": _r(_median(recent_v), 3) if recent_v else None,
                "res_med_session": _r(_median(all_v), 3) if all_v else None,
                "zone_key": resonance_calibration.classify_zone(zone_med, self.language)
                if zone_med is not None
                else None,
                "phone_count": len(self.phones),
                "vowel_count": len(all_v),
                "per_vowel": per_vowel[:12],
                "nn": self._nn,
                "phones_enabled": self.phones_enabled,
                "final": force,
            }
        )


def _r(x: float | None, nd: int = 1) -> float | None:
    return None if x is None else round(float(x), nd)
