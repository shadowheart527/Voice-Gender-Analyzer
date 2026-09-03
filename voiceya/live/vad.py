"""Energy-based sentence segmenter for the live path.

Pure numpy, no models.  Frames of 20 ms are classified as speech when their
level clears an adaptive noise floor; a sentence closes after ``end_silence_s``
of non-speech and is emitted with ``pre_roll_s`` / ``post_roll_s`` of context.
The thresholds mirror the sidecar's ffmpeg silencedetect (-30 dB, 0.5 s) so
live sentence breaks land where the batch path would also put them.

Events are plain tuples so the session layer can stay transport-agnostic:

    ("speaking", bool, t_sec)
    ("sentence", start_sec, end_sec, np.ndarray[int16])
"""

from __future__ import annotations

import math

import numpy as np

SentenceEvent = tuple


class SentenceSegmenter:
    def __init__(
        self,
        *,
        sr: int = 16000,
        frame_ms: int = 20,
        pre_roll_s: float = 0.2,
        post_roll_s: float = 0.2,
        end_silence_s: float = 0.5,
        min_speech_s: float = 0.6,
        max_sentence_s: float = 12.0,
        onset_frames: int = 3,
        abs_floor_db: float = -48.0,
        margin_db: float = 8.0,
        max_threshold_db: float = -22.0,
    ) -> None:
        self.sr = sr
        self.frame_len = int(sr * frame_ms / 1000)
        self.frame_s = self.frame_len / sr
        self.pre_roll = max(0, int(round(pre_roll_s / self.frame_s)))
        self.post_roll = max(0, int(round(post_roll_s / self.frame_s)))
        self.end_silence = max(1, int(round(end_silence_s / self.frame_s)))
        self.min_speech = max(1, int(round(min_speech_s / self.frame_s)))
        self.max_sentence = max(self.min_speech + 1, int(round(max_sentence_s / self.frame_s)))
        self.onset_frames = max(1, onset_frames)
        self.abs_floor_db = abs_floor_db
        self.margin_db = margin_db
        self.max_threshold_db = max_threshold_db

        self._pending = np.zeros(0, dtype=np.int16)  # partial frame carry-over
        self._frames: list[np.ndarray] = []  # retained frames (from sentence start or pre-roll)
        self._levels: list[float] = []  # dBFS per retained frame, same indexing
        self._retained_from = 0  # absolute index of self._frames[0]
        self._frame_index = 0  # absolute index of the next frame to classify
        self._noise_floor: float | None = None
        self._in_speech = False
        self._onset_run = 0
        self._speech_start = 0  # absolute frame index where speech began
        self._last_speech = 0  # absolute index of the last speech frame
        self.threshold_db = abs_floor_db

    # ── public API ──────────────────────────────────────────────────

    @property
    def elapsed_sec(self) -> float:
        return self._frame_index * self.frame_s

    def feed(self, pcm: np.ndarray) -> list[SentenceEvent]:
        """Consume int16 samples; return the events they completed."""
        if pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16)
        if self._pending.size:
            pcm = np.concatenate([self._pending, pcm])
        n_full = pcm.size // self.frame_len
        events: list[SentenceEvent] = []
        for i in range(n_full):
            frame = pcm[i * self.frame_len : (i + 1) * self.frame_len]
            events.extend(self._push_frame(frame))
        self._pending = pcm[n_full * self.frame_len :].copy()
        return events

    def flush(self) -> list[SentenceEvent]:
        """End of stream: close any open sentence."""
        events: list[SentenceEvent] = []
        if self._in_speech:
            end = min(self._frame_index - 1, self._last_speech + self.post_roll)
            events.extend(self._close_sentence(end))
        self._pending = np.zeros(0, dtype=np.int16)
        return events

    # ── internals ───────────────────────────────────────────────────

    @staticmethod
    def _level_db(frame: np.ndarray) -> float:
        x = frame.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        return 20.0 * math.log10(rms + 1e-9)

    def _update_threshold(self, level: float, is_speech: bool) -> None:
        if self._noise_floor is None:
            self._noise_floor = min(level, self.max_threshold_db - self.margin_db)
        elif level < self._noise_floor:
            # Follow drops immediately so a loud first frame cannot pin the floor high.
            self._noise_floor = level
        elif not is_speech:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * level
        thr = max(self._noise_floor + self.margin_db, self.abs_floor_db)
        self.threshold_db = min(thr, self.max_threshold_db)

    def _push_frame(self, frame: np.ndarray) -> list[SentenceEvent]:
        idx = self._frame_index
        self._frame_index += 1
        level = self._level_db(frame)
        is_speech = level > self.threshold_db
        self._update_threshold(level, is_speech)

        self._frames.append(frame.copy())
        self._levels.append(level)
        events: list[SentenceEvent] = []

        if not self._in_speech:
            if is_speech:
                self._onset_run += 1
                if self._onset_run >= self.onset_frames:
                    self._in_speech = True
                    self._speech_start = idx - self.onset_frames + 1
                    self._last_speech = idx
                    self._onset_run = 0
                    events.append(("speaking", True, self._speech_start * self.frame_s))
            else:
                self._onset_run = 0
                # Idle: keep only the pre-roll window.
                keep_from = max(self._retained_from, idx + 1 - self.pre_roll)
                self._drop_before(keep_from)
            return events

        # In speech.
        if is_speech:
            self._last_speech = idx
        silence_run = idx - self._last_speech
        if silence_run >= self.end_silence:
            end = min(idx, self._last_speech + self.post_roll)
            events.extend(self._close_sentence(end))
        elif idx - self._speech_start + 1 >= self.max_sentence:
            events.extend(self._force_cut(idx))
        return events

    def _drop_before(self, abs_index: int) -> None:
        n = abs_index - self._retained_from
        if n <= 0:
            return
        del self._frames[:n]
        del self._levels[:n]
        self._retained_from = abs_index

    def _slice(self, start_abs: int, end_abs: int) -> np.ndarray:
        a = max(0, start_abs - self._retained_from)
        b = end_abs - self._retained_from + 1
        if b <= a:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._frames[a:b])

    def _close_sentence(self, end_abs: int) -> list[SentenceEvent]:
        events: list[SentenceEvent] = []
        speech_frames = self._last_speech - self._speech_start + 1
        if speech_frames >= self.min_speech:
            start_abs = max(self._retained_from, self._speech_start - self.pre_roll)
            samples = self._slice(start_abs, end_abs)
            events.append(
                ("sentence", start_abs * self.frame_s, (end_abs + 1) * self.frame_s, samples)
            )
        events.append(("speaking", False, (self._last_speech + 1) * self.frame_s))
        self._in_speech = False
        self._onset_run = 0
        # Keep a pre-roll window behind the current frame for the next onset.
        self._drop_before(max(self._retained_from, self._frame_index - self.pre_roll))
        return events

    def _force_cut(self, idx: int) -> list[SentenceEvent]:
        """Sentence hit the length cap: cut at the quietest recent frame and continue."""
        window = max(1, int(round(1.5 / self.frame_s)))
        lo_abs = max(self._speech_start + self.min_speech, idx - window)
        lo = lo_abs - self._retained_from
        hi = idx - self._retained_from
        if hi <= lo:
            cut_abs = idx
        else:
            rel = int(np.argmin(self._levels[lo : hi + 1]))
            cut_abs = lo_abs + rel
        start_abs = max(self._retained_from, self._speech_start - self.pre_roll)
        samples = self._slice(start_abs, cut_abs)
        events: list[SentenceEvent] = [
            ("sentence", start_abs * self.frame_s, (cut_abs + 1) * self.frame_s, samples)
        ]
        # The remainder starts a new sentence immediately (still speaking).
        self._speech_start = cut_abs + 1
        self._last_speech = max(self._last_speech, self._speech_start)
        self._drop_before(max(self._retained_from, self._speech_start - self.pre_roll))
        return events
