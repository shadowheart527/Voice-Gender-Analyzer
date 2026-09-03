# Sentence-live mode (phase A)

Status: implemented and verified 03 Sep 2026 (unit tests, headless e2e client, Playwright with a fake
microphone). Measured 1.3 to 2.1 s per sentence on the desktop.

Date: 02 Sep 2026. Branch: `local-setup` (fork shadowheart527). Background and measurements:
`~/claude-personal/analysis/voiceduck_realtime_options.md`.

## Goal

While the user records, analyse each pause-delimited stretch of speech as soon as the pause is
detected and grow the normal result panels (phone heatmap, resonance per vowel, pitch, NN estimate)
sentence by sentence. Target lag about one second after each pause in script mode. The final
batch analysis still runs on the full take when recording stops, so history and exports are
unchanged.

## Architecture

```
browser mic ──AudioWorklet (16 kHz int16)──WebSocket /api/live──▶ live worker (voiceya.live, :8091)
                                                                    ├─ SentenceSegmenter (energy VAD)
                                                                    └─ per sentence, in order:
                                                                         ├─ Whisper (sentence)  ─┐ gather
                                                                         ├─ Engine A (sentence) ─┘
                                                                         ├─ ScriptCursor (script mode)
                                                                         ├─ POST sidecar /engine_c/analyze
                                                                         ├─ pyin (sentence)
                                                                         └─ rebuild batch-shaped summary
                                                                              → {"type":"summary", result}
browser: renderFromSummary(result) + PhoneTimeline.setData(result.summary.engine_c)
```

- The live worker is its own process (`python -m voiceya.live`), spawned by `run_app.py`, because
  CLAUDE.md keeps TensorFlow out of the API process and taskiq is the wrong shape for a stateful
  session. It loads Engine A and warms Whisper at startup.
- Vite proxies `/api/live` (WebSocket) to `LIVE_DEV_PORT` (default 8091).
- The summary sent after every sentence has exactly the batch schema (`status`, `summary`,
  `analysis`, `filename`), plus `summary.live` with sentence count and latency, so the frontend
  reuses `renderFromSummary` untouched.

## Modules

| File | Role | Tests |
|---|---|---|
| `voiceya/live/vad.py` | `SentenceSegmenter`: 20 ms frames, adaptive noise floor, onset 60 ms, end 0.5 s silence, pre/post roll 0.2 s, min speech 0.6 s, max 12 s with cut at quietest frame | `tests/test_live_vad.py` |
| `voiceya/live/script_cursor.py` | `ScriptCursor`: map ASR tokens onto the next span of the script (difflib), return canonical script text; whole-script fallback for restarts; ASR text fallback | `tests/test_live_script_cursor.py` |
| `voiceya/live/wav.py` | int16 PCM → WAV bytes | covered by session e2e |
| `voiceya/live/session.py` | `LiveSession`: feeds VAD, sequential sentence pipeline, running aggregates, summary builder | `tests/test_live_session_summary.py` (aggregation only, no models) |
| `voiceya/live/app.py`, `__main__.py` | FastAPI WebSocket `/api/live`, `/api/live/healthz`, lifespan loads Engine A + warms Whisper | e2e script `scripts/live_e2e_client.py` |
| `voiceya/services/audio_analyser/engine_c.py` | extract `analyze_with_sidecar()` from `run_engine_c` (no behaviour change) | existing tests |
| `voiceya/services/audio_analyser/f0_panel.py` | extract `panel_from_voiced_f0()` so the live path can merge per-sentence pyin output | existing tests |
| `web/src/live-worklet.js` | AudioWorkletProcessor: float → int16, resample to 16 kHz if the context refuses that rate, post 100 ms chunks | manual |
| `web/src/modules/live-session.js` | WebSocket client + worklet wiring | manual |
| `web/src/modules/recorder.js` | new `onStreamStart(stream)` / `onStreamStop()` callbacks; live status line | manual |
| `web/src/main.js` | live toggle in record panel, `live` phase, event → render | Playwright with fake mic |
| `web/src/modules/i18n.js` | new `live.*` keys in all four dictionaries (drift guard) | dev build |

## Protocol

Client → server: text `{"type":"start","language":"en-US","mode":"script","script":"..."}`, then
binary frames of little-endian int16 mono 16 kHz, then text `{"type":"stop"}`.

Server → client: `ready`, `vad {speaking, t}`, `sentence_queued {index, start, end}`,
`summary {index, result, sentence:{start,end,transcript,latency_ms}}`, `sentence_failed {index,
reason}`, `done`, `error {msg}`.

## Sentence pipeline details

- Both modes run Whisper on the sentence (about 0.3 s for base.en on CPU). In script mode the
  recognised words are matched against the script through `ScriptCursor` so the aligner receives
  canonical text and skipped or repeated words do not derail it. If matching fails the ASR text is
  used as is.
- Engine A runs on the sentence WAV in parallel with Whisper; segment times are offset to absolute
  session time. Its 3 s gate does not apply.
- Sidecar call is the existing endpoint. Phones are offset to absolute time. `formant_ceiling_hz`
  is the most common value across sentences.
- pyin runs per sentence; voiced F0 values are concatenated and the panel recomputed, so
  median/p25/p75 are over the whole session so far.
- Silence ranges are the gaps between sentences (used by the frontend as sentence breaks).
- Per-vowel aggregation, zone and advice reuse the batch functions on the accumulated arrays.

## Out of scope for phase A

In-process Praat (parselmouth), raw-PCM sidecar input, session CMVN, frame-level pitch overlay
from InFormant. Listed in the analysis as phases B and C.
