# Voiceduck local setup notes (Charlotte's desktop, Bazzite)

Fork: github.com/shadowheart527/Voice-Gender-Analyzer (origin). Upstream: guojunximi-cell (remote `upstream`).
Set up 02 Sep 2026. Everything below is what differs from the upstream README.

## Run

    local/start.sh      # starts podman containers, then run_app.py (backend + worker + vite), opens http://127.0.0.1:5173
    local/stop.sh       # stops the two podman containers (Ctrl+C the start.sh terminal first)

Backend is on http://127.0.0.1:8090 (not 8080: Steam's webhelper owns 8080). Vite proxies `/api` to it.
The sentence-live worker (`python -m voiceya.live`) is on 127.0.0.1:8091; Vite proxies the `/api/live`
WebSocket to it. `run_app.py` starts all four processes (backend, worker, live, vite).

## What was changed / created

- `.env` (gitignored): bare-metal values (Redis + sidecar on 127.0.0.1, Engine C on, DEBUG_NO_LIMITS on,
  50 MB upload cap). `ENGINE_C_MAX_AUDIO_MB` deliberately left out: pydantic Settings rejects unknown keys in .env,
  so upstream's `.env.example` actually crashes the app if copied verbatim.
- `web/vite.config.js`: proxy target reads `BACKEND_DEV_PORT`.
- `voiceya/sidecars/visualizer-backend.Dockerfile`: fully qualified `docker.io/` base image (podman refuses
  short names without a TTY) and MFA pinned to 3.4.1. Built as `localhost/voiceya-engine-c:dev` (~3 GB).
- Python env: `uv sync` into `.venv` (Python 3.13.9, TF 2.21 CPU, torch 2.11 CPU, funasr, faster-whisper).
- Frontend: `pnpm install` (pnpm installed globally via npm).
- Models cached: inaSpeechSegmenter in `~/.keras/inaSpeechSegmenter/`, faster-whisper base.en in HF cache.
  FunASR Paraformer-zh (zh-CN free mode) is NOT pre-downloaded; first zh-CN free-mode request will pull ~1 GB.

## Containers (recreate if lost)

    podman run -d --name voiceya-redis --restart unless-stopped -p 127.0.0.1:6379:6379 \
      -v voiceya-redis-data:/data docker.io/library/redis:7-alpine redis-server --appendonly yes

    podman build -f voiceya/sidecars/visualizer-backend.Dockerfile -t voiceya-engine-c:dev .
    podman run -d --name voiceya-engine-c --restart unless-stopped -p 127.0.0.1:8001:8001 \
      -e ENGINE_C_MAX_AUDIO_MB=50 localhost/voiceya-engine-c:dev

## Verified 02 Sep 2026

- Engine A on a real 17 s English clip: female 99.7 %, F0 median 210 Hz.
- Engine C free mode (Whisper ASR -> MFA -> Praat): 188 phones, resonance per vowel, 3.6 s wall time after the MFA pin (61 s before).
- Engine C script mode: 188 phones, 3.5 s (52 s before).
- UI renders at 127.0.0.1:5173 with no console errors.

## Sentence-live mode (added 03 Sep 2026)

Tick "Live preview" in the record panel. Each pause-delimited sentence is analysed while you keep
reading and the panels grow; when you stop, the full take is analysed as before. Design and protocol:
`docs/plans/sentence_live_mode.md`. Without a browser:

    python scripts/live_e2e_client.py clip16k.wav --mode script --script-file rainbow.txt

`.env` sets `ADVICE_MINIMAL_TIER_SEC=3` (upstream 10) so the tone / resonance panels appear after the first
sentence instead of after 10 s of audio; `ADVICE_STANDARD_TIER_SEC` (30) is the standard/full boundary.

The live path sends `fast_only=1` to the sidecar so a sentence the kalpy aligner cannot align fails in
~0.5 s instead of falling back to the 45 s MFA CLI and blocking every sentence queued behind it (this was
the "stops updating after a minute" bug on 02 Sep 2026). Per-sentence sidecar timeout is 15 s.

Measured on the desktop: 1.3 to 2.1 s from end of pause to panels updated (Whisper + Engine A +
pyin in parallel ~0.5 s, sidecar ~0.7 to 1.2 s including its ffmpeg and Praat subprocesses).

## Instant mode (experimental branch `realtime-experimental`, 03 Sep 2026)

Second checkbox under "Live preview". Frame-level: Praat pitch/formants every 10 ms via parselmouth,
ARPABET phone labels from the charsiu wav2vec2 frame classifier (CPU, ~250 ms median lag), per-phone
resonance z-scores against the same stats.json, drawn on a scrolling strip. English only. Design and
caveats: `docs/plans/realtime_experimental.md`. The phone model (~380 MB) downloads from Hugging Face
the first time the live worker starts on this branch. Headless check:

    python scripts/live_e2e_client.py clip16k.wav --realtime

## Known wrinkles

- MFA is pinned to 3.4.1 in the sidecar Dockerfile (upstream leaves it unpinned). 3.4.2 removed the
  `language=` kwarg from `tokenize_utterance_text`, which breaks the wrapper's preloaded kalpy aligner; every
  request then silently falls back to the MFA CLI (~45 s cold start per clip). Symptom in sidecar logs:
  `warmup FAILED: tokenize_utterance_text() got an unexpected keyword argument 'language'`.
- Repo tests: test_chunker 12/12, test_advice_v2 18/18 pass. test_resonance_calibration (5 fails) and
  test_multichunk_merge (1 fail) are stale upstream tests asserting the old 5000 Hz English ceiling; the code
  now has `en` in `_ADAPTIVE_LANGS`. Not an environment problem.
- Engine C is CPU-only and slow (~3x realtime for the sidecar). ROCm is not used anywhere in this stack.
