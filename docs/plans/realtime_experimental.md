# Frame-level real-time mode (experimental, phase C)

Date: 03 Sep 2026. Branch: `realtime-experimental` (on top of `sentence-live`). Background:
`~/claude-personal/analysis/voiceduck_realtime_options.md`, tier 3.

## What it does

Tick "Live preview" and then "Instant mode" in the record panel (English UI only). While the mic is
open the player card shows a scrolling 8 s strip:

- F0 lane: one dot per 10 ms, log scale 70 to 420 Hz, the app's 145 / 185 Hz zone bands behind it.
- Phone lane: one block per phone, labelled in ARPABET, filled with its resonance colour. The grey
  band at the right edge is the phone model's lookahead: sound the model has heard but not yet
  labelled.
- Resonance lane: per-phone bars against the language's p25 / p75 zone lines; consonants drawn
  faint, vowels solid.
- Header: F0 median over 3 s, resonance median over 4 s (vowels), zone, Engine A tendency over the
  last 3 s, current phone lag.

When you stop, the full take is still analysed by the batch path as before.

## Pipeline (`voiceya/live/realtime.py`)

| Stage | Cadence | Tool | Lag |
|---|---|---|---|
| Browser capture | 50 ms chunks | AudioWorklet, 16 kHz int16 | 50 ms |
| Pitch + formants | every 50 ms on the last 0.4 s | parselmouth: `to_pitch_ac(75..600)`, `to_formant_burg(5 formants, 5500 Hz, 25 ms, pre-emphasis 50)`; the same algorithms the sidecar's Praat script runs | ~70 ms end to end |
| Phone labels | every 50 ms on the last 0.8 s, frames older than 120 ms committed | charsiu `en_w2v2_fc_10ms` (wav2vec2-base frame classifier, ARPABET, 10 ms frames), CPU, 6 torch threads | median 250 ms, p90 ~600 ms |
| Phone scoring | when a label run ends | median of F1..F3 over the middle 50 % of the phone, z-scored against `stats.json` (stress-stripped, primary stress preferred) with `weights.json`, clamped to 0..1: the batch `compute_resonance` formula | with the label |
| Rolling stats | every 250 ms | medians over 3 s / 4 s / session, per-vowel aggregation via the batch `_aggregate_per_vowel`, zone via `resonance_calibration.classify_zone` | |
| Engine A tendency | every 2 s on the last 3 s | inaSpeechSegmenter | |

Measured with `scripts/live_e2e_client.py clip.wav --realtime` on the three-sentence espeak clip: 2097
acoustic frames, 197 phones, frame lag under one chunk, phone lag median 250 ms (p90 614 ms).

## Protocol additions

`start` takes `"realtime": true`. Server events: `rt_frames {t, frames:[[t,f0,f1,f2,f3],...]}`,
`rt_phones {phones:[...], lag_ms}`, `rt_stats {t, f0_med_3s, res_med_4s, res_med_session, zone_key,
phone_count, vowel_count, per_vowel, nn, phones_enabled, final}`. `/api/live/healthz` reports
`realtime.phone_model`.

## Known limits and honest caveats

- English only. The phone model is trained on English ARPABET; other languages get pitch only.
- The phone model was trained on clean read speech. Frame labels at the right edge (inside the
  lookahead) are not committed, but labels 120 ms from the edge still have less right context than
  the model saw in training, so short phones near the edge are the least reliable.
- Resonance per phone is noisier than in batch: one phone's median over ~30 to 80 ms of frames,
  no clip-level outlier pass. The 4 s median in the header is the number to watch; single blocks
  will flicker.
- Praat here is 6.1.38 (parselmouth) versus 6.4.x in the sidecar; the Burg tracker is the same
  algorithm and the numbers agree to within a few Hz on the same frames, but this has not been
  audited across the calibration set.
- CPU only. The 6800 XT could run the phone model through ROCm, but torch in this venv is the CPU
  build; switching would need a second environment.
- Lag is dominated by the phone model's hop + lookahead + inference (50 + 120 + 60 to 190 ms) and
  the fact that a phone is only closed when the next label starts. Going lower means committing
  frames before their right context, which trades lag for label accuracy.

## Files

- `voiceya/live/realtime.py`: `RealtimeSession`, `PhoneSegmenter`, `score_phone`, `load_stats`.
- `voiceya/live/app.py`: dispatches on `realtime`, loads the phone model at startup.
- `web/src/modules/realtime-view.js`: the strip chart. `web/src/live-worklet.js` takes a chunk size.
- `tests/test_live_realtime.py`: stats lookup, scoring formula, segmentation.
- `pyproject.toml`: new `realtime` dependency group (parselmouth, transformers).
