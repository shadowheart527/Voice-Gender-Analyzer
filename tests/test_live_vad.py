"""Standalone tests for voiceya/live/vad.py (SentenceSegmenter).

Run: ``python tests/test_live_vad.py`` from the repo root (no pytest needed).
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voiceya.live.vad import SentenceSegmenter  # noqa: E402

SR = 16000


def _tone(sec: float, amp: float = 0.3, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    # Add harmonics so the signal looks voice-like rather than a pure sine.
    x = amp * (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2 * freq * t))
    return (x * 32767 / 1.5).astype(np.int16)


def _silence(sec: float, noise_amp: float = 0.002) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.normal(0, noise_amp, int(sec * SR)) * 32767).astype(np.int16)


def _run(signal: np.ndarray, chunk: int = 1600, **kw) -> list:
    seg = SentenceSegmenter(sr=SR, **kw)
    events = []
    for i in range(0, len(signal), chunk):
        events.extend(seg.feed(signal[i : i + chunk]))
    events.extend(seg.flush())
    return events


def _sentences(events):
    return [e for e in events if e[0] == "sentence"]


def test_single_burst_gives_one_sentence_with_rolls():
    sig = np.concatenate([_silence(1.0), _tone(2.0), _silence(1.5)])
    sents = _sentences(_run(sig))
    assert len(sents) == 1, sents
    _, start, end, samples = sents[0]
    assert 0.7 <= start <= 0.9, start  # 1.0 s onset minus 0.2 s pre-roll
    assert 3.1 <= end <= 3.35, end  # 3.0 s offset plus 0.2 s post-roll
    assert abs(len(samples) / SR - (end - start)) < 0.03


def test_short_gap_does_not_split():
    sig = np.concatenate([_silence(0.5), _tone(1.0), _silence(0.3), _tone(1.0), _silence(1.0)])
    sents = _sentences(_run(sig))
    assert len(sents) == 1, [(s[1], s[2]) for s in sents]


def test_long_gap_splits_into_two():
    sig = np.concatenate([_silence(0.5), _tone(1.0), _silence(0.8), _tone(1.0), _silence(1.0)])
    sents = _sentences(_run(sig))
    assert len(sents) == 2, [(s[1], s[2]) for s in sents]
    assert sents[0][2] <= sents[1][1]


def test_blip_is_dropped():
    sig = np.concatenate([_silence(0.5), _tone(0.3), _silence(1.0)])
    assert _sentences(_run(sig)) == []


def test_flush_closes_open_sentence():
    sig = np.concatenate([_silence(0.5), _tone(1.5)])
    sents = _sentences(_run(sig))
    assert len(sents) == 1
    assert sents[0][2] >= 1.9


def test_long_speech_is_cut_below_cap():
    sig = np.concatenate([_silence(0.5), _tone(15.0), _silence(1.0)])
    sents = _sentences(_run(sig, max_sentence_s=12.0))
    assert len(sents) == 2, [(s[1], s[2]) for s in sents]
    for _, s, e, _samples in sents:
        assert e - s <= 12.5
    total = sum(e - s for _, s, e, _x in sents)
    assert total >= 14.5, total


def test_speech_from_first_frame_is_detected():
    sig = np.concatenate([_tone(2.0), _silence(1.0)])
    sents = _sentences(_run(sig))
    assert len(sents) == 1, sents
    assert sents[0][1] < 0.2


def test_speaking_events_bracket_sentence():
    sig = np.concatenate([_silence(0.5), _tone(1.0), _silence(1.0)])
    ev = _run(sig)
    kinds = [e[0] for e in ev]
    assert kinds.index("speaking") < kinds.index("sentence")
    assert ev[0][1] is True
    assert [e for e in ev if e[0] == "speaking"][-1][1] is False


def test_partial_chunks_are_carried():
    sig = np.concatenate([_silence(0.5), _tone(1.0), _silence(1.0)])
    a = _sentences(_run(sig, chunk=1600))
    b = _sentences(_run(sig, chunk=333))
    assert len(a) == len(b) == 1
    assert abs(a[0][1] - b[0][1]) < 0.05 and abs(a[0][2] - b[0][2]) < 0.05


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
