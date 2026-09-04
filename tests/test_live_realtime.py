"""Standalone tests for the pure parts of voiceya/live/realtime.py.

Run: ``python tests/test_live_realtime.py`` from the repo root.  No models:
covers the stats lookup, per-phone scoring and frame→phone segmentation.
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voiceya.live.realtime import (  # noqa: E402
    CHARSIU_VOCAB,
    FRAME_S,
    PhoneSegmenter,
    load_stats,
    score_phone,
)


def test_stats_are_keyed_without_stress_and_cover_vocab():
    stats, weights = load_stats("en")
    assert "IY" in stats and "IY1" not in stats
    assert abs(sum(weights) - 1.0) < 0.01
    phones = {v for v in CHARSIU_VOCAB.values() if not v.startswith("[")}
    missing = sorted(p for p in phones if p not in stats)
    # A few consonants have no calibration entry; every vowel must.
    vowels = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
    assert not (vowels & set(missing)), missing


def test_score_phone_matches_formula():
    stats = {"IY": [{}, {"mean": 400.0, "stdev": 50.0}, {"mean": 2500.0, "stdev": 250.0}, {"mean": 3000.0, "stdev": 300.0}]}
    w = [0.5, 0.5, 0.0]
    r = score_phone("IY", 450.0, 2250.0, 3300.0, stats, w)
    assert r["z_F1"] == 1.0 and r["z_F2"] == -1.0 and r["z_F3"] == 1.0
    assert abs(r["resonance"] - 0.5) < 1e-9  # 0.5 + 0.5*1 + 0.5*(-1)
    r2 = score_phone("IY", 500.0, 2750.0, None, stats, w)
    assert abs(r2["resonance"] - 1.0) < 1e-9  # clamped: 0.5 + 1 + 0.5
    assert score_phone("IY", None, 2500.0, 3000.0, stats, w)["resonance"] is None
    assert score_phone("ZZ", 1, 2, 3, stats, w)["resonance"] is None


def _acoustic(frames):
    return {int(round(t / FRAME_S)): vals for t, vals in frames}


def test_segmenter_closes_on_label_change_with_middle_formants():
    stats = {"IY": [{}, {"mean": 400.0, "stdev": 50.0}, {"mean": 2500.0, "stdev": 250.0}, {"mean": 3000.0, "stdev": 300.0}]}
    seg = PhoneSegmenter(stats, [0.5, 0.5, 0.0])
    # 8 frames of IY (0.10..0.17), then T.  Edge frames carry wild formants
    # that the middle-50 % median must ignore.
    ac = {}
    times = [round(0.10 + i * FRAME_S, 3) for i in range(8)]
    for i, t in enumerate(times):
        edge = i in (0, 1, 6, 7)
        ac[int(round(t / FRAME_S))] = (200.0, 900.0 if edge else 400.0, 900.0 if edge else 2500.0, 3000.0)
    closed = []
    for t in times:
        c = seg.commit(t, "IY", ac)
        assert c is None
    c = seg.commit(0.18, "T", ac)
    assert c is not None and c["phone"] == "IY" and c["is_vowel"]
    assert c["start"] == 0.10 and c["end"] == 0.18
    assert c["F1"] == 400.0 and c["F2"] == 2500.0 and c["pitch"] == 200.0
    assert abs(c["resonance"] - 0.5) < 1e-9
    closed.append(c)
    # Silence closes T; T has no stats → no resonance but still a phone.
    c2 = seg.commit(0.25, "[SIL]", ac)
    assert c2 is not None and c2["phone"] == "T" and c2["resonance"] is None
    # Silence itself never becomes a phone.
    assert seg.commit(0.40, "AH", ac) is None
    assert seg.flush(0.45, ac)["phone"] == "AH"


def test_segmenter_drops_too_short_phones():
    seg = PhoneSegmenter({}, [0.5, 0.5, 0.0])
    seg.commit(0.10, "K", {})
    seg.commit(0.11, "K", {})
    assert seg.commit(0.12, "AH", {}) is None  # 20 ms < MIN_PHONE_S
    for t in (0.13, 0.14, 0.15, 0.16):
        seg.commit(t, "AH", {})
    c = seg.commit(0.17, "[SIL]", {})
    assert c is not None and c["phone"] == "AH" and c["F1"] is None


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
