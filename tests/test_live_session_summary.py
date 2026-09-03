"""Standalone tests for voiceya/live/session.py aggregation (no models, no sidecar).

Run: ``python tests/test_live_session_summary.py`` from the repo root.
Feeds hand-built per-sentence outputs into a LiveSession and checks that
``_build_result()`` produces the batch schema with absolute times and
running aggregates.
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voiceya.live.session import LiveSession  # noqa: E402
from voiceya.services.audio_analyser.seg_analyser import AnalyseResultItem  # noqa: E402


async def _noop_send(_payload: dict) -> None:
    return None


def _phone(start: float, label: str, pitch: float, reso: float, vowel: bool = True) -> dict:
    return {
        "start": start,
        "end": start + 0.08,
        "char": "x",
        "phone": label,
        "is_vowel": vowel,
        "pitch": pitch,
        "resonance": reso,
        "F1": 500.0,
        "F2": 1800.0,
        "F3": 2900.0,
        "z_F1": 0.1,
        "z_F2": -0.2,
        "z_F3": 0.0,
    }


async def _session(
    mode: str = "script", script: str | None = "When the sunlight strikes"
) -> LiveSession:
    s = LiveSession(language="en-US", mode=mode, script=script, send=_noop_send)
    # Stop the idle worker so the test does not leave a pending task.
    await s._queue.put(None)
    await s._worker
    return s


def _seed(s: LiveSession) -> None:
    # Sentence 1: 0.7-3.0 s, sentence 2: 4.0-6.5 s (already absolute, as _process would store them).
    s.sentences = [
        {"index": 0, "start": 0.7, "end": 3.0, "transcript": "When the sunlight", "engine_c": True},
        {"index": 1, "start": 4.0, "end": 6.5, "transcript": "strikes raindrops", "engine_c": True},
    ]
    s.transcripts = ["When the sunlight", "strikes raindrops"]
    s.phones = [
        _phone(0.9, "IY1", 210.0, 0.70),
        _phone(1.1, "AH0", 200.0, 0.60),
        _phone(1.3, "IY1", 215.0, 0.72),
        _phone(4.2, "IY1", 220.0, 0.74),
        _phone(4.5, "AH0", 205.0, 0.58),
        _phone(4.8, "T", None, None, vowel=False),
    ]
    s.word_count = 5
    s.ceilings.update([5500, 5500, 6500])
    s.analyse_results = [
        AnalyseResultItem(
            label="female", start_time=0.7, end_time=3.0, duration=2.3, confidence=0.9
        ),
        AnalyseResultItem(
            label="female", start_time=4.0, end_time=6.5, duration=2.5, confidence=0.8
        ),
    ]
    s.voiced_f0 = np.array([200.0, 205.0, 210.0, 215.0, 220.0] * 20)
    # Pretend 7 s of audio went through the segmenter.
    s.seg._frame_index = int(7.0 / s.seg.frame_s)


def test_result_has_batch_shape():
    s = asyncio.run(_session())
    _seed(s)
    r = s._build_result()
    assert r["status"] == "success" and r["filename"] == "live"
    assert set(r) >= {"summary", "analysis"}
    summ = r["summary"]
    for k in (
        "total_female_time_sec",
        "overall_f0_median_hz",
        "dominant_label",
        "engine_c",
        "advice",
    ):
        assert k in summ, k
    assert summ["dominant_label"] == "female"
    assert len(r["analysis"]) == 2


def test_engine_c_aggregates():
    s = asyncio.run(_session())
    _seed(s)
    ec = s._build_result()["summary"]["engine_c"]
    assert ec["phone_count"] == 6 and ec["word_count"] == 5
    assert ec["transcript"] == "When the sunlight strikes raindrops"
    assert ec["mode"] == "script" and ec["script"] == "When the sunlight strikes"
    assert ec["formant_ceiling_hz"] == 5500
    assert ec["silence_ranges"] == [{"start": 3.0, "end": 4.0}]
    assert ec["median_pitch_hz"] == 210.0
    vowels = {v["vowel"]: v for v in ec["resonance_per_vowel"]}
    assert vowels["IY"]["n"] == 3 and vowels["AH"]["n"] == 2
    assert "T" not in vowels  # no resonance score → not aggregated
    assert ec["resonance_zone_key"] is not None
    assert ec["alignment_confidence"]["phone_ratio"] is not None


def test_f0_panel_is_running_over_session():
    s = asyncio.run(_session())
    _seed(s)
    adv = s._build_result()["summary"]["advice"]
    f0 = adv["f0_panel"]
    assert f0["reliability"] == "ok"
    # prefer_praat_median swaps the median for the phone-midpoint median (210).
    assert f0["median_hz"] == 210.0
    assert f0["p25_hz"] == 205.0 and f0["p75_hz"] == 215.0
    assert adv["gating_tier"] == "minimal"  # 7 s so far


def test_empty_session_is_still_well_formed():
    s = asyncio.run(_session(mode="free", script=None))
    r = s._build_result()
    assert r["summary"]["engine_c"] is None
    assert r["summary"]["dominant_label"] is None
    assert r["summary"]["advice"]["f0_panel"]["reliability"] in (
        "short_recording",
        "insufficient_voiced",
    )


def test_free_mode_ignores_script():
    s = asyncio.run(_session(mode="free", script="ignored"))
    assert s.mode == "free" and s.cursor is None and s.script is None


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
