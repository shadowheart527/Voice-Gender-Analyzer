"""Standalone tests for voiceya/live/script_cursor.py.

Run: ``python tests/test_live_script_cursor.py`` from the repo root.
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voiceya.live.script_cursor import ScriptCursor, tokenize  # noqa: E402

RAINBOW = (
    "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. "
    "The rainbow is a division of white light into many beautiful colors. "
    "These take the shape of a long round arch, with its path high above, "
    "and its two ends apparently beyond the horizon."
)


def test_tokenize_en_keeps_spans_and_apostrophes():
    toks = tokenize("It's a Test, isn't it?", "en-US")
    assert [t.norm for t in toks] == ["it's", "a", "test", "isn't", "it"]
    assert "It's a Test, isn't it?"[toks[0].start : toks[0].end] == "It's"


def test_tokenize_zh_per_char():
    toks = tokenize("你好，世界", "zh-CN")
    assert [t.norm for t in toks] == ["你", "好", "世", "界"]


def test_exact_first_sentence_returns_canonical_text():
    c = ScriptCursor(RAINBOW, "en-US")
    r = c.consume(
        "when the sunlight strikes raindrops in the air they act as a prism and form a rainbow"
    )
    assert r.matched
    assert (
        r.text
        == "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow"
    )
    assert c.cursor == 17


def test_asr_errors_still_match_and_advance():
    c = ScriptCursor(RAINBOW, "en-US")
    c.consume(
        "when the sunlight strikes raindrops in the air they act as a prism and form a rainbow"
    )
    r = c.consume("the rainbow is a division of white lite into many beautiful colours")
    assert r.matched, r
    assert r.text.startswith("The rainbow is a division")
    assert r.text.endswith("colors")
    assert c.cursor == 29, c.cursor


def test_partial_sentence_then_rest():
    c = ScriptCursor(RAINBOW, "en-US")
    r1 = c.consume("when the sunlight strikes raindrops")
    assert r1.matched and r1.text == "When the sunlight strikes raindrops"
    r2 = c.consume("in the air they act as a prism and form a rainbow")
    assert r2.matched and r2.text == "in the air, they act as a prism and form a rainbow"


def test_skipped_sentence_is_found_ahead():
    c = ScriptCursor(RAINBOW, "en-US")
    c.consume(
        "when the sunlight strikes raindrops in the air they act as a prism and form a rainbow"
    )
    # Reader skips the second sentence entirely.
    r = c.consume("these take the shape of a long round arch")
    assert r.matched, r
    assert r.text == "These take the shape of a long round arch"


def test_restart_from_beginning_uses_global_search():
    c = ScriptCursor(RAINBOW, "en-US")
    c.consume("these take the shape of a long round arch with its path high above")
    r = c.consume("when the sunlight strikes raindrops in the air")
    assert r.matched and r.text.startswith("When the sunlight")
    assert c.cursor == 8


def test_off_script_falls_back_to_asr_text():
    c = ScriptCursor(RAINBOW, "en-US")
    r = c.consume("testing one two three can you hear me")
    assert not r.matched
    assert r.text == "testing one two three can you hear me"
    assert c.cursor == 0


def test_empty_inputs():
    c = ScriptCursor("", "en-US")
    r = c.consume("hello there")
    assert not r.matched and r.text == "hello there"
    c2 = ScriptCursor(RAINBOW, "en-US")
    assert not c2.consume("").matched


def test_zh_char_level_matching():
    script = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
    c = ScriptCursor(script, "zh-CN")
    r = c.consume("春眠不觉晓处处闻啼鸟")
    assert r.matched and r.text == "春眠不觉晓，处处闻啼鸟"
    r2 = c.consume("夜来风雨声花落知多少")
    assert r2.matched and r2.text == "夜来风雨声，花落知多少"
    assert c.done


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
