"""Map recognised speech onto the next span of a known script.

In script mode the batch path hands the whole script to the aligner.  Live
sentences are shorter than the script, so we need to know *which* words this
sentence covered.  Whisper tells us roughly what was said; ``ScriptCursor``
finds the best-matching contiguous span of the script near the current
position and returns the script's own text for that span (canonical spelling
and punctuation, so the aligner's dictionary lookups stay clean).

Fallbacks, in order: local window → whole script (reader restarted) → the ASR
text itself (reader went off-script).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_CJK_RE = re.compile(r"[一-鿿가-힣]")
_CHAR_LANGS = {"zh", "ko"}


@dataclass(frozen=True)
class _Tok:
    norm: str
    start: int  # char offset into the original text
    end: int


def _short_lang(language: str) -> str:
    return (language or "en").split("-")[0].lower()


def _norm_word(w: str) -> str:
    w = unicodedata.normalize("NFKC", w).lower().replace("’", "'")
    return w


def tokenize(text: str, language: str) -> list[_Tok]:
    """Tokens with character spans.  zh/ko: one token per CJK/Hangul char."""
    lang = _short_lang(language)
    if lang in _CHAR_LANGS:
        return [_Tok(m.group(0), m.start(), m.end()) for m in _CJK_RE.finditer(text)]
    return [_Tok(_norm_word(m.group(0)), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


@dataclass
class MatchResult:
    text: str
    matched: bool
    start_token: int | None = None
    end_token: int | None = None  # exclusive
    ratio: float = 0.0


class ScriptCursor:
    def __init__(
        self,
        script: str,
        language: str,
        *,
        local_min_ratio: float = 0.5,
        global_min_ratio: float = 0.6,
        lookahead_factor: float = 2.0,
        lookahead_pad: int = 8,
        lookbehind: int = 3,
    ) -> None:
        self.script = script or ""
        self.language = language
        self.tokens = tokenize(self.script, language)
        self.cursor = 0  # next unread token index
        self.local_min_ratio = local_min_ratio
        self.global_min_ratio = global_min_ratio
        self.lookahead_factor = lookahead_factor
        self.lookahead_pad = lookahead_pad
        self.lookbehind = lookbehind

    # ── public ─────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.tokens)

    @property
    def progress(self) -> float:
        return 0.0 if not self.tokens else min(1.0, self.cursor / len(self.tokens))

    def consume(self, asr_text: str) -> MatchResult:
        asr = [t.norm for t in tokenize(asr_text or "", self.language)]
        if not asr or not self.tokens:
            return MatchResult(text=(asr_text or "").strip(), matched=False)

        lo = max(0, self.cursor - self.lookbehind)
        hi = min(
            len(self.tokens),
            self.cursor + int(len(asr) * self.lookahead_factor) + self.lookahead_pad,
        )
        res = self._match(asr, lo, hi, self.local_min_ratio)
        if res is None:
            res = self._match(asr, 0, len(self.tokens), self.global_min_ratio)
        if res is None:
            return MatchResult(text=asr_text.strip(), matched=False)

        start_tok, end_tok, ratio = res
        text = self.script[self.tokens[start_tok].start : self.tokens[end_tok - 1].end]
        self.cursor = end_tok
        return MatchResult(
            text=text, matched=True, start_token=start_tok, end_token=end_tok, ratio=ratio
        )

    # ── internals ───────────────────────────────────────────────────

    def _match(self, asr: list[str], lo: int, hi: int, min_ratio: float):
        window = [t.norm for t in self.tokens[lo:hi]]
        if not window:
            return None
        sm = SequenceMatcher(None, asr, window, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
        if not blocks:
            return None
        matched = sum(b.size for b in blocks)
        ratio = matched / len(asr)
        if ratio < min_ratio:
            return None
        first, last = blocks[0], blocks[-1]
        # Extend the span to cover ASR tokens before the first / after the last
        # matched block (a misrecognised first word still consumed script text),
        # clamped to the window.
        start = max(lo, lo + first.b - first.a)
        tail_unmatched = len(asr) - (last.a + last.size)
        end = min(hi, lo + last.b + last.size + tail_unmatched)
        if end <= start:
            return None
        return start, end, ratio
