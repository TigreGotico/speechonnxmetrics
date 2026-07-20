"""Text normalizers applied before ASR scoring.

Every function is a pure ``str -> str`` transform. ``Normalizer`` chains a
configurable sequence of them; ``BASIC`` and ``STRICT`` are ready-made
presets. Scoring functions in :mod:`speechonnxmetrics.asr.wer` never
normalize on their own — callers opt in explicitly via ``normalizer=``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

_WHITESPACE_RE = re.compile(r"\s+")

_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "shan't": "shall not",
    "n't": " not", "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am",
}
_CONTRACTIONS_RE = re.compile("|".join(re.escape(k) for k in _CONTRACTIONS), re.IGNORECASE)

_FILLER_WORDS = {"um", "uh", "er", "ah", "hmm", "mhm", "uhh", "umm"}


def lowercase(text: str) -> str:
    return text.lower()


def strip_punctuation(text: str) -> str:
    return "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def remove_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def expand_contractions(text: str) -> str:
    return _CONTRACTIONS_RE.sub(lambda m: _CONTRACTIONS[m.group(0).lower()], text)


def strip_filler_words(text: str) -> str:
    return " ".join(w for w in text.split() if w.lower() not in _FILLER_WORDS)


@dataclass(frozen=True)
class Normalizer:
    """Chains normalizer callables in order, left to right."""

    steps: tuple[Callable[[str], str], ...]

    def __call__(self, text: str) -> str:
        for step in self.steps:
            text = step(text)
        return text


BASIC = Normalizer((lowercase, collapse_whitespace))
STRICT = Normalizer((
    lowercase,
    expand_contractions,
    remove_diacritics,
    strip_punctuation,
    strip_filler_words,
    collapse_whitespace,
))
