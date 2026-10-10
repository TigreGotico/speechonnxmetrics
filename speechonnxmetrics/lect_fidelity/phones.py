"""IPA segmentation, phone distance and weighted edit-distance alignment.

Expected phones (from ``orthography2ipa``) and realised phones (from a recogniser) are
cut into segments by one rule, so both sides of every comparison use the same units:
a segment is one base character with the combining marks and modifier letters that
follow it, and a tie bar joins the next base character into the same segment. Stress
marks, syllable dots and linking marks carry no segment and are removed. The Portuguese
affricates written without a tie bar are joined too (:data:`AFFRICATES`).
"""
from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

#: Prosodic marks and boundaries, removed before segmentation.
_PROSODIC = frozenset("ˈˌ.‿|‖-")
_TIE_BARS = "͜͡"
#: The cost of inserting or deleting one segment; a substitution costs at most 1.
INDEL = 1.0
#: Metric decision: a stop followed by the fricative it releases into is read as one
#: affricate. Lexicon transcriptions and recognisers often write ``dʒ`` where the
#: lattice writes ``d͡ʒ``; a Portuguese cluster of the two does not contrast with it.
AFFRICATES = {("t", "ʃ"): "t͡ʃ", ("d", "ʒ"): "d͡ʒ"}


def segment(ipa: str) -> list[str]:
    """Cut an IPA string into segments; whitespace and prosodic marks are dropped."""
    segments: list[str] = []
    joining = False
    for ch in unicodedata.normalize("NFD", ipa):
        if ch.isspace() or ch in _PROSODIC:
            joining = False
            continue
        attaches = unicodedata.category(ch) in ("Mn", "Lm") or ch in _TIE_BARS
        if segments and (attaches or joining):
            segments[-1] += ch
            joining = ch in _TIE_BARS
        else:
            segments.append(ch)
            joining = False
    out: list[str] = []
    for seg in (unicodedata.normalize("NFC", s) for s in segments):
        if out and (out[-1], seg) in AFFRICATES:
            out[-1] = AFFRICATES[out[-1], seg]
        else:
            out.append(seg)
    return out


def canonical(seg: str) -> str:
    """The form two segments are compared in: NFC, without tie bars."""
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", seg) if c not in _TIE_BARS))


#: Glides and the vowels they are the non-syllabic counterparts of. Recognisers often
#: write a glide as its vowel (``w`` as ``u``), and a feature distance alone would put
#: the two in different major classes.
_GLIDE_VOWEL = {"j": "i", "w": "u", "ɥ": "y", "ɰ": "ɯ", "j̃": "ĩ", "w̃": "ũ"}
#: Added for each step of backing off: a glide read as its vowel, or a segment whose
#: diacritics the feature table does not know.
BACKOFF = 0.1


@lru_cache(maxsize=4096)
def _known(seg: str) -> bool:
    from orthography2ipa.feats import vectorize_phones

    try:
        vectorize_phones(seg)
    except (ValueError, KeyError, IndexError):
        return False
    return True


def _strip(seg: str) -> str:
    """``seg`` without combining marks and modifier letters; the base character alone."""
    base = "".join(c for c in unicodedata.normalize("NFD", seg) if unicodedata.category(c) not in ("Mn", "Lm"))
    return unicodedata.normalize("NFC", base) or seg


def _feature_distance(a: str, b: str) -> float:
    """Feature distance, comparing an unknown segment through its base character."""
    from orthography2ipa.distance import segment_distance

    penalty = 0.0
    if not (_known(a) and _known(b)):
        a = a if _known(a) else _strip(a)
        b = b if _known(b) else _strip(b)
        penalty = BACKOFF
    if a == b:
        return penalty
    if not (_known(a) and _known(b)):
        return 1.0
    return min(1.0, penalty + float(segment_distance(a, b)))


@lru_cache(maxsize=65536)
def distance(a: str, b: str) -> float:
    """Phonetic distance between two segments in ``[0, 1]``, from distinctive features.

    The features come from ``orthography2ipa``. A segment the feature table does not
    know is compared through its base character, and a glide is also compared with a
    vowel as the glide's vowel; each such step adds :data:`BACKOFF`.
    """
    a, b = canonical(a), canonical(b)
    if a == b:
        return 0.0
    best = _feature_distance(a, b)
    if (a in _GLIDE_VOWEL) != (b in _GLIDE_VOWEL):
        a2, b2 = _GLIDE_VOWEL.get(a, a), _GLIDE_VOWEL.get(b, b)
        best = min(best, BACKOFF + _feature_distance(a2, b2))
    return min(1.0, best)


@dataclass(frozen=True)
class Op:
    """One alignment step: ``a``/``b`` index the two sequences, ``None`` for a gap."""

    a: int | None
    b: int | None
    cost: float


def align(a: Sequence[str], b: Sequence[str]) -> tuple[float, list[Op]]:
    """Weighted edit-distance alignment of two segment sequences.

    Substitution costs :func:`distance`, insertion and deletion :data:`INDEL`. Ties in
    the backtrace prefer a substitution, then a deletion from ``a``, then an insertion.
    """
    n, m = len(a), len(b)
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i * INDEL
    for j in range(1, m + 1):
        d[0][j] = j * INDEL
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j - 1] + distance(a[i - 1], b[j - 1]),
                d[i - 1][j] + INDEL,
                d[i][j - 1] + INDEL,
            )
    ops: list[Op] = []
    i, j = n, m
    while i or j:
        if i and j:
            sub = distance(a[i - 1], b[j - 1])
            if abs(d[i][j] - (d[i - 1][j - 1] + sub)) < 1e-9:
                ops.append(Op(i - 1, j - 1, sub))
                i, j = i - 1, j - 1
                continue
        if i and abs(d[i][j] - (d[i - 1][j] + INDEL)) < 1e-9:
            ops.append(Op(i - 1, None, INDEL))
            i -= 1
        else:
            ops.append(Op(None, j - 1, INDEL))
            j -= 1
    ops.reverse()
    return d[n][m], ops


def edit_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """The cost of :func:`align`."""
    return align(a, b)[0]
