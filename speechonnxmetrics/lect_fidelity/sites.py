"""The sites of a text where two lects are expected to differ.

The primary readings of the two lects are aligned word by word. A site is one alignment
step where the two differ, with identical segments, or a word edge, on both sides of
it. Each site must belong to a documented class of the lect pair
(:data:`SITE_CLASSES`); a difference outside every class is not a site. Two kinds of
difference are excluded rather than scored:

* **fused** differences, two or more adjacent steps that differ, such as final ``-de``
  read ``dɨ`` against ``d͡ʒi``: the affricate there exists only because the vowel
  raised, so scoring both would count one process twice, and scoring either alone would
  depend on how the recogniser splits the pair;
* **overlapping** sites, whose candidate sets share a member: a realisation one lect
  licenses as readily as the other carries no evidence.

A site's candidate set in a lect is its primary segment, the free variants of that
segment, and whatever each other reading of the word has at that position, so an
optional deletion puts the empty sequence ``()`` in the set.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from speechonnxmetrics.lect_fidelity.phones import align, canonical
from speechonnxmetrics.lect_fidelity.providers import PhoneProvider, WordPhones, get_provider

PRE_LEXICAL = "pre-lexical"
POST_LEXICAL = "post-lexical"


@dataclass(frozen=True)
class SiteClass:
    """A documented difference between a European and a Brazilian lect.

    ``european`` and ``brazilian`` hold the primary segments each side writes at the
    site, ``""`` for none; ``before`` constrains the segment that follows the site.
    """

    name: str
    level: str
    european: frozenset[str]
    brazilian: frozenset[str]
    before: frozenset[str] | None = None


#: The site classes of European against Brazilian Portuguese: coda s, unstressed e and
#: o, t and d before i, and coda l. Unstressed reduction is pre-lexical in both lects;
#: the coda sibilant, dark coda l and affrication are post-lexical allophony.
SITE_CLASSES: tuple[SiteClass, ...] = (
    SiteClass("coda_s", POST_LEXICAL, frozenset({"ʃ", "ʒ"}), frozenset({"s", "z"})),
    SiteClass("unstressed_e", PRE_LEXICAL, frozenset({"ɨ", "ə", ""}), frozenset({"i", "ɪ", "e"})),
    SiteClass("unstressed_o", PRE_LEXICAL, frozenset({"u"}), frozenset({"o"})),
    SiteClass(
        "td_before_i", POST_LEXICAL, frozenset({"t", "d"}), frozenset({"tʃ", "dʒ"}),
        before=frozenset({"i", "ɪ", "ĩ", "j"}),
    ),
    SiteClass("coda_l", POST_LEXICAL, frozenset({"ɫ", "l"}), frozenset({"w", "u", "ʊ"})),
)
CLASS_LEVEL = {c.name: c.level for c in SITE_CLASSES}

_EUROPEAN = re.compile(r"^pt-PT(-|$)")
_BRAZILIAN = re.compile(r"^pt-BR(-|$)")


def roles(lects: Sequence[str]) -> tuple[str, str]:
    """``lects`` as ``(european, brazilian)``; the site classes are documented only for
    a European Portuguese lect against a Brazilian one."""
    eu = [lect for lect in lects if _EUROPEAN.match(lect)]
    br = [lect for lect in lects if _BRAZILIAN.match(lect)]
    if len(lects) != 2 or len(eu) != 1 or len(br) != 1:
        raise ValueError(
            f"site classes are documented for one pt-PT lect against one pt-BR lect, got {tuple(lects)!r}"
        )
    return eu[0], br[0]


def classify(european: str, brazilian: str, following: str | None) -> SiteClass | None:
    """The class of a one-step difference, from each side's segment (``""`` for none)
    and the segment after the site, or ``None`` when no class documents it."""
    eu, br = canonical(european), canonical(brazilian)
    for cls in SITE_CLASSES:
        if eu in cls.european and br in cls.brazilian:
            if cls.before is not None and (following is None or canonical(following) not in cls.before):
                continue
            return cls
    return None


@dataclass(frozen=True)
class Site:
    """One discriminative site.

    ``variants`` holds each lect's candidate set, sequences of canonical segments with
    ``()`` for none; ``spans`` holds the half-open range of the site in each lect's
    primary phone string, all words concatenated.
    """

    index: int
    word_index: int
    word: str
    context: str
    cls: str
    level: str
    variants: dict[str, frozenset[tuple[str, ...]]]
    spans: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class Expectation:
    """The expected phones of one text for a pair of lects, and their sites.

    ``phones`` is each lect's primary reading; ``excluded`` counts the differences that
    are not sites, by reason (``fused``, ``overlapping``, ``unclassified``).
    """

    text: str
    lects: tuple[str, str]
    provider: str
    phones: dict[str, tuple[str, ...]]
    sites: tuple[Site, ...]
    excluded: dict[str, int] = field(default_factory=dict)


def _context(ortho: Sequence[str], k: int) -> str:
    left = " ".join(ortho[max(0, k - 2):k])
    right = " ".join(ortho[k + 1:k + 3])
    return " ".join(p for p in (left, f"[{ortho[k]}]", right) if p)


def _runs(ops, a: Sequence[str], b: Sequence[str]) -> list[list]:
    runs, run = [], []
    for op in ops:
        if op.a is not None and op.b is not None and canonical(a[op.a]) == canonical(b[op.b]):
            if run:
                runs.append(run)
            run = []
        else:
            run.append(op)
    if run:
        runs.append(run)
    return runs


def _span(op, side: str, ops, length: int) -> tuple[int, int]:
    index = getattr(op, side)
    if index is not None:
        return index, index + 1
    later = [getattr(o, side) for o in ops[ops.index(op) + 1:] if getattr(o, side) is not None]
    start = later[0] if later else length
    return start, start


def _candidates(word: WordPhones, span: tuple[int, int]) -> frozenset[tuple[str, ...]]:
    primary = word.primary
    start, end = span
    out = {tuple(canonical(s) for s in primary[start:end])}
    for reading in word.readings[1:]:
        _, ops = align(primary, reading)
        taken, gap = [], 0
        for op in ops:
            if op.a is not None:
                if op.b is not None and start <= op.a < end:
                    taken.append(reading[op.b])
                gap = op.a + 1
            elif start == end == gap:
                taken.append(reading[op.b])
        out.add(tuple(canonical(s) for s in taken))
    for seq in list(out):
        if len(seq) == 1 and seq[0] in word.variants:
            out |= {(v,) if v else () for v in word.variants[seq[0]]}
    return frozenset(out)


def expect(
    text: str, lects: Sequence[str], provider: str | PhoneProvider = "orthography2ipa"
) -> Expectation:
    """The expected phones of ``text`` in both lects and its discriminative sites."""
    lects = tuple(lects)
    if len(lects) != 2 or lects[0] == lects[1]:
        raise ValueError(f"expected two distinct lects, got {lects!r}")
    european, _ = roles(lects)
    provider = get_provider(provider)
    a, b = lects
    words = {lect: provider.candidates(text, lect) for lect in lects}
    phones = {lect: tuple(s for w in words[lect] for s in w.primary) for lect in lects}
    if [w.word for w in words[a]] != [w.word for w in words[b]]:
        return Expectation(text, lects, provider.name, phones, (), {"word_mismatch": 1})
    ortho = [w.word for w in words[a]]
    sites: list[Site] = []
    excluded: Counter = Counter()
    offset = {a: 0, b: 0}
    for k, (wa, wb) in enumerate(zip(words[a], words[b])):
        pa, pb = wa.primary, wb.primary
        _, ops = align(pa, pb)
        for run in _runs(ops, pa, pb):
            if len(run) > 1:
                excluded["fused"] += 1
                continue
            op = run[0]
            seg = {a: pa[op.a] if op.a is not None else "", b: pb[op.b] if op.b is not None else ""}
            span = {a: _span(op, "a", ops, len(pa)), b: _span(op, "b", ops, len(pb))}
            following = pa[span[a][1]] if span[a][1] < len(pa) else None
            brazilian = b if european == a else a
            cls = classify(seg[european], seg[brazilian], following)
            if cls is None:
                excluded["unclassified"] += 1
                continue
            variants = {a: _candidates(wa, span[a]), b: _candidates(wb, span[b])}
            if variants[a] & variants[b]:
                excluded["overlapping"] += 1
                continue
            sites.append(Site(
                index=len(sites),
                word_index=k,
                word=ortho[k],
                context=_context(ortho, k),
                cls=cls.name,
                level=cls.level,
                variants=variants,
                spans={lect: (offset[lect] + span[lect][0], offset[lect] + span[lect][1]) for lect in lects},
            ))
        offset[a] += len(pa)
        offset[b] += len(pb)
    return Expectation(text, lects, provider.name, phones, tuple(sites), dict(excluded))


def discriminative_sites(
    text: str, lects: Sequence[str], provider: str | PhoneProvider = "orthography2ipa"
) -> tuple[Site, ...]:
    """The sites of ``text`` where the expected phones of the two lects differ."""
    return expect(text, lects, provider).sites
