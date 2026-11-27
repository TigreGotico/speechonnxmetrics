"""Read the realised variant at every discriminative site and score the lects.

Each site is read as the lect whose candidate set holds the nearest realisation, or as
neither when the two are equally near. Three figures come out of the readings:

* **shares**: the percentage of sites read as each lect and as neither, overall and per
  pre-lexical and post-lexical group. This is the per-clip figure.
* **log-likelihood ratio**: the sum over sites of the measured log-ratio of each
  reading under the two lects, from a calibration table
  (:mod:`speechonnxmetrics.lect_fidelity.calibration`); a site class the calibration
  found unusable carries no weight. The clip decision compares it with the calibrated
  decision point.
* **posterior**: only for pooled clips (:func:`pool`), with the pooled duration beside
  it, because a few seconds of speech hold too few sites to support one.

A clip with too little realised speech returns no result rather than a verdict: fewer
than :data:`MIN_PHONES` realised segments, or fewer than :data:`MIN_COVERAGE` of the
expected count. So does a clip holding far more speech than its text, more than
:data:`MAX_COVERAGE` times the expected count, because its alignment to the text is
then arbitrary.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from speechonnxmetrics.lect_fidelity.phones import align, canonical, edit_distance, segment
from speechonnxmetrics.lect_fidelity.sites import POST_LEXICAL, PRE_LEXICAL, Expectation, Site

#: Two candidate distances closer than this leave a site read as neither lect.
TIE = 1e-6
#: The label of a site read as neither lect.
NEITHER = "neither"
#: Fewer realised segments than this is not speech enough to read sites from.
MIN_PHONES = 4
#: Fewer realised segments than this fraction of the expected count is not speech
#: enough either. On the real-speech evaluation sets the first percentile of the realised
#: count is 0.7 of the expected one for Allosaurus.
MIN_COVERAGE = 0.4
#: More realised segments than this multiple of the expected count is not speech of the
#: text alone. On the CLUL recordings the 99th percentile of the ratio is 1.09 to 1.24
#: across both backends, both providers and both decodings, and the largest on any
#: recognised clip is 1.77, so the bound turns away no real clip measured.
MAX_COVERAGE = 2.0


@dataclass(frozen=True)
class SiteReading:
    """What was realised at one site, its distance to each lect's nearest candidate,
    and the lect it is read as; ``lect`` is ``None`` when it is read as neither."""

    site: Site
    realised: tuple[str, ...]
    distances: dict[str, float]
    lect: str | None

    @property
    def label(self) -> str:
        return self.lect if self.lect is not None else NEITHER


def _shares(readings: Iterable[SiteReading], lects: Sequence[str]) -> dict[str, float]:
    counts = Counter(r.label for r in readings)
    total = sum(counts.values())
    return {k: (100.0 * counts[k] / total if total else 0.0) for k in (*lects, NEITHER)}


@dataclass(frozen=True)
class LectFidelityResult:
    """The site readings of one clip and the figures derived from them.

    ``log_likelihood_ratio`` is in favour of ``lects[0]`` and ``decision`` is the
    calibrated clip decision; both are ``None`` without a calibration table.
    """

    lects: tuple[str, str]
    readings: tuple[SiteReading, ...]
    realised: tuple[str, ...]
    expected: dict[str, tuple[str, ...]] = field(default_factory=dict)
    log_likelihood_ratio: float | None = None
    decision: str | None = None
    duration: float | None = None

    @property
    def n_sites(self) -> int:
        return len(self.readings)

    @property
    def n_decided(self) -> int:
        return sum(r.lect is not None for r in self.readings)

    @property
    def counts(self) -> dict[str, int]:
        counts = Counter(r.label for r in self.readings)
        return {k: counts[k] for k in (*self.lects, NEITHER)}

    @property
    def shares(self) -> dict[str, float]:
        """Percent of sites read as each lect and as neither; sums to 100."""
        return _shares(self.readings, self.lects)

    @property
    def shares_by_level(self) -> dict[str, dict[str, float]]:
        """:attr:`shares` for the pre-lexical and the post-lexical sites apart."""
        return {
            level: _shares([r for r in self.readings if r.site.level == level], self.lects)
            for level in (PRE_LEXICAL, POST_LEXICAL)
        }

    def fidelity(self, lect: str) -> float:
        """The share of sites read as ``lect``, 0 to 100."""
        return self.shares[lect]

    def report(self) -> list[dict]:
        """One plain dict per site, for logs and tables."""
        return [
            {
                "site": r.site.index,
                "class": r.site.cls,
                "level": r.site.level,
                "word": r.site.word,
                "context": r.site.context,
                **{
                    f"expected.{lect}": " | ".join(sorted("".join(v) or "∅" for v in r.site.variants[lect]))
                    for lect in self.lects
                },
                "realised": "".join(r.realised) or "∅",
                **{f"distance.{lect}": round(r.distances[lect], 4) for lect in self.lects},
                "reading": r.label,
            }
            for r in self.readings
        ]


def _realised_span(ops, span: tuple[int, int]) -> set[int]:
    """Indices of the realised segments aligned to an expected span.

    ``ops`` align realised (``a``) to expected (``b``). A realised segment belongs to
    the span when it is aligned to an expected segment inside it, or when it is an
    insertion lying between two expected positions at or inside the span's edges.
    """
    start, end = span
    out: set[int] = set()
    gap = 0
    for op in ops:
        if op.b is not None:
            if op.a is not None and start <= op.b < end:
                out.add(op.a)
            gap = op.b + 1
        elif op.a is not None and start <= gap <= end:
            out.add(op.a)
    return out


def _neighbours_realised(ops, span: tuple[int, int], length: int) -> bool:
    """Whether the expected segments just before and just after ``span`` were both
    realised, so that nothing realised inside it is evidence of a deletion."""
    start, end = span
    wanted = {i for i in (start - 1, end) if 0 <= i < length}
    if not wanted:
        return False
    hit = {op.b for op in ops if op.b in wanted and op.a is not None}
    return hit == wanted


def _nearest(seg: tuple[str, ...], candidates) -> float:
    return min(edit_distance(seg, c) for c in candidates)


def read_sites(realised: Sequence[str], expectation: Expectation) -> list[SiteReading]:
    """Align ``realised`` to each lect's primary reading and read every site."""
    realised = tuple(realised)
    lects = expectation.lects
    alignments = {lect: align(realised, expectation.phones[lect])[1] for lect in lects}
    readings = []
    for site in expectation.sites:
        idx: set[int] = set()
        for lect in lects:
            idx |= _realised_span(alignments[lect], site.spans[lect])
        seg = tuple(canonical(s) for s in realised[min(idx):max(idx) + 1]) if idx else ()
        dist = {lect: _nearest(seg, site.variants[lect]) for lect in lects}
        a, b = lects
        winner = None if abs(dist[a] - dist[b]) < TIE else min(dist, key=dist.get)
        if not seg and winner is not None and not all(
            _neighbours_realised(alignments[lect], site.spans[lect], len(expectation.phones[lect]))
            for lect in lects
        ):
            winner = None
        readings.append(SiteReading(site, seg, dist, winner))
    return readings


def enough_speech(realised: Sequence[str], expectation: Expectation) -> bool:
    """Whether ``realised`` holds enough segments to read sites from."""
    expected = min(len(p) for p in expectation.phones.values())
    return len(realised) >= max(MIN_PHONES, MIN_COVERAGE * expected)


def fits_text(realised: Sequence[str], expectation: Expectation) -> bool:
    """Whether ``realised`` holds no more segments than the text accounts for."""
    expected = min(len(p) for p in expectation.phones.values())
    return len(realised) <= MAX_COVERAGE * expected


def score_phones(
    realised: Sequence[str] | str,
    expectation: Expectation,
    calibration=None,
    duration: float | None = None,
) -> LectFidelityResult | None:
    """Score a realised phone string against an expectation.

    ``realised`` is a sequence of IPA segments or an IPA string, which is segmented.
    ``None`` when the text has no site, or too little or far too much was realised.
    ``calibration`` is a :class:`~speechonnxmetrics.lect_fidelity.calibration.Calibration`.
    """
    if not expectation.sites:
        return None
    phones = segment(realised) if isinstance(realised, str) else list(realised)
    if not enough_speech(phones, expectation) or not fits_text(phones, expectation):
        return None
    readings = tuple(read_sites(phones, expectation))
    llr = decision = None
    if calibration is not None:
        llr = calibration.log_likelihood_ratio(readings)
        decision = calibration.decide(llr)
    return LectFidelityResult(
        lects=expectation.lects,
        readings=readings,
        realised=tuple(phones),
        expected=dict(expectation.phones),
        log_likelihood_ratio=llr,
        decision=decision,
        duration=duration,
    )


@dataclass(frozen=True)
class PooledResult:
    """Site readings pooled over several clips of one voice.

    ``posterior`` is over the two lects, from the summed log-likelihood ratio and the
    calibrated temperature; ``duration`` is the pooled speech in seconds, the figure the
    posterior's strength depends on.
    """

    lects: tuple[str, str]
    n_clips: int
    duration: float
    shares: dict[str, float]
    shares_by_level: dict[str, dict[str, float]]
    log_likelihood_ratio: float | None
    posterior: dict[str, float] | None
    decision: str | None
    n_sites: int


def pool(results: Sequence[LectFidelityResult], calibration=None) -> PooledResult:
    """Pool clip results of one voice; clips without a result are left out by the
    caller. Shares count every site of every clip once."""
    results = [r for r in results if r is not None]
    if not results:
        raise ValueError("no clip results to pool")
    lects = results[0].lects
    readings = [s for r in results for s in r.readings]
    llr = posterior = decision = None
    if calibration is not None:
        llr = sum(calibration.log_likelihood_ratio(r.readings) for r in results)
        p = calibration.posterior(llr, n_clips=len(results))
        posterior = {lects[0]: p, lects[1]: 1.0 - p}
        decision = calibration.decide(llr, n_clips=len(results))
    return PooledResult(
        lects=lects,
        n_clips=len(results),
        duration=sum(r.duration or 0.0 for r in results),
        shares=_shares(readings, lects),
        shares_by_level={
            level: _shares([s for s in readings if s.site.level == level], lects)
            for level in (PRE_LEXICAL, POST_LEXICAL)
        },
        log_likelihood_ratio=llr,
        posterior=posterior,
        decision=decision,
        n_sites=len(readings),
    )
