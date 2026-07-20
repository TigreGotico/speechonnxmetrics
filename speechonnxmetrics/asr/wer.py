"""Word/character error-rate metrics via a single Levenshtein alignment.

Pure Python, no dependencies. ``compute()`` runs the alignment once and
derives WER, MER, WIL and WIP from the same substitution/deletion/insertion/
hit counts. ``align()`` exposes the underlying edit-op sequence for callers
that want to render a diff. Reference/hypothesis accept either a single
string or a list of strings (corpus mode); corpus mode aggregates counts
across the whole corpus and derives rates from the totals, it does not
average per-utterance rates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from speechonnxmetrics.asr.normalize import Normalizer

Text = str | list[str]


class EmptyReferenceError(ValueError):
    """Raised when the (tokenized) reference is empty — the error rate is undefined."""


@dataclass(frozen=True)
class Op:
    """One edit operation aligning reference[ref_start:ref_end] with hypothesis[hyp_start:hyp_end]."""

    tag: str  # "equal" | "sub" | "del" | "ins"
    ref_start: int
    ref_end: int
    hyp_start: int
    hyp_end: int


@dataclass(frozen=True)
class AsrMetrics:
    """Counts and derived rates from one word-level alignment (single utterance or aggregated corpus)."""

    hits: int
    substitutions: int
    deletions: int
    insertions: int
    ref_length: int
    hyp_length: int
    wer: float
    mer: float
    wil: float
    wip: float


def _as_corpus(value: Text, name: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise TypeError(f"{name} list must contain only strings")
        return value
    raise TypeError(f"{name} must be a str or a list[str], got {type(value).__name__}")


def _counts(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int, int]:
    """Levenshtein alignment counts (hits, substitutions, deletions, insertions), O(len(hyp)) memory."""
    n, m = len(ref), len(hyp)
    # row[j] = (cost, hits, subs, dels, inss) for the best alignment of ref[:i] to hyp[:j]
    row = [(j, 0, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        prev_row = row
        row = [(i, 0, 0, i, 0)] + [(0, 0, 0, 0, 0)] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost, hits, subs, dels, inss = prev_row[j - 1]
                row[j] = (cost, hits + 1, subs, dels, inss)
                continue
            del_cost = prev_row[j][0] + 1
            ins_cost = row[j - 1][0] + 1
            sub_cost = prev_row[j - 1][0] + 1
            best = min(del_cost, ins_cost, sub_cost)
            if best == del_cost:
                _, hits, subs, dels, inss = prev_row[j]
                row[j] = (best, hits, subs, dels + 1, inss)
            elif best == ins_cost:
                _, hits, subs, dels, inss = row[j - 1]
                row[j] = (best, hits, subs, dels, inss + 1)
            else:
                _, hits, subs, dels, inss = prev_row[j - 1]
                row[j] = (best, hits, subs + 1, dels, inss)
    _, hits, subs, dels, inss = row[m]
    return hits, subs, dels, inss


def _aggregate(
    reference: Text, hypothesis: Text, tokenize: Callable[[str], Sequence[str]], normalizer: Normalizer | None
) -> tuple[int, int, int, int]:
    refs = _as_corpus(reference, "reference")
    hyps = _as_corpus(hypothesis, "hypothesis")
    if isinstance(reference, str) != isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must both be str or both be list[str]")
    if len(refs) != len(hyps):
        raise ValueError(f"reference and hypothesis corpora must have equal length: {len(refs)} != {len(hyps)}")
    total_hits = total_subs = total_dels = total_inss = 0
    for ref, hyp in zip(refs, hyps):
        ref = normalizer(ref) if normalizer else ref
        hyp = normalizer(hyp) if normalizer else hyp
        hits, subs, dels, inss = _counts(tokenize(ref), tokenize(hyp))
        total_hits += hits
        total_subs += subs
        total_dels += dels
        total_inss += inss
    return total_hits, total_subs, total_dels, total_inss


def compute(reference: Text, hypothesis: Text, normalizer: Normalizer | None = None) -> AsrMetrics:
    """Word-level alignment counts plus WER/MER/WIL/WIP, all derived from one pass."""
    hits, subs, dels, inss = _aggregate(reference, hypothesis, str.split, normalizer)
    ref_length = hits + subs + dels
    if ref_length == 0:
        raise EmptyReferenceError("reference has no words; error rate is undefined")
    hyp_length = hits + subs + inss
    wer_ = (subs + dels + inss) / ref_length
    mer_ = (subs + dels + inss) / (subs + dels + inss + hits)
    wil_ = 1.0 if hyp_length == 0 else 1.0 - (hits / ref_length) * (hits / hyp_length)
    return AsrMetrics(
        hits=hits, substitutions=subs, deletions=dels, insertions=inss,
        ref_length=ref_length, hyp_length=hyp_length,
        wer=wer_, mer=mer_, wil=wil_, wip=1.0 - wil_,
    )


def wer(reference: Text, hypothesis: Text, normalizer: Normalizer | None = None) -> float:
    return compute(reference, hypothesis, normalizer).wer


def mer(reference: Text, hypothesis: Text, normalizer: Normalizer | None = None) -> float:
    return compute(reference, hypothesis, normalizer).mer


def wil(reference: Text, hypothesis: Text, normalizer: Normalizer | None = None) -> float:
    return compute(reference, hypothesis, normalizer).wil


def wip(reference: Text, hypothesis: Text, normalizer: Normalizer | None = None) -> float:
    return compute(reference, hypothesis, normalizer).wip


def cer(reference: Text, hypothesis: Text, normalizer: Normalizer | None = None) -> float:
    """Character error rate: (S+D+I)/N over characters instead of words."""
    hits, subs, dels, inss = _aggregate(reference, hypothesis, list, normalizer)
    ref_length = hits + subs + dels
    if ref_length == 0:
        raise EmptyReferenceError("reference has no characters; error rate is undefined")
    return (subs + dels + inss) / ref_length


def align(reference: str, hypothesis: str, normalizer: Normalizer | None = None) -> list[Op]:
    """Word-level edit-op sequence aligning reference to hypothesis, in reference order."""
    ref = normalizer(reference) if normalizer else reference
    hyp = normalizer(hypothesis) if normalizer else hypothesis
    ref_tok, hyp_tok = ref.split(), hyp.split()
    n, m = len(ref_tok), len(hyp_tok)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    backptr = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], backptr[i][0] = i, "del"
    for j in range(1, m + 1):
        cost[0][j], backptr[0][j] = j, "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tok[i - 1] == hyp_tok[j - 1]:
                cost[i][j], backptr[i][j] = cost[i - 1][j - 1], "equal"
                continue
            del_cost, ins_cost, sub_cost = cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + 1
            best = min(del_cost, ins_cost, sub_cost)
            cost[i][j] = best
            backptr[i][j] = "del" if best == del_cost else "ins" if best == ins_cost else "sub"
    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        tag = backptr[i][j]
        if tag in ("equal", "sub"):
            ops.append(Op(tag, i - 1, i, j - 1, j))
            i, j = i - 1, j - 1
        elif tag == "del":
            ops.append(Op("del", i - 1, i, j, j))
            i -= 1
        else:
            ops.append(Op("ins", i, i, j - 1, j))
            j -= 1
    ops.reverse()
    return ops
