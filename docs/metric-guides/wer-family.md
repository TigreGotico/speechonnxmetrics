# WER, CER, MER, WIL, WIP — ASR text metrics

Covers all five text metrics. They compare a hypothesis transcript against a reference
transcript; they take strings, not audio, so `score()` does not dispatch them — call them
from `speechonnxmetrics.asr`.

## History
All five are built on **edit distance** — the minimum number of substitutions, deletions and
insertions to turn one sequence into another — the idea Vladimir Levenshtein introduced in
1965/1966. **WER** (word error rate) applies it at the word level and became the standard
automatic-speech-recognition scoring measure, entrenched by the NIST benchmark evaluations
and their scoring tooling. **CER** (character error rate) is the same computation at the
character level.

**MER, WIL and WIP** come from Andrew Morris, Viktoria Maier and Phil Green, *From WER and
RIL to MER and WIL: Improved Evaluation Measures for Connected Speech Recognition*
(Interspeech 2004). They observed two problems with WER: it has **no upper bound** (so it
tells you one system beats another but not how good either is) and it is not symmetric
between deletions and insertions. **MER** (match error rate) is the proportion of word slots
that are errors; **WIL** (word information lost) approximates the fraction of word
information lost; **WIP** (word information preserved) is its complement, `1 − WIL`.

## What they measure
The gap between recognized text and reference text. WER/CER count edit operations relative to
reference length; MER/WIL/WIP recast that as a bounded, information-theoretic view.

## Range & direction
| metric | range | direction |
|---|---|---|
| `wer` | ≥ 0 (can exceed 1) | **lower is better** |
| `cer` | 0–1 | **lower is better** |
| `mer` | 0–1 | **lower is better** |
| `wil` | 0–1 | **lower is better** |
| `wip` | 0–1 | **higher is better** |

`compute()` returns an `AsrMetrics` dataclass with the raw counts (hits, substitutions,
deletions, insertions, lengths) and all five rates in one pass; `align()` returns the edit
operations.

## When to use / when not
- **WER** — the universal default for word-based languages.
- **CER** — better for languages without clear word boundaries, morphologically rich
  languages, or when you want partial credit for near-miss words.
- **MER / WIL / WIP** — when you need a **bounded [0,1]** measure (WER's lack of an upper
  bound is a real nuisance in dashboards) or a symmetric treatment of deletions/insertions.
- **Do not** compare scores across different **normalization** choices. This package
  *never normalizes on its own* — you pass `normalizer=` explicitly (`BASIC`, `STRICT`, or
  the ASCII-only `LEGACY`, which is migration-only; see [../metrics.md](../metrics.md)).
  Casing, punctuation and number formatting can swing WER by points, so fix and document
  your normalizer before comparing systems.

## Domain
Automatic speech recognition evaluation (and any transcript-vs-transcript comparison).

## In this library
```python
from speechonnxmetrics import asr
ref = "the quick brown fox"
hyp = "the quick brown box"
asr.wer(ref, hyp)                       # 0.25
asr.cer(ref, hyp)                       # per-character rate
asr.compute(ref, hyp)                   # AsrMetrics: all five rates + raw counts
asr.wer(ref, hyp, normalizer=asr.STRICT)  # opt-in normalization
```

## Further reading
Morris, Maier, Green, *From WER and RIL to MER and WIL…*, Interspeech 2004; Levenshtein
(1966) for the underlying edit distance.
