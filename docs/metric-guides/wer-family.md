# WER, CER, MER, WIL and WIP
<!-- agent-hint
covers: what the five text metrics measure, their range, normalization and their call
source: speechonnxmetrics/asr/wer.py, speechonnxmetrics/asr/normalize.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/wer-family.md"
see: speaker-similarity.md
-->

These five metrics compare a hypothesis transcript with a reference transcript. They
take strings, not audio, so `score()` does not run them. Call them from
`speechonnxmetrics.asr`.

## Example

```python
from speechonnxmetrics import asr

ref = "the quick brown fox"
hyp = "the quick brown box"
print(asr.wer(ref, hyp))  # 0.25
print(asr.cer(ref, hyp))  # 0.05263157894736842
print(asr.compute(ref, hyp))  # AsrMetrics(hits=3, substitutions=1, deletions=0, insertions=0, ref_length=4, hyp_length=4, wer=0.25, mer=0.25, wil=0.4375, wip=0.5625)
print(asr.wer("The quick, brown fox!", "the quick brown fox", normalizer=asr.STRICT))  # 0.0
```

## History

All five use edit distance: the minimum number of substitutions, deletions and
insertions that change one sequence into another. Vladimir Levenshtein introduced it in
1965 and 1966. WER (word error rate) applies it to words. It is the standard score for
speech recognition, through the NIST evaluations and their tools. CER (character error
rate) applies it to characters.

Morris, Maier and Green introduced MER, WIL and WIP in *From WER and RIL to MER and WIL:
Improved Evaluation Measures for Connected Speech Recognition* (Interspeech 2004). They
found two problems with WER. It has no upper bound, and it does not treat deletions and
insertions the same. MER (match error rate) is the share of word slots that are errors.
WIL (word information lost) approximates the share of word information that is lost.
WIP (word information preserved) is `1 − WIL`.

## Range and direction

| Metric | Range | Direction |
| :--- | :--- | :--- |
| `wer` | 0 or more, can be more than 1 | lower is better |
| `cer` | 0 or more, can be more than 1 | lower is better |
| `mer` | 0 to 1 | lower is better |
| `wil` | 0 to 1 | lower is better |
| `wip` | 0 to 1 | higher is better |

`compute()` returns an `AsrMetrics` dataclass with the counts and all five rates in one
pass. `align()` returns the edit operations.

## When to use it

- Use WER by default for languages with words.
- Use CER for languages without clear word boundaries, for languages with rich
  morphology, or to give part credit for near misses.
- Use MER, WIL or WIP when you need a bound of 0 to 1, or the same weight for deletions
  and insertions.
- Do not compare scores across normalizers. This package never normalizes unless you
  pass `normalizer=`: `BASIC`, `STRICT` or `LEGACY`, which keeps only ASCII and is for
  migration only. See the [metrics reference](../metrics.md#asr-based-text-metrics).
  Case, punctuation and number format can move WER by points, so fix and record your
  normalizer.

## Further reading

Morris, Maier, Green, *From WER and RIL to MER and WIL*, Interspeech 2004. Levenshtein
(1966) for the edit distance.

---
[← Spectral distances](spectral.md) · [Home](../index.md) · [Speaker similarity →](speaker-similarity.md)
