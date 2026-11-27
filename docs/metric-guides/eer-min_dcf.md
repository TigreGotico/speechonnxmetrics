# EER and minDCF
<!-- agent-hint
covers: what eer, min_dcf and equal_error_threshold measure, their range, when to use them and their call
source: speechonnxmetrics/speaker/similarity.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/eer-min_dcf.md"
see: ../index.md
-->

These metrics score a speaker verification system over many trials. Each trial has a
similarity score and a label, genuine or impostor. This page covers `eer`, `min_dcf`
and the helper `equal_error_threshold`.

## Example

```python
from speechonnxmetrics.speaker import eer, equal_error_threshold, min_dcf

scores = [0.9, 0.8, 0.4, 0.3, 0.6, 0.2]
labels = [1, 1, 1, 0, 0, 0]  # 1 is genuine, 0 is impostor
print(eer(scores, labels))  # 0.33333333333333337
print(min_dcf(scores, labels, p_target=0.01))  # 0.33333333333333337
print(equal_error_threshold(scores, labels))  # 0.6
```

The functions are numpy code. But `speechonnxmetrics.speaker` imports `speakeronnx`,
so the import needs the `speaker` extra ([issue #7](https://github.com/TigreGotico/speechonnxmetrics/issues/7)).

## History

Both metrics come from speaker verification and the NIST Speaker Recognition
Evaluation (SRE) series. Verification is a detection problem. Each trial asks "same
speaker or not?". A decision threshold trades false alarms (an impostor accepted)
against misses (a genuine speaker refused).

- EER (equal error rate) is the point where the false-alarm rate equals the miss rate.
  It gives one number that is easy to read. It weights the two errors the same, and
  real applications seldom do.
- DCF (detection cost function) is a weighted sum of the miss and false-alarm rates at
  one operating point. minDCF is the DCF at its best threshold.

## Range and direction

- `eer` is 0 to 1, often given as a percent. Lower is better.
- `min_dcf` is a normalized cost. Lower is better. `p_target` is 0.01 by default, with
  the NIST convention `c_miss = c_fa = 1`.
- `equal_error_threshold` returns the score threshold where the false-acceptance rate
  is about equal to the false-rejection rate. It is a threshold, not a score.

## When to use it

- Use them to evaluate a speaker verification or anti-spoofing system over many trials,
  or to select a threshold with `equal_error_threshold`.
- Use minDCF when misses and false alarms have different costs. Use EER for a quick
  summary.
- Do not compute them from one pair. They need a population of genuine and impostor
  trials. For one pair, use [speaker similarity](speaker-similarity.md).

## Further reading

The NIST Speaker Recognition Evaluation series defines the detection cost function and
the DET curve.

---
[← Speaker similarity](speaker-similarity.md) · [Home](../index.md)
