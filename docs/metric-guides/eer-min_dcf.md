# EER and minDCF — speaker-verification metrics

Covers `eer`, `min_dcf` and the helper `equal_error_threshold`. These score a *detection*
problem: given a similarity score per trial and a genuine/impostor label, how well does the
system separate the two?

## History
Both come from the **speaker-verification** tradition, shaped heavily by the **NIST Speaker
Recognition Evaluation (SRE)** series, which has run since the mid-1990s and standardized how
the field reports verification performance. Verification is framed as detection theory:
every trial is "same speaker or not?", and you trade off **false alarms** (accepting an
impostor) against **misses** (rejecting a genuine speaker) by moving a decision threshold.

- **EER** (equal error rate) is the threshold-independent point where the false-alarm rate
  equals the miss rate — one easy-to-read number. It is popular for its simplicity, though
  NIST itself has generally *not* favored it because it weights the two error types equally,
  which real applications rarely do.
- **DCF** (detection cost function) is NIST's preferred measure: a weighted combination of
  miss and false-alarm rates for a chosen operating point. **minDCF** is the DCF at its best
  threshold, so systems are compared at their optimal operating point rather than a fixed one.

## What they measure
How separable genuine and impostor trials are. Lower EER / minDCF = better verification.

## Range & direction
- `eer` — 0–1 (often quoted as a percentage), **lower is better**.
- `min_dcf` — a normalized cost, **lower is better**; `p_target` defaults to 0.01 with the
  NIST convention `c_miss = c_fa = 1`.
- `equal_error_threshold` — returns the *score threshold* where FAR ≈ FRR (a threshold, not a
  quality score).

All three are pure numpy and need **no** model or extra — you supply the arrays. Not in the
`score()` registry; call from `speechonnxmetrics.speaker`.

## When to use / when not
- **Use** to evaluate a speaker-verification / anti-spoofing system across many trials, or to
  pick an operating threshold (`equal_error_threshold`). minDCF when the application weights
  misses and false alarms differently; EER for a quick single-number summary.
- **Do not** compute them from a single pair — they need a *population* of genuine and impostor
  trials to estimate error rates. For a one-off "are these two the same voice?" you want
  [speaker-similarity.md](speaker-similarity.md), not EER.

## Domain
Speaker verification, biometric detection, anti-spoofing (NIST SRE lineage).

## In this library
```python
import numpy as np
from speechonnxmetrics.speaker import eer, min_dcf, equal_error_threshold

scores = np.array([0.9, 0.85, 0.2, 0.1, 0.75, 0.3])   # verification scores
labels = np.array([1,   1,    0,   0,   1,    0])       # 1 = genuine, 0 = impostor
eer(scores, labels)                    # equal error rate
min_dcf(scores, labels, p_target=0.01) # min normalized detection cost
equal_error_threshold(scores, labels)  # threshold where FAR == FRR
```

## Further reading
The NIST Speaker Recognition Evaluation series (detection cost function; DET curves) is the
canonical reference for minDCF and verification-metric conventions.
