# DNSMOS and DNSMOS P.808
<!-- agent-hint
covers: what dnsmos and dnsmos_p808 predict, their range, when to use them and their call
source: speechonnxmetrics/mos/dnsmos.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/dnsmos.md"
see: sigmos.md
-->

DNSMOS predicts the quality of noise-suppressed speech without a reference. This page
covers two registry metrics: `dnsmos`, the P.835 triplet, and `dnsmos_p808`, one P.808
score.

## History

DNSMOS comes from the Microsoft Deep Noise Suppression (DNS) Challenge. The challenge
had too many enhanced clips for an ITU-T listening test on each one. Reddy, Gopal and
Cutler introduced DNSMOS at ICASSP 2021. *DNSMOS P.835* (ICASSP 2022) added the three
scores of the ITU-T P.835 protocol.

- `dnsmos_p808` predicts one P.808 MOS: the overall quality from a crowdsourced
  absolute category rating test.
- `dnsmos` predicts the P.835 triplet: SIG (the speech signal), BAK (the intrusiveness
  of the background noise) and OVRL (overall). The triplet tells a damaged voice apart
  from residual noise.

## Range and direction

Each score is 1 to 5, and higher is better. `dnsmos` returns a dict that the result
flattens to `dnsmos.sig`, `dnsmos.bak` and `dnsmos.ovrl`. `dnsmos_p808` returns a float.
The native rate is 16 kHz. On long audio, DNSMOS averages windows of 9.01 s at a hop of
1 s.

The network gives scores on an uncalibrated scale. A polynomial for each dimension maps
them to MOS, and this package applies it. Without that step the numbers look correct
but are wrong.

## When to use it

- Use it as the main quality metric for denoisers and speech enhancement. The SIG, BAK
  and OVRL split shows what an enhancement model did wrong.
- Use it for TTS as a second opinion next to UTMOS.
- Do not use it for intelligibility (use STOI) or speaker identity (use speaker
  similarity). It measures quality, not content.

The weights are MIT, so commercial use is permitted.

## In this library

The call downloads the models on the first run, so this page shows it without output:

```python
import speechonnxmetrics as s

s.score("test/fixtures/audio/source.wav", ["dnsmos", "dnsmos_p808"])
```

## Further reading

Reddy, Gopal, Cutler, *DNSMOS* (ICASSP 2021) and *DNSMOS P.835* (ICASSP 2022). ITU-T
P.808 and P.835. [Models and licences](../models.md) gives the model details.

---
[← UTMOS](utmos.md) · [Home](../index.md) · [SIGMOS →](sigmos.md)
