# SIGMOS
<!-- agent-hint
covers: what sigmos predicts, its range, when to use it and its call
source: speechonnxmetrics/mos/sigmos.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/sigmos.md"
see: nisqa.md
-->

SIGMOS predicts seven quality dimensions of ITU-T P.804 without a reference. This page
tells you where it comes from and when to use it.

## History

SIGMOS is the quality predictor of the Microsoft ICASSP 2024 Speech Signal Improvement
(SIG) Challenge (Ristea, Saabas, Cutler et al.). DNSMOS targets noise suppression. The
SIG Challenge targets a wider task: it also fixes coloration, discontinuity, loudness
and reverberation. It needed a predictor for the finer P.804 dimensions.

## What it measures

SIGMOS gives seven scores for each clip: `col` (coloration), `disc` (discontinuity),
`loud` (loudness), `noise` (noisiness), `reverb` (reverberation), `sig` (signal) and
`ovrl` (overall). Of the MOS predictors in this package, it gives the most detail about
what is wrong.

## Range and direction

Each dimension is 1 to 5, and higher is better. The result flattens to `sigmos.col`,
`sigmos.disc`, `sigmos.loud`, `sigmos.noise`, `sigmos.reverb`, `sigmos.sig` and
`sigmos.ovrl`. The native rate is 48 kHz, and the package resamples the input. SIGMOS
scores the full utterance in one pass and needs no rescale in the client.

## When to use it

- Use it to find which degradation an enhancement or generation model changes. For
  example, a dereverberation model can improve `reverb` and keep `col`.
- Its native rate of 48 kHz suits full-band audio.
- Do not use it for intelligibility or speaker identity. The extra dimensions are only
  as good as the predictor, so read them as a direction.

The weights are MIT, so commercial use is permitted.

## In this library

The call downloads the model on the first run, so this page shows it without output:

```python
import speechonnxmetrics as s

s.score("test/fixtures/audio/source.wav", ["sigmos"])
```

## Further reading

Ristea, Saabas, Cutler et al., *ICASSP 2024 Speech Signal Improvement Challenge*. ITU-T
P.804. [Models and licences](../models.md) gives the model details.

---
[← DNSMOS](dnsmos.md) · [Home](../index.md) · [NISQA →](nisqa.md)
