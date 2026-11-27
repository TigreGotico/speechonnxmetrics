# NISQA
<!-- agent-hint
covers: what nisqa predicts, its range, its licence, when to use it and its call
source: speechonnxmetrics/mos/nisqa.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/nisqa.md"
see: stoi-estoi.md
-->

NISQA predicts the overall quality of a clip and four quality dimensions, without a
reference. Its weights forbid commercial use.

## History

Gabriel Mittag and colleagues at the TU Berlin Quality and Usability Lab made NISQA.
They describe it in *NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech
Quality Prediction with Crowdsourced Datasets* (Mittag, Naderi, Chehadi and Möller,
Interspeech 2021). One overall MOS does not show why the quality is low. So NISQA also
predicts noisiness, coloration, discontinuity and loudness. Its training data is large
crowdsourced sets of telephone and network degradations. This package uses NISQA-v2.

## Range and direction

Each score is 1 to 5, and higher is better. The result flattens to `nisqa.mos`,
`nisqa.noi`, `nisqa.dis`, `nisqa.col` and `nisqa.loud`. The output order follows the
upstream code. The upstream class docstring puts coloration before discontinuity, and
it is wrong.

NISQA does not resample. It takes its mel hop and window from the sample rate of the
file, as the reference does. It pools mel patches of 15 frames inside the graph, so it
rejects audio shorter than 15 mel frames.

## When to use it

- Use it for communication quality with a breakdown by dimension, next to DNSMOS. It
  gives a second MOS from an independent model.
- Do not use it for commercial work. Its weights are CC BY-NC-SA 4.0 (NonCommercial),
  and its code is MIT. For commercial use, choose UTMOS, DNSMOS or SIGMOS.
- Do not use it on very short clips, because of the 15-frame floor.

## In this library

The call downloads the model on the first run, so this page shows it without output:

```python
import speechonnxmetrics as s

s.score("test/fixtures/audio/source.wav", ["nisqa"])
```

## Further reading

Mittag, Naderi, Chehadi, Möller, *NISQA*, Interspeech 2021. [Models and
licences](../models.md) gives the licence terms.

---
[← SIGMOS](sigmos.md) · [Home](../index.md) · [STOI and ESTOI →](stoi-estoi.md)
