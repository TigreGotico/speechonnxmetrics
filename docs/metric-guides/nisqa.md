# NISQA — multidimensional speech quality

## History
NISQA was developed at TU Berlin's Quality and Usability Lab by Gabriel Mittag and
colleagues, presented in *NISQA: A Deep CNN-Self-Attention Model for Multidimensional
Speech Quality Prediction with Crowdsourced Datasets* (Mittag, Naderi, Chehadi & Möller,
Interspeech 2021). Its motivation was that a single overall MOS hides *why* quality is low,
so NISQA predicts overall MOS **plus** four perceptual quality dimensions — noisiness,
coloration, discontinuity, loudness — trained on large crowdsourced datasets covering
telephone- and communication-network degradations. The model here is NISQA-v2.

## What it measures
Overall quality and its breakdown: `mos` (overall), `noi` (noisiness), `dis`
(discontinuity), `col` (coloration), `loud` (loudness). No reference needed.

## Range & direction
Each score 1–5, **higher is better**. `nisqa` → dict flattened to `nisqa.mos`, `nisqa.noi`,
`nisqa.dis`, `nisqa.col`, `nisqa.loud`. **Rate-adaptive**: NISQA never resamples — it
derives its mel hop/window from the file's own sample rate, as the reference does. It pools
15-frame mel patches inside the graph, so audio shorter than 15 mel frames is *rejected*
rather than padded.

## When to use / when not
- **Use** for a communication-quality read with a dimensional breakdown, complementary to
  DNSMOS. Handy when you want a second, independently-trained MOS opinion.
- **Do not** use it commercially: its **weights are CC BY-NC-SA 4.0 (NonCommercial)** — the
  one NonCommercial artifact in this package (the code is MIT). If your use is commercial,
  drop NISQA and use UTMOS/DNSMOS/SIGMOS. Also avoid it on very short clips (see the 15-frame
  floor). Output order is `mos, noi, dis, col, loud` — the upstream class docstring lists
  coloration before discontinuity and is wrong; this package follows the code.

## Domain
Telecommunication / VoIP speech quality; multidimensional degradation diagnosis.

## In this library
```python
import speechonnxmetrics as s
s.score("test/fixtures/audio/source.wav", ["nisqa"])
# -> {'nisqa.mos': .., 'nisqa.noi': .., 'nisqa.dis': .., 'nisqa.col': .., 'nisqa.loud': ..}
```

## Further reading
Mittag, Naderi, Chehadi, Möller, *NISQA…*, Interspeech 2021. Licence caveat:
[../models.md](../models.md).
