# LSD, MSD and mel-L1
<!-- agent-hint
covers: what lsd, msd and mel_l1 measure, their range, when to use them and their call
source: speechonnxmetrics/intrusive/spectral.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/spectral.md"
see: wer-family.md
-->

These three metrics measure the distance between two spectrograms. This page covers
`lsd` (log-spectral distance), `msd` (mel-spectral distortion) and `mel_l1` (L1 on
log-mel spectrograms).

## Example

```python
import speechonnxmetrics as s

ref = "test/fixtures/audio/source.wav"
deg = "test/fixtures/audio/facodec_aria.wav"
print(s.score(deg, ["lsd", "msd", "mel_l1"], ref=ref))  # {'lsd': 12.207452774047852, 'msd': 8.374044401926286, 'mel_l1': 1.514757084921227}
```

## History

These are classical spectral distances with no single source paper. Speech coding and
analysis used log-spectral distance for decades. The mel variants apply the same idea
on the perceptual mel scale. Vocoders and neural codecs use them as training losses and
as evaluation metrics.

## What they measure

- `lsd`: the log-spectral distance on the linear-frequency log-magnitude spectrum, in
  dB.
- `msd`: the same distance on a mel spectrum, in dB.
- `mel_l1`: the mean absolute distance between two log-mel spectrograms.

## Range and direction

All three have no bound, and lower is better. `lsd` and `msd` are in dB. All three are
intrusive, so pass `ref=`. The code is in `speechonnxmetrics/intrusive/spectral.py`.

## When to use it

- Use them to evaluate vocoder and codec resynthesis, and as low-level fidelity checks.
  They are fast and deterministic. Many vocoders train on a loss like `mel_l1`.
- Do not use them for perceived quality. Two clips with a similar mel-L1 can sound
  different.
- They need a reference aligned in time, with the same content.

## Further reading

No single paper defines them. Compare with [MCD](mcd.md).

---
[← Pitch](pitch.md) · [Home](../index.md) · [WER family →](wer-family.md)
