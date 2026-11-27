# UTMOS
<!-- agent-hint
covers: what UTMOS predicts, its range, when to use it and its call
source: speechonnxmetrics/mos/utmos.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/utmos.md"
see: dnsmos.md
-->

UTMOS predicts the naturalness MOS of a clip without a reference. This page tells you
where it comes from and when to use it.

## History

UTMOS won the VoiceMOS Challenge 2022, a shared task to predict the MOS of synthesized
speech. Saeki, Xin, Nakata, Koshinaka and Takamichi describe it in *UTMOS:
UTokyo-SaruLab System for VoiceMOS Challenge 2022* (Interspeech 2022). TTS and voice
conversion papers need a naturalness MOS, and a listening test is slow and costly.
UTMOS predicts the human MOS from the waveform. This package uses the `utmos22_strong`
variant.

## What it measures

UTMOS gives one naturalness score: its estimate of the rating from a P.800-style
listening panel. Its training data is human ratings of synthesized speech, so it reacts
to the artifacts of TTS models and vocoders.

## Range and direction

The range is 1 to 5, and higher is better. `utmos` returns a float. The exported ONNX
graph holds the affine rescale to the MOS range. The native rate is 16 kHz mono, and the
package resamples the input.

## When to use it

- Use it as the default quality number for TTS and voice conversion, and as a quick
  regression check while you train a model.
- Do not treat it as ground truth. It predicts a listening test, but it is not one. See
  the golden rule in [Concepts](../concepts.md).
- Do not expect calibrated scores far from its training data, such as heavy noise,
  non-speech or unusual codecs. For noise suppression, DNSMOS and SIGMOS match the task
  better.

## In this library

The call downloads the model on the first run, so this page shows it without output:

```python
import speechonnxmetrics as s

s.score("test/fixtures/audio/source.wav", ["utmos"])
```

## Further reading

Saeki et al., *UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022*, Interspeech
2022. [Models and licences](../models.md) gives the export details and the reason the
package has no UTMOSv2.

---
[← Metric guides](README.md) · [Home](../index.md) · [DNSMOS →](dnsmos.md)
