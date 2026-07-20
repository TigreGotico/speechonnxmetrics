# UTMOS — predicted naturalness MOS

## History
UTMOS was the winning entry of the **VoiceMOS Challenge 2022**, a shared task on
predicting mean-opinion scores of *synthesized* speech. It is described in Saeki, Xin,
Nakata, Koshinaka, Takamichi & Takamichi's *UTMOS: UTokyo-SaruLab System for VoiceMOS
Challenge 2022* (Interspeech 2022). The problem it was built to solve: TTS and voice-
conversion research had exploded, but every paper still needed slow, expensive human
listening tests to report a naturalness MOS. UTMOS learns to predict that human MOS
directly from the waveform, so researchers can rank systems without a panel. The model
here is the `utmos22_strong` variant.

## What it measures
A single **naturalness** score — the model's estimate of what a P.800-style listening
panel would rate the clip's quality. It is trained on human ratings of synthesized speech,
so it is tuned to the kinds of artifacts TTS and vocoders produce.

## Range & direction
1–5, **higher is better** (`utmos`, scalar float). The affine rescale onto the MOS range
is baked into the exported ONNX graph. Native rate 16 kHz mono (resampled for you).

## When to use / when not
- **Use** as the default one-number quality score for TTS and voice conversion, and as a
  quick regression guard during model iteration.
- **Do not** treat it as ground truth — it is a prediction of a listening test, not the
  test (see [../concepts.md](../concepts.md), the golden rule). Do not expect calibrated
  agreement on domains far from its training data (heavy noise, non-speech, exotic codecs).
  For noise-suppression specifically, DNSMOS/SIGMOS are the better-matched predictors.

## Domain
Born in the TTS / voice-conversion evaluation community (VoiceMOS Challenge); today it is
one of the most-cited quick naturalness proxies for synthesized speech.

## In this library
```python
import speechonnxmetrics as s
s.score("test/fixtures/audio/source.wav", ["utmos"])
# -> {'utmos': 4.41...}   (downloads ~30 MB of weights on first call, then caches)
```

## Further reading
Saeki et al., *UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022*, Interspeech 2022.
Export details and the UTMOSv2 omission: [../models.md](../models.md).
