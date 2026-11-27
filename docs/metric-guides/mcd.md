# MCD
<!-- agent-hint
covers: what mcd measures, its range, when to use it and its call
source: speechonnxmetrics/intrusive/cepstral.py, speechonnxmetrics/_dsp/dtw.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/mcd.md"
see: pitch.md
-->

MCD (mel-cepstral distortion) measures how far the spectral envelope of a clip is from
that of a reference with the same words.

## Example

```python
import speechonnxmetrics as s

ref = "test/fixtures/audio/source.wav"
deg = "test/fixtures/audio/facodec_aria.wav"
print(s.score(deg, ["mcd"], ref=ref))  # {'mcd': 10.45972843367896}
```

## History

Robert Kubichek defined MCD in *Mel-Cepstral Distance Measure for Objective Speech
Quality Assessment* (IEEE PACRIM, 1993). It came from speech coding, as a low-cost
measure of how far a coded signal drifts from the original. TTS and voice conversion
research then took it as their main objective metric. It measures the spectral envelope
differences that those systems change.

## What it measures

MCD is the distance, in dB, between the mel-cepstral coefficients of two signals, frame
by frame. A lower value means that the output is acoustically closer to the target.

## Range and direction

The value is in dB with no bound, and lower is better. MCD is intrusive, so pass
`ref=`. It aligns the two signals with dynamic time warping over frames
(`speechonnxmetrics/_dsp/dtw.py`). Thus small timing differences have little effect.
This suits TTS and voice conversion, where the output says the same words as the
reference but not at the same instants.

## When to use it

- Use it for TTS, voice conversion and vocoder or codec resynthesis, when you have a
  reference utterance with the same content.
- Do not compare clips with different content.
- It does not measure naturalness (use a MOS predictor) or speaker identity (use
  speaker similarity).

## Further reading

Kubichek, *Mel-Cepstral Distance Measure for Objective Speech Quality Assessment*, IEEE
PACRIM 1993.

---
[← SI-SDR, SDR and SNR](si_sdr-sdr-snr.md) · [Home](../index.md) · [Pitch →](pitch.md)
