# SI-SDR, SDR and SNR
<!-- agent-hint
covers: what si_sdr, sdr and snr measure, their range, when to use them and their call
source: speechonnxmetrics/intrusive/sdr.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/si_sdr-sdr-snr.md"
see: mcd.md
-->

SI-SDR, SDR and SNR give, in decibels, the ratio of the wanted signal to all else in
the output. This page covers `si_sdr`, `sdr` and `snr`.

## Example

```python
import speechonnxmetrics as s

ref = "test/fixtures/audio/source.wav"
deg = "test/fixtures/audio/facodec_aria.wav"
print(s.score(deg, ["si_sdr", "sdr", "snr"], ref=ref))  # {'si_sdr': -26.937894650414812, 'sdr': -2.362625413781588, 'snr': -2.362625413781588}
```

The codec output is not aligned sample by sample with the source, so the values are low.
This is the alignment effect described below.

## History

SNR (signal-to-noise ratio) is the ratio of signal power to noise power, in dB. It is
older than speech processing. The BSS-Eval toolkit made SDR (signal-to-distortion
ratio) common for source separation (Vincent, Gribonval and Févotte, *Performance
Measurement in Blind Audio Source Separation*, IEEE TASLP, 2006). BSS-Eval splits a
separated signal into target, interference and artifacts.

Le Roux, Wisdom, Erdogan and Hershey introduced SI-SDR (scale-invariant SDR) in *SDR –
Half-baked or Well Done?* (ICASSP 2019). The BSS-Eval SDR permits a distortion filter,
so a system can raise its score by a change of scale. SI-SDR projects the target onto
the estimate with one scalar before it measures the residual. Thus the output gain does
not change the score.

## Range and direction

All three are in dB with no bound, and higher is better. SI-SDR removes the effect of
the overall scale. SDR and SNR do not. All three are intrusive, so pass `ref=`.

They compare sample by sample, so they are sensitive to alignment. A delay of a few ms
between the clip and the reference can drop the score, even when the two sound the
same. See the alignment trap in [Concepts](../concepts.md).

## When to use it

- Use SI-SDR for source separation and target extraction, and for denoising when you
  have the clean reference.
- Use SNR or SDR when you need the classical definitions that depend on scale, for
  example to match an older benchmark.
- Do not use them for TTS, which has no aligned reference, or for a pipeline with a
  variable delay. A large negative SI-SDR on a clip that sounds correct usually means
  misalignment.

## Further reading

Le Roux, Wisdom, Erdogan, Hershey, *SDR – Half-baked or Well Done?*, ICASSP 2019.
Vincent, Gribonval, Févotte, *Performance Measurement in Blind Audio Source
Separation*, IEEE TASLP 2006.

---
[← STOI and ESTOI](stoi-estoi.md) · [Home](../index.md) · [MCD →](mcd.md)
