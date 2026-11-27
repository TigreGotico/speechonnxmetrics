# STOI and ESTOI
<!-- agent-hint
covers: what stoi and estoi measure, their range, when to use them and their call
source: speechonnxmetrics/intrusive/stoi.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/stoi-estoi.md"
see: si_sdr-sdr-snr.md
-->

STOI and ESTOI estimate how intelligible a degraded clip is, compared with a clean
reference. This page covers `stoi` and `estoi`.

## Example

```python
import speechonnxmetrics as s

ref = "test/fixtures/audio/source.wav"
deg = "test/fixtures/audio/facodec_aria.wav"
print(s.score(deg, ["stoi", "estoi"], ref=ref))  # {'stoi': 0.6620030195244008, 'estoi': 0.3994431470718137}
```

## History

Taal, Hendriks, Heusdens and Jensen introduced STOI (Short-Time Objective
Intelligibility) at ICASSP 2010. The full method is in *An Algorithm for Intelligibility
Prediction of Time-Frequency Weighted Noisy Speech* (IEEE TASLP, 2011). Noise reduction
and hearing-aid research needed to predict how well listeners understand processed
speech. Older quality measures did not follow intelligibility well for noise-suppressed
speech. STOI splits the clean and the degraded speech into one-third-octave bands and
correlates their short-time envelopes.

Jensen and Taal introduced ESTOI (Extended STOI) in *An Algorithm for Predicting the
Intelligibility of Speech Masked by Modulated Noise Maskers* (IEEE/ACM TASLP, 2016).
STOI treats each band as independent, and that fails under strongly modulated noise.
ESTOI correlates full spectral envelopes over short segments.

## Range and direction

STOI is about 0 to 1. ESTOI is a correlation from -1 to 1, and is about 0 to 1 in practice. Higher is better for both. The analysis runs at 10 kHz, and the
package resamples the input. Both are intrusive, so pass `ref=`.
`test/test_intrusive.py` compares both with a committed fixture of `pystoi` outputs.

## When to use it

- Use it for speech enhancement, noise suppression and hearing aids, when the question
  is comprehension. Use ESTOI when the noise is strongly modulated.
- Do not use STOI for quality. A clip can be intelligible and still sound robotic. The
  MOS predictors measure quality.
- It needs a clean reference aligned in time, so it does not apply to TTS without one.

## Further reading

Taal, Hendriks, Heusdens, Jensen (ICASSP 2010, IEEE TASLP 2011). Jensen and Taal
(IEEE/ACM TASLP 2016).

---
[← NISQA](nisqa.md) · [Home](../index.md) · [SI-SDR, SDR and SNR →](si_sdr-sdr-snr.md)
