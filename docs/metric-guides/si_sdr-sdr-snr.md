# SI-SDR, SDR and SNR — signal-fidelity ratios

Covers `si_sdr`, `sdr`, `snr`. All three express, in decibels, how much of the output is
the wanted signal versus everything else.

## History
**SNR** (signal-to-noise ratio) is the oldest idea in the book — the ratio of signal power
to noise power, in dB — and long predates speech processing as a general engineering
measure. **SDR** (signal-to-distortion ratio) was popularized for audio source separation
by the **BSS-Eval** toolkit (Vincent, Gribonval & Févotte, *Performance Measurement in Blind
Audio Source Separation*, IEEE TASLP, 2006), which decomposed a separated signal into target,
interference, and artifact components.

**SI-SDR** (scale-invariant SDR) was introduced by Jonathan Le Roux, Scott Wisdom, Hakan
Erdogan and John Hershey in *SDR – Half-baked or Well Done?* (ICASSP 2019). Their argument:
the BSS-Eval SDR was being "abused", especially for single-channel separation, because its
allowance for a distortion filter let systems get flattering scores through arbitrary
rescaling. SI-SDR fixes this by finding the optimal *scalar* projection of the target onto
the estimate before measuring residual — making the score invariant to output gain, which is
what you want when comparing separators.

## What it measures
The ratio of desired-signal energy to residual (distortion/noise/interference) energy, in
dB. SI-SDR removes the effect of overall scale; SDR and SNR do not.

## Range & direction
Unbounded **dB**, **higher is better** (all three). Intrusive: pass `ref=`. Because these
compare sample-for-sample, they are **alignment-sensitive** — a few ms of latency between
degraded and reference can collapse the score even when the audio sounds identical (see
[../concepts.md](../concepts.md), alignment gotcha).

## When to use / when not
- **Use SI-SDR** as the default for source separation and target extraction, and for
  denoising when you hold the clean reference — it is the modern, gain-safe choice.
- **Use SNR/SDR** when you specifically need the classical, scale-dependent definitions
  (e.g. to match a legacy benchmark).
- **Do not** use any of them for TTS (no aligned reference) or for pipelines with variable
  latency; a large negative SI-SDR on a clip that *sounds* fine usually means misalignment,
  not bad audio.

## Domain
Source separation, speech enhancement, target-speaker extraction. SI-SDR is the field
standard for separation.

## In this library
```python
import speechonnxmetrics as s
s.score("test/fixtures/audio/facodec_aria.wav", ["si_sdr", "sdr", "snr"],
        ref="test/fixtures/audio/source.wav")
# -> {'si_sdr': -26.9..., 'sdr': ..., 'snr': ...}  (very low here: codec output is
#    not sample-aligned to the source — exactly the alignment effect to watch for)
```

## Further reading
Le Roux, Wisdom, Erdogan, Hershey, *SDR – Half-baked or Well Done?*, ICASSP 2019;
Vincent, Gribonval, Févotte, *Performance Measurement in Blind Audio Source Separation*,
IEEE TASLP 2006.
