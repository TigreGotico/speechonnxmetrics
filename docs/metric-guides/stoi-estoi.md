# STOI and ESTOI — objective intelligibility

Covers `stoi` and `estoi`.

## History
STOI (Short-Time Objective Intelligibility) was introduced by Cees Taal, Richard Hendriks,
Richard Heusdens and Jesper Jensen in *A Short-Time Objective Intelligibility Measure for
Time-Frequency Weighted Noisy Speech* (ICASSP 2010), with the full treatment in *An
Algorithm for Intelligibility Prediction of Time–Frequency Weighted Noisy Speech* (IEEE
Transactions on Audio, Speech, and Language Processing, 2011). The problem it solved: noise-
reduction and hearing-aid research needed to predict *how intelligible* processed speech
would be to human listeners, and older quality measures correlated poorly with
intelligibility for time-frequency-weighted (e.g. noise-suppressed) speech. STOI decomposes
clean and degraded speech into one-third-octave bands and correlates their short-time
temporal envelopes; it showed high correlation with listener word-recognition scores.

**ESTOI** (Extended STOI) is Jesper Jensen and Cees Taal, *An Algorithm for Predicting the
Intelligibility of Speech Masked by Modulated Noise Maskers* (IEEE/ACM TASLP, 2016). It
generalizes STOI to hold up under highly modulated noise, where the original's per-band
independence assumption weakens, by correlating whole spectral envelopes over short segments.

## What it measures
**Intelligibility** — an estimate of the proportion of words a listener would understand —
not quality or naturalness. Reference-based (intrusive).

## Range & direction
0–1, **higher is better** (both `stoi` and `estoi`). Analysis is performed at 10 kHz (the
package resamples for you). Intrusive: pass `ref=`.

## When to use / when not
- **Use** for speech enhancement, noise suppression and hearing-aid evaluation where the
  question is comprehension; use **ESTOI** when the interfering noise is strongly modulated.
- **Do not** use STOI as a *quality* proxy — a clip can be perfectly intelligible yet sound
  robotic or unpleasant (that is the MOS predictors' job). And, being intrusive, it needs a
  time-aligned clean reference; it does not apply to open-ended TTS without one.

## Domain
Speech-intelligibility / noise-reduction / hearing research; a de facto standard there.

## In this library
```python
import speechonnxmetrics as s
s.score("test/fixtures/audio/facodec_aria.wav", ["stoi", "estoi"],
        ref="test/fixtures/audio/source.wav")
# -> {'stoi': 0.66..., 'estoi': 0.4...}
```
Parity: STOI matches `pystoi` to 2.6e-10 (see [../metrics.md](../metrics.md)).

## Further reading
Taal, Hendriks, Heusdens, Jensen (ICASSP 2010; IEEE TASLP 2011); Jensen & Taal (IEEE/ACM
TASLP 2016).
