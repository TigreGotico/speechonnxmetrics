# Log-F0 RMSE and voiced/unvoiced error — prosody metrics

Covers `log_f0_rmse` and `vuv_error`.

## History
Unlike STOI or MCD, these two do not trace to a single landmark paper — they are
**conventional prosody measures** that became standard tooling in text-to-speech and
voice-conversion evaluation, reported alongside MCD in countless system papers. They
measure the pitch side of speech that MCD (a spectral-envelope metric) leaves out. Treat
their lineage as "community convention", not one canonical citation.

## What they measure
- **`log_f0_rmse`** — the root-mean-square error between the two signals' fundamental-
  frequency (F0, i.e. pitch) contours, in the **log** domain (log-Hz), computed over frames
  where both are voiced. Log-scaling matches how pitch is perceived and keeps the error from
  being dominated by high-pitched frames. It captures intonation/prosody mismatch.
- **`vuv_error`** — the **voiced/unvoiced decision error rate**: the fraction of frames where
  one signal is voiced and the other is not. It captures a coarser failure — the system
  putting voicing (vocal-fold vibration) in the wrong places.

## Range & direction
- `log_f0_rmse` — unbounded, **lower is better**.
- `vuv_error` — 0–1, **lower is better** (a rate).

Both intrusive: pass `ref=`. Pitch extraction lives in `speechonnxmetrics/_dsp/pitch.py`.

## When to use / when not
- **Use** as prosody companions to MCD when evaluating TTS/VC — MCD tells you the timbre is
  right, log-F0 RMSE and V/UV error tell you the *intonation* is right. Especially relevant
  for expressive/emotional TTS and for pitch-sensitive voice conversion.
- **Do not** read too much into them in isolation: pitch tracking is noisy, and on content
  that is nearly all unvoiced (whispers, heavy fricatives) the estimates are unstable.

## Domain
TTS and voice-conversion prosody evaluation.

## In this library
```python
import speechonnxmetrics as s
s.score("test/fixtures/audio/facodec_aria.wav", ["log_f0_rmse", "vuv_error"],
        ref="test/fixtures/audio/source.wav")
# -> {'log_f0_rmse': ..., 'vuv_error': ...}   (both lower = better)
```

## Further reading
No single defining paper; see any TTS/VC objective-evaluation setup that reports F0 RMSE and
V/UV error alongside MCD. Pitch metrics pair naturally with [mcd.md](mcd.md).
