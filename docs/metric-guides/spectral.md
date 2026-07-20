# LSD, MSD and mel-L1 — spectral distances

Covers `lsd` (log-spectral distance), `msd` (mel-spectral distortion) and `mel_l1` (L1 on
log-mel spectrograms).

## History
These are **classical spectral distances**, not the invention of a single paper. Log-spectral
distance in particular is a long-standing measure in speech coding and analysis — the
distance between two signals' short-time log-magnitude spectra — used for decades to quantify
how far a processed spectrum sits from a reference. The mel-domain variants (`msd`, `mel_l1`)
apply the same idea on a perceptual mel scale and have become routine reconstruction losses
and evaluation metrics for vocoders and neural audio codecs. Treat them as standard DSP
tooling with a long, diffuse lineage rather than one citable origin.

## What they measure
Frame-by-frame magnitude difference between two spectrograms:
- **`lsd`** — log-spectral distance on the linear-frequency log-magnitude spectrum (dB).
- **`msd`** — the same idea on a mel-scaled spectrum (dB).
- **`mel_l1`** — the mean L1 (absolute) distance between two log-mel spectrograms.

All three answer "how different do these two signals look in the time-frequency plane?"

## Range & direction
- `lsd` — **dB**, unbounded, **lower is better**.
- `msd` — **dB**, unbounded, **lower is better**.
- `mel_l1` — unbounded, **lower is better**.

All intrusive: pass `ref=`. Implementations live in `speechonnxmetrics/intrusive/spectral.py`.

## When to use / when not
- **Use** for vocoder and neural-codec resynthesis evaluation, and as low-level fidelity
  checks — they are cheap, deterministic, and directly reflect reconstruction error. `mel_l1`
  in particular mirrors a loss many vocoders are trained on, so it is a natural eval echo.
- **Do not** treat them as perceptual quality: two clips can have similar mel-L1 yet sound
  quite different, and vice versa. Like all intrusive metrics they need a time-aligned
  reference and the *same* content.

## Domain
Speech coding, vocoders, neural audio codecs; reconstruction-fidelity evaluation.

## In this library
```python
import speechonnxmetrics as s
s.score("test/fixtures/audio/facodec_aria.wav", ["lsd", "msd", "mel_l1"],
        ref="test/fixtures/audio/source.wav")
# -> {'lsd': ..., 'msd': ..., 'mel_l1': ...}   (all lower = better)
```

## Further reading
No single defining paper; these are standard spectral-distance measures. For the perceptual
side of "how close is the spectrum", compare with [mcd.md](mcd.md).
