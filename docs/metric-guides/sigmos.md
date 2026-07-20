# SIGMOS — P.804 degradation dimensions

## History
SIGMOS is the reference quality predictor released with Microsoft's **ICASSP 2024 Speech
Signal Improvement (SIG) Challenge** (Ristea, Saabas, Cutler et al.). Where DNSMOS focuses
on noise suppression, the SIG Challenge targeted broader *speech signal improvement* —
fixing coloration, discontinuities, loudness and reverberation as well as noise — and it
needed a predictor aligned to the finer **ITU-T P.804** listening-test dimensions. SIGMOS
predicts those dimensions directly.

## What it measures
Seven no-reference quality scores per clip, following P.804: `col` (coloration), `disc`
(discontinuity), `loud` (loudness), `noise` (noisiness), `reverb` (reverberation), `sig`
(signal), and `ovrl` (overall). It gives the most granular "what specifically is wrong"
read of the MOS predictors here.

## Range & direction
Each dimension 1–5, **higher is better**. `sigmos` → dict flattened to `sigmos.col`,
`sigmos.disc`, `sigmos.loud`, `sigmos.noise`, `sigmos.reverb`, `sigmos.sig`, `sigmos.ovrl`.
Native rate **48 kHz** (resampled for you); scores the whole utterance in one pass. No
client-side rescale is needed.

## When to use / when not
- **Use** when you want to *diagnose* an enhancement or generation model along named
  degradation axes — e.g. to see that a dereverberation model improved `reverb` without
  hurting `col`. Its 48 kHz native rate makes it a natural fit for full-band audio.
- **Do not** use it as an intelligibility or speaker-identity measure, and remember the
  extra dimensions are only as reliable as the predictor — treat them as directional.

## Domain
Full-band speech signal improvement (the SIG Challenge lineage). MIT-licensed weights —
safe for commercial use.

## In this library
```python
import speechonnxmetrics as s
s.score("test/fixtures/audio/source.wav", ["sigmos"])
# -> {'sigmos.col': .., 'sigmos.disc': .., 'sigmos.loud': .., 'sigmos.noise': ..,
#     'sigmos.reverb': .., 'sigmos.sig': .., 'sigmos.ovrl': ..}
```

## Further reading
Ristea, Saabas, Cutler et al., *ICASSP 2024 Speech Signal Improvement Challenge*;
ITU-T P.804. Model details: [../models.md](../models.md).
