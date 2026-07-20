# DNSMOS and DNSMOS P.808 — MOS prediction for noise suppression

Covers two registry metrics: `dnsmos` (the P.835 triplet) and `dnsmos_p808` (a single
P.808 score).

## History
DNSMOS comes from Microsoft's **Deep Noise Suppression (DNS) Challenge**, where thousands
of enhanced clips had to be ranked but running an ITU-T listening test on all of them was
impractical. Reddy, Gopal & Cutler introduced it in *DNSMOS: A Non-Intrusive Perceptual
Objective Speech Quality Metric to Evaluate Noise Suppressors* (ICASSP 2021), then
extended it in *DNSMOS P.835* (ICASSP 2022) to output the three separate scores of the
**ITU-T P.835** protocol. The two variants map to two ITU-T recommendations:
- **`dnsmos_p808`** predicts a single **P.808** crowdsourced-listening MOS (overall
  quality from a crowdsourced absolute-category-rating test).
- **`dnsmos`** predicts the **P.835** triplet: **SIG** (the speech signal alone), **BAK**
  (background-noise intrusiveness), and **OVRL** (overall). This split is the whole point —
  it tells "the denoiser hurt the voice" apart from "the denoiser left noise in".

## What it measures
Perceived quality of (typically noise-suppressed) speech, without a reference. The P.835
triplet separates signal quality, background intrusiveness, and overall impression.

## Range & direction
Each score 1–5, **higher is better**. `dnsmos` → dict `sig`, `bak`, `ovrl` (flattened to
`dnsmos.sig`, `dnsmos.bak`, `dnsmos.ovrl`). `dnsmos_p808` → scalar float. Native rate
16 kHz. DNSMOS averages 9.01 s windows at a 1 s hop over long audio. Note: the network
emits an uncalibrated scale and a per-dimension polynomial maps it onto MOS — this package
applies that post-processing; omitting it gives plausible-looking but wrong numbers.

## When to use / when not
- **Use** as the primary quality metric for denoisers and speech enhancement, and for TTS
  as a second opinion alongside UTMOS. The SIG/BAK/OVRL split is uniquely useful for
  diagnosing *what* an enhancement model got wrong.
- **Do not** expect it to measure intelligibility (that is STOI) or speaker identity (that
  is speaker_similarity). It judges quality, not content.

## Domain
Noise suppression / speech enhancement (the DNS Challenge lineage); widely reused for
general speech-quality estimation. MIT-licensed weights — safe for commercial use.

## In this library
```python
import speechonnxmetrics as s
s.score("test/fixtures/audio/source.wav", ["dnsmos", "dnsmos_p808"])
# -> {'dnsmos.sig': 3.4.., 'dnsmos.bak': 3.6.., 'dnsmos.ovrl': 2.9..,
#     'dnsmos_p808': 3.3..}   (downloads weights on first call)
```

## Further reading
Reddy, Gopal, Cutler, *DNSMOS* (ICASSP 2021) and *DNSMOS P.835* (ICASSP 2022);
ITU-T P.808 (crowdsourced ACR) and P.835 (SIG/BAK/OVRL). Model details:
[../models.md](../models.md).
