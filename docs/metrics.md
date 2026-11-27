# Metrics reference
<!-- agent-hint
covers: each metric family, its range, direction, sample rate, reference need and paper
source: speechonnxmetrics/registry.py, speechonnxmetrics/mos/, speechonnxmetrics/intrusive/, speechonnxmetrics/asr/, speechonnxmetrics/speaker/
generated: no
verify: python -m pytest -q test/test_docs.py -k "docs/metrics.md"
see: lect-fidelity.md
-->

This page is the lookup table for each metric, grouped by family. The
[metric guides](metric-guides/README.md) explain the history and use of each metric.
`speechonnxmetrics.list_metrics()` prints the registered set. The table gives
the range and direction of each metric. The registry metadata
(`RegistryEntry.range` and `RegistryEntry.higher_is_better`) differs from it for
`cer`, `estoi` and `wip` ([#8](https://github.com/TigreGotico/speechonnxmetrics/issues/8)).

`score()` and `score_batch()` dispatch only `kind="audio"` metrics. `kind="text"`
metrics (WER/CER/MER/WIL/WIP) compare strings and are called from
`speechonnxmetrics.asr` directly.

## No-reference MOS

Guides: [utmos](metric-guides/utmos.md) · [dnsmos + dnsmos_p808](metric-guides/dnsmos.md) · [sigmos](metric-guides/sigmos.md) · [nisqa](metric-guides/nisqa.md).

Neural mean-opinion-score predictors. No reference signal is needed. Each downloads an
ONNX model on first call (`requires_download=True`) and caches it. All are bounded 1-5,
higher is better. Sample-rate conversion to each model's native rate is handled by the
base class.

| metric | native rate | output | notes |
|---|---|---|---|
| `utmos` | 16 kHz mono | float | UTMOS22 naturalness MOS. Affine rescale baked into the exported graph |
| `dnsmos` | 16 kHz | dict `sig`, `bak`, `ovrl` | P.835 triplet. Per-dimension polynomial rescale applied client-side |
| `dnsmos_p808` | 16 kHz | float | P.808 single crowdsourced-listening MOS |
| `sigmos` | 48 kHz | dict `col`, `disc`, `loud`, `noise`, `reverb`, `sig`, `ovrl` | P.804 seven-dimension prediction |
| `nisqa` | rate-adaptive | dict `mos`, `noi`, `dis`, `col`, `loud` | never resamples. Derives mel hop/window from the file's own rate |

Long-audio handling differs per model. DNSMOS averages 9.01 s windows at a 1 s hop.
SIGMOS and UTMOS score the whole utterance in one pass. NISQA pools 15-frame mel patches
inside the graph and rejects audio shorter than 15 mel frames rather than padding it.

**Licence:** `utmos`, `dnsmos`, `dnsmos_p808`, `sigmos` are MIT. **`nisqa` weights are
CC BY-NC-SA 4.0 (NonCommercial)**: its code is MIT but the weights forbid commercial
use. See [models.md](models.md).

**Parity:** `test/test_mos.py` compares each predictor with its reference
implementation. The tolerances are 1e-5 for `dnsmos` and `dnsmos_p808`, 1e-4 for
`utmos` and `nisqa`, and 2e-3 for `sigmos`. These tests download the models and carry
the `models` mark.

Papers:

- DNSMOS: Reddy, Gopal, Cutler, *DNSMOS: A Non-Intrusive Perceptual Objective Speech
  Quality Metric to Evaluate Noise Suppressors*, ICASSP 2021, and *DNSMOS P.835*, ICASSP
  2022.
- SIGMOS: Ristea, Saabas, Cutler et al., *ICASSP 2024 Speech Signal Improvement Challenge*.
- UTMOS: Saeki, Xin, Nakata, Koshinaka, Takamichi, *UTMOS: UTokyo-SaruLab System for
  VoiceMOS Challenge 2022*, Interspeech 2022.
- NISQA: Mittag, Naderi, Chehadi, Möller, *NISQA: A Deep CNN-Self-Attention Model for
  Multidimensional Speech Quality Prediction with Crowdsourced Datasets*, Interspeech 2021.

## Intrusive (reference-based)

Guides: [stoi + estoi](metric-guides/stoi-estoi.md) · [si_sdr + sdr + snr](metric-guides/si_sdr-sdr-snr.md) · [mcd](metric-guides/mcd.md) · [log_f0_rmse + vuv_error](metric-guides/pitch.md) · [lsd + msd + mel_l1](metric-guides/spectral.md).

Pure-numpy signal metrics that compare a degraded signal against a clean reference. No
download is needed. Each requires `ref=` (`intrusive=True`). Calling without a reference
raises. Signature: `metric(deg, sr, *, ref, ref_sr=None)`. Length- and rate-matching
policy is shared in `speechonnxmetrics.intrusive._common`. Unscorable inputs raise
`IntrusiveMetricError`.

| metric | range | ↑better | measures |
|---|---|---|---|
| `stoi` | 0-1 | yes | short-time objective intelligibility (analysis resampled to 10 kHz) |
| `estoi` | -1 to 1, about 0 to 1 in practice | yes | extended STOI, correlation over spectral envelopes |
| `si_sdr` | dB | yes | scale-invariant signal-to-distortion ratio |
| `sdr` | dB | yes | signal-to-distortion ratio |
| `snr` | dB | yes | signal-to-noise ratio |
| `mcd` | dB | no | mel-cepstral distortion |
| `log_f0_rmse` | unbounded | no | RMSE of log-F0 (pitch-contour error) |
| `vuv_error` | 0-1 | no | voiced/unvoiced decision error rate |
| `lsd` | dB | no | log-spectral distance |
| `msd` | dB | no | mel-spectral distortion |
| `mel_l1` | unbounded | no | L1 distance on log-mel spectrograms |

**Parity:** `test/test_intrusive.py` compares `stoi` and `estoi` with a committed
fixture of `pystoi` outputs, to 1e-4 and 3e-4.

Papers:

- STOI/ESTOI: Taal et al., *An Algorithm for Intelligibility Prediction of Time-Frequency
  Weighted Noisy Speech*, IEEE TASLP 2011, and Jensen & Taal, *An Algorithm for
  Predicting the Intelligibility of Speech Masked by Modulated Noise Maskers*, IEEE/ACM
  TASLP 2016.
- SI-SDR: Le Roux et al., *SDR - Half-Baked or Well Done?*, ICASSP 2019.
- MCD: Kubichek, *Mel-Cepstral Distance Measure for Objective Speech Quality
  Assessment*, IEEE PACRIM 1993.

## ASR-based text metrics

Guide: [wer + cer + mer + wil + wip](metric-guides/wer-family.md).

Pure-numpy string comparisons. `kind="text"`, so not dispatched by `score()`. Signature:
`metric(reference, hypothesis, normalizer=None)`, accepting a single string or a list of
strings (corpus). Normalization is opt-in: scoring never normalizes on its own. An empty
reference raises `EmptyReferenceError`.

| metric | range | ↑better | measures |
|---|---|---|---|
| `wer` | ≥0 | no | word error rate = (S+D+I)/N |
| `cer` | ≥0 | no | character error rate |
| `mer` | 0-1 | no | match error rate |
| `wil` | 0-1 | no | word information lost |
| `wip` | 0-1 | yes | word information preserved |

`compute()` returns an `AsrMetrics` dataclass with the raw counts (hits, substitutions,
deletions, insertions, ref/hyp length) and all five rates in one pass. `align()` returns
the edit operations.

Normalizer presets (`speechonnxmetrics.asr`): `BASIC` (lowercase + collapse whitespace),
`STRICT` (also expand contractions, strip diacritics/punctuation/filler words), and
`LEGACY`. **LEGACY** reproduces the `re.sub(r"[^a-z' ]", " ", text.lower()).split()`
tokenizer from older call sites and is provided only so migrating them stays
value-preserving. It discards every non-ASCII character (accented Latin, Arabic, and
others), so do not use it in other code. Build your own with the `Normalizer` chain and
the exported step functions (`lowercase`, `strip_punctuation`, `remove_diacritics`, and
others).

## Speaker similarity and verification

Guides: [speaker_similarity](metric-guides/speaker-similarity.md) · [eer + min_dcf](metric-guides/eer-min_dcf.md).

`speechonnxmetrics.speaker` (not in the `score()` registry):

- `speaker_similarity(deg, sr, *, ref, ref_sr=None, model=None)`: cosine similarity
  between speaker embeddings, backed by `speakeronnx`. Needs the `speaker` extra.
  `SpeakerSimilarity` is the reusable form with a lazily-loaded embedder.
- `eer(scores, labels)`: equal error rate over verification scores. Labels are 1 for
  genuine, 0 for impostor pairs. Pure numpy.
- `min_dcf(scores, labels, p_target=0.01)`: minimum normalized detection cost (NIST SRE
  convention, c_miss = c_fa = 1). Pure numpy.
- `equal_error_threshold(scores, labels)`: score threshold where FAR is about equal to FRR.

The three verification metrics are numpy code. But `speechonnxmetrics.speaker`
imports `speakeronnx`, so the import needs the `speaker` extra for all four functions ([issue #7](https://github.com/TigreGotico/speechonnxmetrics/issues/7)).

```python
from speechonnxmetrics.speaker import eer, equal_error_threshold, min_dcf

scores = [0.9, 0.8, 0.4, 0.3, 0.6, 0.2]
labels = [1, 1, 1, 0, 0, 0]
print(eer(scores, labels))  # 0.33333333333333337
print(min_dcf(scores, labels))  # 0.33333333333333337
print(equal_error_threshold(scores, labels))  # 0.6
```

## Metrics the package does not give

- **PESQ.** The ITU-T P.862 licence terms do not suit an open package on PyPI. Use a
  PESQ package under your own licence review.
- **UTMOSv2.** Its published score is a mean over five folds and five random 3 s
  crops. An export of one fold and one crop is a different estimator.

---
[← CLI](cli.md) · [Home](index.md) · [Lect fidelity →](lect-fidelity.md)
