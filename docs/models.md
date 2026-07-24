# Models

Every ONNX model this package downloads, and what it is. Each model lives in its own
HuggingFace repo under the `TigreGotico` org and is fetched on first use into
`$XDG_DATA_HOME/speechonnxmetrics`. Constructing a metric downloads nothing — the
session, and therefore the download, is created lazily on the first call.

Every `ModelEntry` is pinned to an immutable commit SHA (`revision=`), not a branch —
an upstream update to a model repo can never silently change scores already in use.
Bumping a pin is a deliberate, reviewed code change in its own commit. Each `OnnxMetric`
instance exposes its exact provenance via the `.model_info` property (`repo_id`,
`filename`, `revision`), so downstream evaluators can record which judge weights
produced each score.

## Licence at a glance

Licences differ per model, so check this before selecting metrics.

| licence | metrics | what it means for you |
|---|---|---|
| **MIT** | `dnsmos`, `dnsmos_p808`, `sigmos`, `utmos` | Commercial use, redistribution and modification all permitted. Keep the attribution. |
| **CC BY-NC-SA 4.0 — NonCommercial** | `nisqa` | **Commercial use is forbidden.** ShareAlike also applies: derivatives must carry the same licence. Only the NISQA *weights* are NC — its code is MIT. |

Only NISQA carries NonCommercial terms. If your use is commercial, every other metric
here is safe and NISQA is not. This package makes no choice for you — it exposes the
metric and states the terms; selecting it is the downstream user's call.

## No-reference MOS predictors

| metric | HF repo | file | pinned revision | licence | upstream source | rate | output |
|---|---|---|---|---|---|---|---|
| `dnsmos` | `TigreGotico/dnsmos-onnx` | `sig_bak_ovr.onnx` | `27691a53aa069b27be6ac957013d43b3c442da9d` | MIT | [`microsoft/DNS-Challenge`](https://github.com/microsoft/DNS-Challenge) `DNSMOS/` | 16 kHz | `sig`, `bak`, `ovrl` — ITU-T P.835 speech quality, background intrusiveness and overall quality, each 1-5 |
| `dnsmos` (`personalized=True`) | `TigreGotico/dnsmos-onnx` | `personalized/pdnsmos_sig_bak_ovr.onnx` | `27691a53aa069b27be6ac957013d43b3c442da9d` | MIT | same, personalized-noise-suppression track | 16 kHz | as above, with the personalized score-fitting polynomials |
| `dnsmos_p808` | `TigreGotico/dnsmos-onnx` | `model_v8.onnx` | `27691a53aa069b27be6ac957013d43b3c442da9d` | MIT | same | 16 kHz | single ITU-T P.808 crowdsourced-listening MOS, 1-5 |
| `sigmos` | `TigreGotico/sigmos-onnx` | `model-sigmos_1697718653_41d092e8-epo-200.onnx` | `33ccd4fca5b8ffe03828530753f0b35769b8e880` | MIT | [`microsoft/SIG-Challenge`](https://github.com/microsoft/SIG-Challenge) `ICASSP2024/sigmos/` | 48 kHz | ITU-T P.804 dimensions `col`, `disc`, `loud`, `noise`, `reverb`, `sig`, `ovrl`, each 1-5 |
| `utmos` | `TigreGotico/utmos-onnx` | `utmos22_strong.onnx` | `ff41b8f440cb12ecda18261f9ff7326d058275ce` | MIT | exported here from [`tarepan/SpeechMOS`](https://github.com/tarepan/SpeechMOS) `utmos22_strong`, itself a port of [`sarulab-speech/UTMOS22`](https://github.com/sarulab-speech/UTMOS22) | 16 kHz | single naturalness MOS, 1-5 |
| `nisqa` | `TigreGotico/nisqa-onnx` | `nisqa.onnx` | `3de0221b7bb4919dc2ba9a891da7fba76b06e573` | code MIT, **weights CC BY-NC-SA 4.0 (NonCommercial)** | exported here from [`gabrielmittag/NISQA`](https://github.com/gabrielmittag/NISQA) `weights/nisqa.tar` (NISQAv2) | native, any rate | `mos`, `noi`, `dis`, `col`, `loud`, each 1-5 |

NISQA lives in its own HF repo because its weights are the only NonCommercial artifact
in the set. Its output order is `mos`, `noi`, `dis`, `col`, `loud`, taken from
`NISQA_lib.py`; the upstream `NISQA_DIM` class docstring lists coloration before
discontinuity and is wrong.

NISQA is also the only rate-adaptive metric: it never resamples, and derives its 10 ms
mel hop and 20 ms window from the file's own sample rate, as the reference does.

Unused DNSMOS sub-models are mirrored for completeness but are not wired to a metric:
`sig.onnx` and `bak_ovr.onnx` are the older two-model P.835 split that
`sig_bak_ovr.onnx` supersedes.

### Defining papers

* DNSMOS — C. K. A. Reddy, V. Gopal, R. Cutler, *DNSMOS: A Non-Intrusive Perceptual
  Objective Speech Quality Metric to Evaluate Noise Suppressors*, ICASSP 2021; and
  *DNSMOS P.835*, ICASSP 2022.
* SIGMOS — N.-C. Ristea, A. Saabas, R. Cutler et al., *ICASSP 2024 Speech Signal
  Improvement Challenge*.
* UTMOS — T. Saeki, D. Xin, W. Nakata, T. Koshinaka, S. Takamichi, *UTMOS:
  UTokyo-SaruLab System for VoiceMOS Challenge 2022*, Interspeech 2022.
* NISQA — G. Mittag, B. Naderi, A. Chehadi, S. Möller, *NISQA: A Deep CNN-Self-Attention
  Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets*,
  Interspeech 2021.

## Score post-processing

DNSMOS is the only metric with client-side post-processing: the network emits scores on
an uncalibrated internal scale, and a per-dimension polynomial maps them onto MOS.
Omitting it yields plausible-looking but wrong numbers. SIGMOS and UTMOS need none —
UTMOS carries its affine rescale inside the exported graph.

Long audio is handled per model. DNSMOS scores 9.01 s windows at a 1 s hop and averages,
tiling audio shorter than one window; SIGMOS and UTMOS score the whole utterance in one
pass. NISQA segments its mel spectrogram into 15-frame patches at a 4-frame hop and
pools them inside the graph, so audio shorter than 15 mel frames is rejected rather
than padded.

## Models deliberately absent

UTMOSv2 is not offered, for a technical reason rather than a licensing one: its published
score is an average over five folds × five random 3 s crops, so any single-fold
single-crop export would be a different estimator rather than an approximation of the
published numbers.

## Reproducing the UTMOS export

`conversion/export_utmos.py` regenerates `utmos22_strong.onnx` from the torch.hub
checkpoint. DNSMOS and SIGMOS are shipped as ONNX upstream and are mirrored verbatim;
NISQA was exported from its PyTorch checkpoint at opset 17.
