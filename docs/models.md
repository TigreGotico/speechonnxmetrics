# Models

Every ONNX model this package downloads, and what it is. Each model lives in its own
HuggingFace repo under the `TigreGotico` org and is fetched on first use into
`$XDG_DATA_HOME/speechonnxmetrics`. Constructing a metric downloads nothing — the
session, and therefore the download, is created lazily on the first call.

## Licence at a glance

Licences differ per model, so check this before selecting metrics.

| licence | metrics | what it means for you |
|---|---|---|
| **MIT** | `dnsmos`, `dnsmos_p808`, `sigmos`, `utmos` | Commercial use, redistribution and modification all permitted. Keep the attribution. |
| **CC BY-NC-SA 4.0 — NonCommercial** | `nisqa` *(pending)* | **Commercial use is forbidden.** ShareAlike also applies: derivatives must carry the same licence. Only the NISQA *weights* are NC — its code is MIT. |

Only NISQA carries NonCommercial terms. If your use is commercial, every other metric
here is safe and NISQA is not. This package makes no choice for you — it exposes the
metric and states the terms; selecting it is the downstream user's call.

## No-reference MOS predictors

| metric | HF repo | file | licence | upstream source | rate | output |
|---|---|---|---|---|---|---|
| `dnsmos` | `TigreGotico/dnsmos-onnx` | `sig_bak_ovr.onnx` | MIT | [`microsoft/DNS-Challenge`](https://github.com/microsoft/DNS-Challenge) `DNSMOS/` | 16 kHz | `sig`, `bak`, `ovrl` — ITU-T P.835 speech quality, background intrusiveness and overall quality, each 1-5 |
| `dnsmos` (`personalized=True`) | `TigreGotico/dnsmos-onnx` | `personalized/pdnsmos_sig_bak_ovr.onnx` | MIT | same, personalized-noise-suppression track | 16 kHz | as above, with the personalized score-fitting polynomials |
| `dnsmos_p808` | `TigreGotico/dnsmos-onnx` | `model_v8.onnx` | MIT | same | 16 kHz | single ITU-T P.808 crowdsourced-listening MOS, 1-5 |
| `sigmos` | `TigreGotico/sigmos-onnx` | `model-sigmos_1697718653_41d092e8-epo-200.onnx` | MIT | [`microsoft/SIG-Challenge`](https://github.com/microsoft/SIG-Challenge) `ICASSP2024/sigmos/` | 48 kHz | ITU-T P.804 dimensions `col`, `disc`, `loud`, `noise`, `reverb`, `sig`, `ovrl`, each 1-5 |
| `utmos` | `TigreGotico/utmos-onnx` | `utmos22_strong.onnx` | MIT | exported here from [`tarepan/SpeechMOS`](https://github.com/tarepan/SpeechMOS) `utmos22_strong`, itself a port of [`sarulab-speech/UTMOS22`](https://github.com/sarulab-speech/UTMOS22) | 16 kHz | single naturalness MOS, 1-5 |
| `nisqa` **(pending)** | `TigreGotico/nisqa-onnx` | *export in progress* | code MIT, **weights CC BY-NC-SA 4.0 (NonCommercial)** | [`gabrielmittag/NISQA`](https://github.com/gabrielmittag/NISQA) `weights/nisqa.tar` | 48 kHz | `mos`, `noi`, `dis`, `col`, `loud`, each 1-5 |

NISQA is not yet registered as a metric — its ONNX export is still being produced and
verified. It is kept in a separate repo precisely because its weights are the only
NonCommercial artifact in the set.

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
pass.

## Models deliberately absent

UTMOSv2 is not offered, for a technical reason rather than a licensing one: its published
score is an average over five folds × five random 3 s crops, so any single-fold
single-crop export would be a different estimator rather than an approximation of the
published numbers.

## Reproducing the UTMOS export

`conversion/export_utmos.py` regenerates `utmos22_strong.onnx` from the torch.hub
checkpoint. The other models are shipped as ONNX upstream and are mirrored verbatim.
