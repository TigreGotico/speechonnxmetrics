"""DNSMOS — no-reference speech quality prediction (ITU-T P.835 and P.808).

Two models, two standards, registered as two metrics:

* :class:`DNSMOS` wraps ``sig_bak_ovr.onnx``, the P.835 model predicting the
  speech-signal, background-noise and overall quality triplet.
* :class:`DNSMOSP808` wraps ``model_v8.onnx``, the P.808 model predicting a single
  crowdsourced-listening MOS.

Reference: C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS: A Non-Intrusive
Perceptual Objective Speech Quality Metric to Evaluate Noise Suppressors",
ICASSP 2021, and "DNSMOS P.835: A Non-Intrusive Perceptual Objective Speech
Quality Metric to Evaluate Noise Suppressors", ICASSP 2022.

Weights are Microsoft's, MIT licensed (``DNS-Challenge/LICENSE-CODE``).
"""
from __future__ import annotations

import numpy as np

from speechonnxmetrics._dsp.mel import melspectrogram
from speechonnxmetrics.base import ModelEntry, OnnxMetric, Score

#: DNSMOS is defined at 16 kHz on windows of exactly 9.01 s.
SR = 16000
INPUT_LENGTH = 9.01
WINDOW_SAMPLES = int(INPUT_LENGTH * SR)  # 144160

HF_REPO = "TigreGotico/dnsmos-onnx"
#: pinned immutable commit SHA — an upstream model update must never silently
#: change scores; bump deliberately, in its own commit, if the weights change.
#: Shared by both DNSMOS variants (P.835 and P.808) — one repo, one revision.
HF_REVISION = "27691a53aa069b27be6ac957013d43b3c442da9d"

#: Cubic/quadratic score-fitting polynomials mapping raw network outputs onto the
#: MOS scale, in ``np.poly1d`` (highest-order-first) order. Omitting these leaves
#: scores on an uncalibrated internal scale.
POLYFIT = {
    "sig": (-0.08397278, 1.22083953, 0.0052439),
    "bak": (-0.13166888, 1.60915514, -0.39604546),
    "ovrl": (-0.06766283, 1.11546468, 0.04602535),
}
POLYFIT_PERSONALIZED = {
    "sig": (-0.01019296, 0.02751166, 1.19576786, -0.24348726),
    "bak": (-0.04976499, 0.44276479, -0.1644611, 0.96883132),
    "ovrl": (-0.00533021, 0.005101, 1.18058466, -0.11236046),
}


def segment(audio: np.ndarray) -> np.ndarray:
    """Split ``audio`` into the ``[N, 144160]`` stack of 9.01 s windows DNSMOS scores.

    Audio shorter than one window is repeatedly concatenated with itself until it
    covers a window; longer audio is cut into 9.01 s windows at a 1 s hop, and a
    trailing partial window is dropped. This is the reference segmentation — the
    per-window scores are averaged afterwards.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("cannot score empty audio")
    while x.size < WINDOW_SAMPLES:
        x = np.concatenate([x, x])
    n_hops = int(np.floor(x.size / SR) - INPUT_LENGTH) + 1
    windows = [
        x[i * SR: i * SR + WINDOW_SAMPLES] for i in range(n_hops)
        if x.size - i * SR >= WINDOW_SAMPLES
    ]
    return np.stack(windows).astype(np.float32)


class DNSMOS(OnnxMetric):
    """P.835 DNSMOS: speech quality (``sig``), background intrusiveness (``bak``) and
    overall quality (``ovrl``), each on the 1-5 MOS scale.

    ``personalized=True`` selects the personalized-noise-suppression variant, which
    has its own weights and its own score-fitting polynomials.
    """

    name = "dnsmos"
    intrusive = False
    range = (1.0, 5.0)
    higher_is_better = True

    def __init__(self, personalized: bool = False, **kwargs: object) -> None:
        entry = ModelEntry(
            alias="dnsmos_personalized" if personalized else "dnsmos",
            hf_repo=HF_REPO,
            hf_file=(
                "personalized/pdnsmos_sig_bak_ovr.onnx" if personalized
                else "sig_bak_ovr.onnx"
            ),
            revision=HF_REVISION,
            license="MIT",
            sample_rate=SR,
            description="DNSMOS P.835 sig/bak/ovrl predictor",
        )
        super().__init__(entry, **kwargs)  # type: ignore[arg-type]
        self.personalized = personalized
        self.name = "dnsmos_personalized" if personalized else "dnsmos"

    def _frontend(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        return {"input_1": segment(audio)}

    def _postprocess(self, outputs: list[np.ndarray]) -> Score:
        raw = np.asarray(outputs[0], dtype=np.float64)  # [N, 3] = sig, bak, ovrl
        coeffs = POLYFIT_PERSONALIZED if self.personalized else POLYFIT
        scores = {
            key: float(np.mean(np.polyval(coeffs[key], raw[:, i])))
            for i, key in enumerate(("sig", "bak", "ovrl"))
        }
        if not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("DNSMOS produced a non-finite score")
        return scores


class DNSMOSP808(OnnxMetric):
    """P.808 DNSMOS: a single crowdsourced-listening MOS on the 1-5 scale.

    The frontend is a 120-band log-mel spectrogram of the first 9 s of each 9.01 s
    window (the reference drops the trailing 160 samples so the model sees exactly
    900 frames), power-to-dB referenced to the window maximum and clamped 80 dB
    below it, then affinely mapped to roughly ``[-1, 0]``.
    """

    name = "dnsmos_p808"
    intrusive = False
    range = (1.0, 5.0)
    higher_is_better = True

    def __init__(self, **kwargs: object) -> None:
        entry = ModelEntry(
            alias="dnsmos_p808",
            hf_repo=HF_REPO,
            hf_file="model_v8.onnx",
            revision=HF_REVISION,
            license="MIT",
            sample_rate=SR,
            description="DNSMOS P.808 MOS predictor",
        )
        super().__init__(entry, **kwargs)  # type: ignore[arg-type]

    def _frontend(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        windows = segment(audio)
        feats = np.stack([_p808_melspec(w[:-160]) for w in windows])
        return {"input_1": feats.astype(np.float32)}

    def _postprocess(self, outputs: list[np.ndarray]) -> Score:
        mos = float(np.mean(np.asarray(outputs[0], dtype=np.float64)))
        if not np.isfinite(mos):
            raise ValueError("DNSMOS P.808 produced a non-finite score")
        return mos


def _p808_melspec(window: np.ndarray) -> np.ndarray:
    """``[900, 120]`` log-mel feature for one P.808 window, via the shared STFT/mel."""
    mel = melspectrogram(
        window, SR, n_fft=321, hop_size=160, n_mels=120, norm="slaney",
        power=2.0, pad_mode="constant",
    )
    db = 10.0 * np.log10(np.maximum(mel.astype(np.float64), 1e-10))
    db -= 10.0 * np.log10(max(float(mel.max()), 1e-10))
    db = np.maximum(db, db.max() - 80.0)
    return ((db + 40.0) / 40.0).T
