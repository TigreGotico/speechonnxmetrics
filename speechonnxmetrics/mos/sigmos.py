"""SIGMOS — no-reference multi-dimensional speech quality prediction (ITU-T P.804).

Predicts seven perceptual dimensions in one pass: coloration, discontinuity,
loudness, noisiness, reverberation, signal quality and overall quality.

Reference: N.-C. Ristea, A. Saabas, R. Cutler et al., "ICASSP 2024 Speech Signal
Improvement Challenge", and the P.804 SIGMOS estimator released with it.

Weights are Microsoft's, MIT licensed (``SIG-Challenge/LICENSE``).
"""
from __future__ import annotations

import numpy as np

from speechonnxmetrics._dsp.stft import stft
from speechonnxmetrics.base import ModelEntry, OnnxMetric, Score

#: SIGMOS is defined at 48 kHz.
SR = 48000
DFT_SIZE = 960
FRAME_SIZE = 480
WINDOW_LENGTH = 960
COMPRESS_FACTOR = 0.3

HF_REPO = "TigreGotico/sigmos-onnx"

#: Output order of the seven heads, fixed by the released estimator.
DIMENSIONS = ("col", "disc", "loud", "noise", "reverb", "sig", "ovrl")


def _sqrt_hann(length: int) -> np.ndarray:
    """Square-root periodic Hann analysis window, as the reference estimator uses."""
    return np.sqrt(np.hanning(length + 1)[:-1])


def features(audio: np.ndarray) -> np.ndarray:
    """``[1, 3, frames, 481]`` compressed magnitude-plus-complex STFT features.

    The three channels are the power-compressed magnitude and the matching
    compressed real and imaginary parts. Padding follows the reference estimator:
    ``window_length - frame_size`` leading samples and enough trailing samples to
    complete the final frame, with no centre padding.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("cannot score empty audio")
    last_frame = x.size % FRAME_SIZE or FRAME_SIZE
    padded = np.pad(x, (WINDOW_LENGTH - FRAME_SIZE, WINDOW_LENGTH - last_frame))
    spec = stft(
        padded, n_fft=DFT_SIZE, hop_size=FRAME_SIZE, win_size=WINDOW_LENGTH,
        center=False, window=_sqrt_hann(WINDOW_LENGTH),
    ).T  # [frames, 481]

    parts = np.stack([spec.real, spec.imag], axis=-2).astype(np.float32)  # [frames, 2, 481]
    power = np.maximum((parts * parts).sum(axis=-2, keepdims=True), 1e-12)
    parts = np.power(power, (COMPRESS_FACTOR - 1) / 2) * parts
    mag = np.power(power, COMPRESS_FACTOR / 2)
    stacked = np.concatenate((mag, parts), axis=-2)  # [frames, 3, 481]
    return np.expand_dims(np.transpose(stacked, (1, 0, 2)), 0).astype(np.float32)


class SIGMOS(OnnxMetric):
    """P.804 SIGMOS: seven perceptual quality dimensions, each on the 1-5 MOS scale.

    Returns a dict keyed ``col``, ``disc``, ``loud``, ``noise``, ``reverb``, ``sig``
    and ``ovrl``. Unlike DNSMOS the whole utterance is scored in a single pass —
    there is no windowing and no score-fitting polynomial.
    """

    name = "sigmos"
    intrusive = False
    range = (1.0, 5.0)
    higher_is_better = True

    def __init__(self, **kwargs: object) -> None:
        entry = ModelEntry(
            alias="sigmos",
            hf_repo=HF_REPO,
            hf_file="model-sigmos_1697718653_41d092e8-epo-200.onnx",
            license="MIT",
            sample_rate=SR,
            description="SIGMOS P.804 seven-dimension quality predictor",
        )
        super().__init__(entry, **kwargs)  # type: ignore[arg-type]

    def _frontend(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        feats = features(audio)
        return {inp.name: feats for inp in self.session.get_inputs()}

    def _postprocess(self, outputs: list[np.ndarray]) -> Score:
        raw = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        scores = dict(zip(DIMENSIONS, (float(v) for v in raw)))
        if not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("SIGMOS produced a non-finite score")
        return scores
