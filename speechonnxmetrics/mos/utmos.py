"""UTMOS — no-reference naturalness MOS prediction for synthesised speech.

Reference: T. Saeki, D. Xin, W. Nakata, T. Koshinaka and S. Takamichi, "UTMOS:
UTokyo-SaruLab System for VoiceMOS Challenge 2022", Interspeech 2022. The weights
are the ``utmos22_strong`` learner, ported by ``tarepan/SpeechMOS`` from the
original ``sarulab-speech/UTMOS22`` checkpoint; both are MIT licensed.

The exported graph carries the whole pipeline — wav2vec2 frontend, LSTM, mean
pooling and the affine rescale onto the MOS range. There is no frontend beyond
"16 kHz mono float32" and no post-processing: the graph does no resampling, so
callers must supply 16 kHz audio, which the base class guarantees.
"""
from __future__ import annotations

import numpy as np

from speechonnxmetrics.base import ModelEntry, OnnxMetric, Score

SR = 16000
HF_REPO = "TigreGotico/utmos-onnx"

#: Receptive field of the wav2vec2 convolutional feature extractor. Anything shorter
#: produces no frames at all and makes the graph fail inside a Conv node, so shorter
#: audio is zero-padded up to this length rather than surfacing an opaque ONNX error.
MIN_SAMPLES = 400


class UTMOS(OnnxMetric):
    """UTMOS naturalness MOS: one scalar on the 1-5 scale, higher is more natural."""

    name = "utmos"
    intrusive = False
    range = (1.0, 5.0)
    higher_is_better = True

    def __init__(self, **kwargs: object) -> None:
        entry = ModelEntry(
            alias="utmos",
            hf_repo=HF_REPO,
            hf_file="utmos22_strong.onnx",
            license="MIT",
            sample_rate=SR,
            description="UTMOS22 strong-learner naturalness MOS predictor",
        )
        super().__init__(entry, **kwargs)  # type: ignore[arg-type]

    def _frontend(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0:
            raise ValueError("cannot score empty audio")
        if x.size < MIN_SAMPLES:
            x = np.pad(x, (0, MIN_SAMPLES - x.size))
        return {"wave": x[np.newaxis, :]}

    def _postprocess(self, outputs: list[np.ndarray]) -> Score:
        mos = float(np.asarray(outputs[0], dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(mos):
            raise ValueError("UTMOS produced a non-finite score")
        return mos
