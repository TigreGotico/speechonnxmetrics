"""NISQA — no-reference multi-dimensional speech quality prediction.

Predicts an overall MOS plus the four perceptual dimensions of the NISQA-v2
multidimensional model: noisiness, discontinuity, coloration and loudness.

Reference: G. Mittag, B. Naderi, A. Chehadi and S. Möller, "NISQA: A Deep
CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with
Crowdsourced Datasets", Interspeech 2021.

NISQA's code is MIT licensed, but its **weights are CC BY-NC-SA 4.0** — the only
NonCommercial artifact in this package. Commercial use of this metric is forbidden by
the weights' licence and derivatives must carry the same terms; selecting it is the
caller's decision, and nothing here blocks it.
"""
from __future__ import annotations

import numpy as np

from speechonnxmetrics._dsp.mel import melspectrogram
from speechonnxmetrics.base import ModelEntry, OnnxMetric, Score

HF_REPO = "TigreGotico/nisqa-onnx"
#: pinned immutable commit SHA — an upstream model update must never silently
#: change scores; bump deliberately, in its own commit, if the weights change.
HF_REVISION = "3de0221b7bb4919dc2ba9a891da7fba76b06e573"

#: Mel frontend, as the checkpoint's own ``args`` define it. The hop and window are
#: given in *seconds* upstream and derived per file from its native rate, so NISQA
#: never resamples and the metric's ``sample_rate`` is ``None``.
N_FFT = 4096
HOP_SECONDS = 0.01
WIN_SECONDS = 0.02
N_MELS = 48
FMIN = 0.0
#: 20 kHz exceeds Nyquist for anything below 40 kHz, so the upper mel filters come out
#: empty on 16 kHz audio (librosa warns about exactly this). That is upstream's
#: configured value and the weights were trained through it — reproduced deliberately.
FMAX = 20000.0
AMIN = 1e-4
TOP_DB = 80.0

#: Segmentation of the ``[48, T]`` spectrogram into CNN patches.
SEG_LENGTH = 15
SEG_HOP = 4

#: Output order, from ``NISQA_lib.py`` (``mos_pred`` … ``loud_pred``, columns 0-4).
#: The ``NISQA_DIM`` class docstring disagrees — it lists coloration before
#: discontinuity — but the docstring is wrong and the code is right. Do not "fix"
#: this to match the docstring.
DIMENSIONS = ("mos", "noi", "dis", "col", "loud")


def melspec(audio: np.ndarray, sr: int) -> np.ndarray:
    """``[48, T]`` dB-scaled magnitude mel spectrogram, in librosa's semantics.

    Equivalent to ``librosa.feature.melspectrogram(power=1.0, htk=False,
    norm='slaney')`` followed by ``librosa.amplitude_to_db(ref=1.0, amin=1e-4,
    top_db=80.0)``, computed through the package's single shared STFT/mel.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("cannot score empty audio")
    # Reflect padding needs at least n_fft//2 + 1 samples to mirror, and the CNN needs
    # SEG_LENGTH frames; check both here so short audio fails with a NISQA-level
    # message rather than deep inside the strided STFT.
    hop = int(sr * HOP_SECONDS)
    if x.size < N_FFT // 2 + 1 or 1 + x.size // hop < SEG_LENGTH:
        raise ValueError(
            f"audio too short for NISQA: {x.size} samples at {sr} Hz, "
            f"at least {max(N_FFT // 2 + 1, (SEG_LENGTH - 1) * hop)} are needed"
        )
    mel = melspectrogram(
        x, sr, n_fft=N_FFT, hop_size=int(sr * HOP_SECONDS), win_size=int(sr * WIN_SECONDS),
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, norm="slaney", power=1.0, pad_mode="reflect",
    ).astype(np.float64)
    db = 20.0 * np.log10(np.maximum(mel, AMIN))
    return np.maximum(db, db.max() - TOP_DB)


def segment(spec: np.ndarray) -> np.ndarray:
    """``[n_segments, 1, 48, 15]`` CNN patches from a ``[48, T]`` spectrogram.

    Sliding 15-frame windows taken every ``SEG_HOP`` frames, giving
    ``ceil((T - 14) / 4)`` segments. There is no padding: the exported graph is
    padding-free and carries no ``n_wins`` input.
    """
    n_wins = spec.shape[1] - (SEG_LENGTH - 1)
    if n_wins < 1:
        raise ValueError(
            f"audio too short for NISQA: {spec.shape[1]} mel frames, "
            f"at least {SEG_LENGTH} are needed"
        )
    idx = np.arange(SEG_LENGTH)[None, :] + np.arange(0, n_wins, SEG_HOP)[:, None]
    return spec[:, idx].transpose(1, 0, 2)[:, None].astype(np.float32)


class NISQA(OnnxMetric):
    """NISQA-v2: overall ``mos`` plus ``noi``, ``dis``, ``col`` and ``loud``, each 1-5.

    Rate-adaptive: audio is scored at whatever rate it arrives at, with the mel hop
    and window derived from that rate, exactly as the reference implementation does.
    """

    name = "nisqa"
    intrusive = False
    range = (1.0, 5.0)
    higher_is_better = True

    def __init__(self, **kwargs: object) -> None:
        entry = ModelEntry(
            alias="nisqa",
            hf_repo=HF_REPO,
            hf_file="nisqa.onnx",
            revision=HF_REVISION,
            license="CC BY-NC-SA 4.0 (NonCommercial; NISQA's code is MIT)",
            sample_rate=None,
            description="NISQA-v2 five-dimension quality predictor (NonCommercial weights)",
        )
        super().__init__(entry, **kwargs)  # type: ignore[arg-type]

    def _frontend(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        return {"segments": segment(melspec(audio, sr))[None]}

    def _postprocess(self, outputs: list[np.ndarray]) -> Score:
        raw = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        scores = dict(zip(DIMENSIONS, (float(v) for v in raw)))
        if not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("NISQA produced a non-finite score")
        return scores
