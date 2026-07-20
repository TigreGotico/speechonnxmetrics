"""Speaker similarity and speaker-verification metrics, built on speakeronnx.

Requires the optional ``speaker`` extra (``pip install speechonnxmetrics[speaker]``)
for embedding extraction. ``eer``, ``min_dcf`` and ``equal_error_threshold`` are pure
numpy and never touch speakeronnx.
"""
from __future__ import annotations

from functools import cached_property
from typing import Optional

import numpy as np

try:
    from speakeronnx import DEFAULT_MODEL, SpeakerEmbedder, cosine
except ImportError as exc:
    raise ImportError(
        "speechonnxmetrics.speaker requires the 'speaker' extra: "
        "pip install speechonnxmetrics[speaker]"
    ) from exc


def speaker_similarity(
    deg: np.ndarray,
    sr: int,
    *,
    ref: np.ndarray,
    ref_sr: Optional[int] = None,
    model: Optional[str] = None,
) -> float:
    """Cosine similarity between the speaker embeddings of ``deg`` and ``ref``.

    ``deg``/``ref`` are 1-D float32 waveforms; ``ref_sr`` defaults to ``sr`` when
    both audios already share the same sample rate. ``model`` is a speakeronnx
    registry alias or ``.onnx`` path, defaulting to speakeronnx's ``DEFAULT_MODEL``.
    """
    embedder = SpeakerEmbedder(model=model or DEFAULT_MODEL)
    deg_emb = embedder.embed(deg if sr == embedder.sample_rate else _resample(deg, sr, embedder.sample_rate))
    ref_rate = ref_sr if ref_sr is not None else sr
    ref_emb = embedder.embed(ref if ref_rate == embedder.sample_rate else _resample(ref, ref_rate, embedder.sample_rate))
    return cosine(deg_emb, ref_emb)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_out = int(len(audio) * ratio)
    x_old = np.linspace(0, len(audio) - 1, len(audio))
    x_new = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class SpeakerSimilarity:
    """Reusable speaker-similarity scorer with a lazily-loaded embedder.

    Construction never touches the network or disk — the underlying
    :class:`speakeronnx.SpeakerEmbedder` (and its ONNX model download) is created
    only on first use, via a cached property.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self._model = model or DEFAULT_MODEL

    @cached_property
    def embedder(self) -> SpeakerEmbedder:
        return SpeakerEmbedder(model=self._model)

    def score(
        self,
        deg: np.ndarray,
        sr: int,
        *,
        ref: np.ndarray,
        ref_sr: Optional[int] = None,
    ) -> float:
        """Cosine similarity between ``deg`` and ``ref``, loading the embedder on first call."""
        target_sr = self.embedder.sample_rate
        deg_emb = self.embedder.embed(deg if sr == target_sr else _resample(deg, sr, target_sr))
        ref_rate = ref_sr if ref_sr is not None else sr
        ref_emb = self.embedder.embed(ref if ref_rate == target_sr else _resample(ref, ref_rate, target_sr))
        return cosine(deg_emb, ref_emb)


def _validate_scores_labels(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.shape != labels.shape:
        raise ValueError(f"scores and labels must have matching shapes: {scores.shape} != {labels.shape}")
    if scores.size == 0:
        raise ValueError("scores/labels must not be empty")
    if scores.size == 1:
        raise ValueError("scores/labels must contain more than one sample")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must not contain NaN or inf")
    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels <= {0, 1}:
        raise ValueError(f"labels must be binary (0=impostor, 1=genuine), got {sorted(unique_labels)}")
    if len(unique_labels) < 2:
        raise ValueError("labels must contain both genuine (1) and impostor (0) samples")
    return scores, labels.astype(np.int64)


def _roc(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False-accept-rate and false-reject-rate at every distinct score threshold, descending.

    Tied scores are grouped: the threshold sits at the tied value and every sample
    sharing it is accepted together, rather than splitting ties across thresholds.
    """
    n_genuine = int(labels.sum())
    n_impostor = int(labels.size - n_genuine)

    thresholds = np.unique(scores)[::-1]
    # accepted(t) = samples with score >= t; searchsorted on ascending scores finds the
    # count of samples strictly below t, i.e. the count NOT accepted.
    scores_ascending = np.sort(scores)
    n_below = np.searchsorted(scores_ascending, thresholds, side="left")
    n_accepted = scores.size - n_below

    genuine_sorted_by_score = labels[np.argsort(scores)]
    cum_genuine_ascending = np.concatenate(([0], np.cumsum(genuine_sorted_by_score)))
    genuine_below = cum_genuine_ascending[n_below]
    genuine_accepted = n_genuine - genuine_below
    impostor_accepted = n_accepted - genuine_accepted

    far = impostor_accepted / n_impostor
    frr = 1.0 - genuine_accepted / n_genuine
    return thresholds, far, frr


def equal_error_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Score threshold at which the false-accept and false-reject rates are closest."""
    scores, labels = _validate_scores_labels(scores, labels)
    thresholds, far, frr = _roc(scores, labels)
    idx = int(np.argmin(np.abs(far - frr)))
    return float(thresholds[idx])


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal error rate: the point where false-accept rate equals false-reject rate.

    ``scores`` are verification scores (higher = more likely genuine); ``labels``
    are 1 for genuine pairs and 0 for impostor pairs.
    """
    scores, labels = _validate_scores_labels(scores, labels)
    _, far, frr = _roc(scores, labels)
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0)


def min_dcf(scores: np.ndarray, labels: np.ndarray, p_target: float = 0.01) -> float:
    """Minimum detection cost function (NIST SRE convention, c_miss = c_fa = 1).

    ``normDCF = min_threshold[ p_target * frr + (1 - p_target) * far ] / min(p_target, 1 - p_target)``.
    """
    scores, labels = _validate_scores_labels(scores, labels)
    if not 0.0 < p_target < 1.0:
        raise ValueError(f"p_target must be in (0, 1), got {p_target}")
    _, far, frr = _roc(scores, labels)
    dcf = p_target * frr + (1.0 - p_target) * far
    normalizer = min(p_target, 1.0 - p_target)
    return float(dcf.min() / normalizer)
