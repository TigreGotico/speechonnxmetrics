"""Speaker similarity and speaker-verification metrics, built on speakeronnx.

Requires the optional ``speaker`` extra (``pip install speechonnxmetrics[speaker]``)
for embedding extraction.
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
