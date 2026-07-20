"""Speaker similarity metrics: cosine similarity and verification metrics (EER, minDCF).

Embedding extraction requires the optional ``speaker`` extra
(``pip install speechonnxmetrics[speaker]``); ``eer``, ``min_dcf`` and
``equal_error_threshold`` are pure numpy and work without it.
"""
from speechonnxmetrics.speaker.similarity import (
    SpeakerSimilarity,
    eer,
    equal_error_threshold,
    min_dcf,
    speaker_similarity,
)

__all__ = [
    "SpeakerSimilarity",
    "speaker_similarity",
    "eer",
    "min_dcf",
    "equal_error_threshold",
]
