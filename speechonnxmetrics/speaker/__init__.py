"""Speaker similarity metrics: cosine similarity and verification metrics (EER, minDCF).

Embedding extraction requires the optional ``speaker`` extra
(``pip install speechonnxmetrics[speaker]``); ``eer``, ``min_dcf`` and
``equal_error_threshold`` are pure numpy and work without it.
"""
from speechonnxmetrics.speaker.verification import eer, equal_error_threshold, min_dcf

__all__ = [
    "SpeakerSimilarity",
    "speaker_similarity",
    "eer",
    "min_dcf",
    "equal_error_threshold",
]


def __getattr__(name: str):
    # Import speakeronnx only when an embedding-based name is requested.
    if name in ("SpeakerSimilarity", "speaker_similarity"):
        from speechonnxmetrics.speaker import similarity

        return getattr(similarity, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
