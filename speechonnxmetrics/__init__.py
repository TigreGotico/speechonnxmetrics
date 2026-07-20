"""speechonnxmetrics — unified speech evaluation metrics (MOS, intrusive, ASR, speaker)."""
from __future__ import annotations

from speechonnxmetrics.api import score, score_batch
from speechonnxmetrics.base import Metric, ModelEntry, OnnxMetric
from speechonnxmetrics.registry import RegistryEntry, get, list_metrics, register
from speechonnxmetrics.version import __version__

__all__ = [
    "__version__",
    "score",
    "score_batch",
    "Metric",
    "ModelEntry",
    "OnnxMetric",
    "RegistryEntry",
    "get",
    "list_metrics",
    "register",
]
