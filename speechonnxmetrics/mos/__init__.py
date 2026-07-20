"""No-reference MOS metrics: DNSMOS (P.835 and P.808), SIGMOS (P.804), UTMOS and NISQA.

Each metric is an :class:`~speechonnxmetrics.base.OnnxMetric` subclass, so importing
this module builds no inference session and touches neither disk nor network — the
ONNX weights are fetched on first call.

NISQA's weights are CC BY-NC-SA 4.0 (NonCommercial) while every other model here is
MIT — see ``docs/models.md`` for the per-model licence split. Nothing gates its use;
the choice is the caller's.
"""
from __future__ import annotations

from speechonnxmetrics.mos.dnsmos import DNSMOS, DNSMOSP808
from speechonnxmetrics.mos.nisqa import NISQA
from speechonnxmetrics.mos.sigmos import SIGMOS
from speechonnxmetrics.mos.utmos import UTMOS

__all__ = ["DNSMOS", "DNSMOSP808", "NISQA", "SIGMOS", "UTMOS"]
