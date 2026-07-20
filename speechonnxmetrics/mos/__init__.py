"""No-reference MOS metrics: DNSMOS (P.835 and P.808), SIGMOS (P.804) and UTMOS.

Each metric is an :class:`~speechonnxmetrics.base.OnnxMetric` subclass, so importing
this module builds no inference session and touches neither disk nor network — the
ONNX weights are fetched on first call.

NISQA is not registered here yet; its ONNX export is still being verified. Note that
its weights are CC BY-NC-SA 4.0 (NonCommercial) while every model above is MIT — see
``docs/models.md`` for the per-model licence split.
"""
from __future__ import annotations

from speechonnxmetrics.mos.dnsmos import DNSMOS, DNSMOSP808
from speechonnxmetrics.mos.sigmos import SIGMOS
from speechonnxmetrics.mos.utmos import UTMOS

__all__ = ["DNSMOS", "DNSMOSP808", "SIGMOS", "UTMOS"]
