"""Shared input handling for every intrusive (reference-requiring) metric.

Every metric in :mod:`speechonnxmetrics.intrusive` shares one signature shape —
``metric(deg, sr, *, ref, ref_sr=None)`` — and one policy for getting a reference and
a degraded signal onto a common sample rate and a common length before scoring.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

import numpy as np

from speechonnxmetrics._dsp.audio import AudioLoadError, load_audio
from speechonnxmetrics._dsp.resample import kaiser_resample

AudioLike = Union[str, Path, bytes, np.ndarray]

# Above this ratio the shorter signal is truncated to match, with a warning.
_LENGTH_WARN_RATIO = 1.05
# Above this ratio the lengths are treated as mismatched files, not a trim-worthy
# recording-boundary difference, and scoring is refused.
_LENGTH_RAISE_RATIO = 2.0


class IntrusiveMetricError(ValueError):
    """Raised when an intrusive metric cannot be scored: silent reference, mismatched
    or too-short audio, or non-finite samples."""


def _load(source: AudioLike, sr: int | None) -> tuple[np.ndarray, int]:
    try:
        audio, out_sr = load_audio(source, sample_rate=sr)
    except AudioLoadError as exc:
        raise IntrusiveMetricError(str(exc)) from exc
    return audio.astype(np.float64), out_sr


def load_pair(
    deg: AudioLike, sr: int, ref: AudioLike, ref_sr: int | None, target_sr: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load ``deg``/``ref``, resample both onto one rate, and length-align them.

    ``ref_sr`` defaults to ``sr``. ``target_sr`` overrides the common analysis rate
    (e.g. STOI's fixed 10 kHz); by default both signals are resampled onto ``sr``.
    Length mismatch is truncated to the shorter signal, with a warning once the longer
    is more than ``1.05x`` the shorter, and raises :class:`IntrusiveMetricError` past
    ``2x`` — that ratio is a strong signal the caller paired the wrong two files rather
    than two takes of the same recording.
    """
    deg_audio, deg_sr = _load(deg, sr)
    ref_audio, ref_sr_loaded = _load(ref, ref_sr if ref_sr is not None else sr)

    common_sr = target_sr if target_sr is not None else deg_sr
    if deg_sr != common_sr:
        deg_audio = kaiser_resample(deg_audio.astype(np.float32), deg_sr, common_sr).astype(np.float64)
    if ref_sr_loaded != common_sr:
        ref_audio = kaiser_resample(ref_audio.astype(np.float32), ref_sr_loaded, common_sr).astype(np.float64)

    n_deg, n_ref = deg_audio.size, ref_audio.size
    if n_deg == 0 or n_ref == 0:
        raise IntrusiveMetricError("reference and degraded audio must both be non-empty")

    shorter, longer = min(n_deg, n_ref), max(n_deg, n_ref)
    ratio = longer / shorter
    if ratio > _LENGTH_RAISE_RATIO:
        raise IntrusiveMetricError(
            f"reference ({n_ref} samples) and degraded ({n_deg} samples) audio differ by "
            f"{ratio:.2f}x at the matched sample rate — this usually means mismatched files"
        )
    if ratio > _LENGTH_WARN_RATIO:
        warnings.warn(
            f"truncating reference/degraded audio to the shorter length ({shorter} samples, "
            f"{ratio:.2f}x mismatch) before scoring", stacklevel=3,
        )
    n = shorter
    return deg_audio[:n], ref_audio[:n], common_sr


def check_finite_nonzero(ref: np.ndarray, deg: np.ndarray) -> None:
    """Raise on non-finite samples or a silent (all-zero) reference — every metric here
    divides by a reference-derived energy term at some point, so a silent reference is
    undefined rather than a degenerate zero."""
    if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(deg))):
        raise IntrusiveMetricError("reference or degraded audio contains NaN/inf")
    if not np.any(ref):
        raise IntrusiveMetricError("reference audio is all-zero — the metric is undefined")
