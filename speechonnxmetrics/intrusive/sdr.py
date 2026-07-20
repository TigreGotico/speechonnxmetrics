"""Scale-invariant and plain signal-to-distortion/noise ratios.

``si_sdr`` follows Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R. (2019).
"SDR – half-baked or well done?" ICASSP — the orthogonal-projection form: the
degraded signal is decomposed into a scaled copy of the reference (the "target") and
a residual, and the ratio is taken between their energies. ``sdr`` and ``snr`` are the
plain (non-scale-invariant) waveform ratios that ``si_sdr`` is defined against; for a
single-reference pair without permutation/scaling ambiguity they reduce to the same
target/residual-energy ratio.
"""
from __future__ import annotations

import numpy as np

from speechonnxmetrics.intrusive._common import AudioLike, check_finite_nonzero, load_pair

# A near-zero residual (identical signals, or a mathematically perfect scale-invariant
# estimate) is capped rather than reported as +inf — the true ratio is unbounded, but
# an unbounded float is worse for callers (plotting, thresholding, aggregation) than a
# documented ceiling far above any signal that could occur in practice.
_CAP_DB = 100.0
_EPS = 1e-12


def _db_ratio(signal_energy: float, error_energy: float) -> float:
    if error_energy < _EPS * max(signal_energy, _EPS):
        return _CAP_DB
    return float(min(10.0 * np.log10(signal_energy / error_energy), _CAP_DB))


def si_sdr(deg: AudioLike, sr: int, *, ref: AudioLike, ref_sr: int | None = None) -> float:
    """Scale-invariant SDR in dB (higher is better); +inf for a perfect estimate is
    capped at 100 dB (see module docstring)."""
    deg_a, ref_a, _ = load_pair(deg, sr, ref, ref_sr)
    check_finite_nonzero(ref_a, deg_a)
    alpha = np.dot(deg_a, ref_a) / np.dot(ref_a, ref_a)
    target = alpha * ref_a
    residual = deg_a - target
    return _db_ratio(float(np.dot(target, target)), float(np.dot(residual, residual)))


def sdr(deg: AudioLike, sr: int, *, ref: AudioLike, ref_sr: int | None = None) -> float:
    """Plain (non-scale-invariant) signal-to-distortion ratio in dB."""
    deg_a, ref_a, _ = load_pair(deg, sr, ref, ref_sr)
    check_finite_nonzero(ref_a, deg_a)
    error = deg_a - ref_a
    return _db_ratio(float(np.dot(ref_a, ref_a)), float(np.dot(error, error)))


def snr(deg: AudioLike, sr: int, *, ref: AudioLike, ref_sr: int | None = None) -> float:
    """Waveform signal-to-noise ratio in dB, noise := deg - ref."""
    deg_a, ref_a, _ = load_pair(deg, sr, ref, ref_sr)
    check_finite_nonzero(ref_a, deg_a)
    noise = deg_a - ref_a
    return _db_ratio(float(np.dot(ref_a, ref_a)), float(np.dot(noise, noise)))
