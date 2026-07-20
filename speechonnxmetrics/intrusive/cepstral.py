"""Mel-cepstral distortion and F0-based intrusive pitch metrics.

MCD: Kubichek, R. (1993). "Mel-cepstral distance measure for objective speech
quality assessment." IEEE Pacific Rim Conference on Communications, Computers and
Signal Processing. The defining constant ``10/ln(10) * sqrt(2)`` converts a Euclidean
distance in the mel-cepstral domain into a base-10 log-power distortion in dB.

Mel-cepstral coefficients are derived from the shared log-mel spectrogram
(:func:`speechonnxmetrics._dsp.mel.log_melspectrogram`) via an orthonormal DCT-II
along the mel-frequency axis — the standard "DCT of the log-mel spectrum" cepstral
recipe used in MFCC extraction. The DCT-II is a few lines and specific to this
cepstral computation, not a general DSP primitive, so it stays local to this module
rather than joining ``_dsp``.

``log_f0_rmse`` and ``vuv_error`` are built on the shared YIN tracker
(:func:`speechonnxmetrics._dsp.pitch.yin`).
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from speechonnxmetrics._dsp.dtw import dtw
from speechonnxmetrics._dsp.mel import log_melspectrogram
from speechonnxmetrics._dsp.pitch import yin
from speechonnxmetrics.intrusive._common import AudioLike, IntrusiveMetricError, check_finite_nonzero, load_pair

_MCD_CONST = 10.0 / np.log(10.0) * np.sqrt(2.0)
AlignMode = Literal["dtw", "frame"]


def _dct2_ortho(x: np.ndarray) -> np.ndarray:
    """Orthonormal DCT-II along axis 0 of ``x`` (``[n_mels, frames]``); matches
    ``scipy.fft.dct(x, type=2, norm="ortho", axis=0)``."""
    n = x.shape[0]
    k = np.arange(n)[:, None]
    m = np.arange(n)[None, :]
    basis = np.cos(np.pi / n * (m + 0.5) * k)
    basis[0, :] *= 1.0 / np.sqrt(2.0)
    basis *= np.sqrt(2.0 / n)
    return basis @ x


def _mel_cepstra(audio: np.ndarray, sr: int, n_mels: int, n_mfcc: int, n_fft: int, hop_size: int) -> np.ndarray:
    log_mel = log_melspectrogram(audio, sr, n_fft=n_fft, hop_size=hop_size, n_mels=n_mels)
    return _dct2_ortho(log_mel.astype(np.float64))[:n_mfcc]  # [n_mfcc, frames]


def mcd(
    deg: AudioLike, sr: int, *, ref: AudioLike, ref_sr: int | None = None,
    n_mels: int = 80, n_mfcc: int = 25, n_fft: int = 1024, hop_size: int = 256,
    include_c0: bool = False, align: AlignMode = "dtw",
) -> float:
    """Mel-cepstral distortion in dB (lower is better; ``0`` for identical signals).

    Cepstral trajectories are aligned with dynamic time warping
    (:func:`speechonnxmetrics._dsp.dtw.dtw`, ``align="dtw"``, the default) before the
    distance is computed, or truncated to a common frame count and compared
    frame-by-frame (``align="frame"``) when the two signals are already time-aligned.
    ``c0`` (log energy) is excluded by default, per the metric's standard usage for
    comparing spectral *shape* independent of loudness; set ``include_c0=True`` to
    include it.
    """
    deg_a, ref_a, common_sr = load_pair(deg, sr, ref, ref_sr)
    check_finite_nonzero(ref_a, deg_a)
    if deg_a.size < n_fft or ref_a.size < n_fft:
        raise IntrusiveMetricError(f"signal shorter than one {n_fft}-sample analysis frame")

    ref_mc = _mel_cepstra(ref_a, common_sr, n_mels, n_mfcc, n_fft, hop_size)
    deg_mc = _mel_cepstra(deg_a, common_sr, n_mels, n_mfcc, n_fft, hop_size)
    lo = 0 if include_c0 else 1

    if align == "dtw":
        path, _ = dtw(ref_mc[lo:].T, deg_mc[lo:].T)
        diff = ref_mc[lo:, path[:, 0]] - deg_mc[lo:, path[:, 1]]
    elif align == "frame":
        n = min(ref_mc.shape[1], deg_mc.shape[1])
        diff = ref_mc[lo:, :n] - deg_mc[lo:, :n]
    else:
        raise ValueError(f"align must be 'dtw' or 'frame', got {align!r}")

    per_frame = _MCD_CONST * np.sqrt(np.sum(diff ** 2, axis=0))
    return float(np.mean(per_frame))


def log_f0_rmse(
    deg: AudioLike, sr: int, *, ref: AudioLike, ref_sr: int | None = None,
    fmin: float = 65.0, fmax: float = 2093.0,
) -> float:
    """RMSE of ``log(f0)`` (natural log, Hz) over frames voiced in *both* signals.

    Frames are compared by index after truncating to the shorter track — YIN's fixed
    hop size gives both signals the same frame rate, so no further alignment is
    needed. Raises :class:`IntrusiveMetricError` if no frame is voiced in both.
    """
    deg_a, ref_a, common_sr = load_pair(deg, sr, ref, ref_sr)
    check_finite_nonzero(ref_a, deg_a)
    ref_f0, ref_voiced, _ = yin(ref_a, common_sr, fmin=fmin, fmax=fmax)
    deg_f0, deg_voiced, _ = yin(deg_a, common_sr, fmin=fmin, fmax=fmax)
    n = min(ref_f0.size, deg_f0.size)
    if n == 0:
        raise IntrusiveMetricError("signal too short for one pitch-analysis frame")
    both_voiced = ref_voiced[:n] & deg_voiced[:n]
    if not both_voiced.any():
        raise IntrusiveMetricError("no frame is voiced in both reference and degraded audio")
    diff = np.log(ref_f0[:n][both_voiced].astype(np.float64)) - np.log(deg_f0[:n][both_voiced].astype(np.float64))
    return float(np.sqrt(np.mean(diff ** 2)))


def vuv_error(
    deg: AudioLike, sr: int, *, ref: AudioLike, ref_sr: int | None = None,
    fmin: float = 65.0, fmax: float = 2093.0,
) -> float:
    """Fraction of frames where the reference and degraded voiced/unvoiced decisions
    disagree, in ``[0, 1]``."""
    deg_a, ref_a, common_sr = load_pair(deg, sr, ref, ref_sr)
    check_finite_nonzero(ref_a, deg_a)
    _, ref_voiced, _ = yin(ref_a, common_sr, fmin=fmin, fmax=fmax)
    _, deg_voiced, _ = yin(deg_a, common_sr, fmin=fmin, fmax=fmax)
    n = min(ref_voiced.size, deg_voiced.size)
    if n == 0:
        raise IntrusiveMetricError("signal too short for one pitch-analysis frame")
    return float(np.mean(ref_voiced[:n] != deg_voiced[:n]))
