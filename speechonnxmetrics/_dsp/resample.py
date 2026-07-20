"""Windowed-sinc (kaiser) resampling, a pure-numpy port of torchaudio's ``resample``.

Some metrics need a specific bandlimited-interpolation front-end (torchaudio's
``sinc_interp_kaiser`` with ``lowpass_filter_width=16``, ``rolloff=0.945``,
``beta≈14.77``). Reproducing that resampler exactly — rather than a generic
``resample_poly`` — keeps results faithful to the reference implementation. This is a
direct numpy transcription of torchaudio's kernel construction and strided convolution.
"""
from __future__ import annotations

import math

import numpy as np


def _sinc_resample_kernel(
    orig_freq: int, new_freq: int, gcd: int,
    lowpass_filter_width: int, rolloff: float, beta: float,
):
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)

    idx = np.arange(-width, width + orig_freq, dtype=np.float64)[None, None] / orig_freq
    t = np.arange(0, -new_freq, -1, dtype=np.float64)[:, None, None] / new_freq + idx
    t = t * base_freq
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)

    from numpy import i0
    window = i0(beta * np.sqrt(1 - (t / lowpass_filter_width) ** 2)) / i0(beta)

    t = t * math.pi
    scale = base_freq / orig_freq
    kernels = np.where(t == 0, 1.0, np.sin(t) / np.where(t == 0, 1.0, t))
    kernels = kernels * window * scale
    return kernels.astype(np.float32), width, orig_freq, new_freq


def kaiser_resample(
    x: np.ndarray, orig_sr: int, new_sr: int, *,
    lowpass_filter_width: int = 16, rolloff: float = 0.945,
    beta: float = 14.769656459379492,
) -> np.ndarray:
    """Resample mono float32 ``x`` from ``orig_sr`` to ``new_sr`` (torchaudio-faithful)."""
    if orig_sr == new_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    g = math.gcd(int(orig_sr), int(new_sr))
    kernel, width, orig_freq, new_freq = _sinc_resample_kernel(
        orig_sr, new_sr, g, lowpass_filter_width, rolloff, beta
    )
    length = x.shape[0]
    padded = np.pad(x.astype(np.float32), (width, width + orig_freq))
    K = kernel.shape[-1]
    # strided windows: [num_frames, K] with step = orig_freq
    n_frames = (padded.shape[0] - K) // orig_freq + 1
    windows = np.lib.stride_tricks.as_strided(
        padded, shape=(n_frames, K),
        strides=(padded.strides[0] * orig_freq, padded.strides[0]),
    )
    # [num_frames, new_freq] = windows @ kernel[c].T, then interleave phases row-major
    out = windows @ kernel[:, 0, :].T  # (n_frames, new_freq)
    resampled = out.reshape(-1)
    target_len = int(math.ceil(new_freq * length / orig_freq))
    return np.ascontiguousarray(resampled[:target_len], dtype=np.float32)
