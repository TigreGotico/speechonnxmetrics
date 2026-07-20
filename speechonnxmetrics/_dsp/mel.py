"""Mel filterbank construction and log-mel spectrograms, built on the shared STFT.

Independent of the Kaldi-specific filterbank in :mod:`speechonnxmetrics._dsp.fbank`
(which reproduces ``torchaudio.compliance.kaldi.fbank`` step for step). This module
builds a generic mel filterbank — Slaney (librosa/HTK-free) or HTK convention — on top
of the single shared :func:`speechonnxmetrics._dsp.stft.stft`, for metrics that expect a
librosa-style log-mel spectrogram rather than a Kaldi one.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from speechonnxmetrics._dsp.stft import PadMode, stft

MelNorm = Literal["slaney", "htk"]


def _hz_to_mel(freq: np.ndarray, htk: bool) -> np.ndarray:
    if htk:
        return 2595.0 * np.log10(1.0 + freq / 700.0)
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = freq >= min_log_hz
    return np.where(
        log_region, min_log_mel + np.log(np.maximum(freq, min_log_hz) / min_log_hz) / logstep, mels,
    )


def _mel_to_hz(mel: np.ndarray, htk: bool) -> np.ndarray:
    if htk:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mel
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mel >= min_log_mel
    return np.where(log_region, min_log_hz * np.exp(logstep * (mel - min_log_mel)), freqs)


def mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int,
    fmin: float = 0.0, fmax: float | None = None, norm: MelNorm = "slaney",
) -> np.ndarray:
    """Triangular mel filterbank ``[n_mels, n_fft//2+1]`` (Slaney or HTK convention).

    Matches ``librosa.filters.mel(htk=False)`` for ``norm="slaney"`` and
    ``librosa.filters.mel(htk=True)`` for ``norm="htk"``.
    """
    if n_fft <= 0 or n_mels <= 0:
        raise ValueError(f"n_fft and n_mels must be positive, got n_fft={n_fft}, n_mels={n_mels}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    htk = norm == "htk"
    fmax = sample_rate / 2.0 if fmax is None else fmax

    n_freqs = n_fft // 2 + 1
    # rfft bin centres: k * sr / n_fft. Identical to linspace(0, sr/2, n_freqs) for even
    # n_fft, but not for odd n_fft, where the last bin sits below Nyquist.
    fft_freqs = np.arange(n_freqs) * (sample_rate / n_fft)

    mel_pts = np.linspace(_hz_to_mel(np.asarray(fmin), htk), _hz_to_mel(np.asarray(fmax), htk), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts, htk)

    fdiff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))

    if not htk:  # Slaney-style area normalization
        enorm = 2.0 / (hz_pts[2:n_mels + 2] - hz_pts[:n_mels])
        weights *= enorm[:, None]

    return weights.astype(np.float32)


def melspectrogram(
    audio: np.ndarray, sample_rate: int, n_fft: int = 1024, hop_size: int = 256,
    win_size: int | None = None, n_mels: int = 80,
    fmin: float = 0.0, fmax: float | None = None, norm: MelNorm = "slaney",
    power: float = 2.0, pad_mode: PadMode = "reflect",
) -> np.ndarray:
    """Mel spectrogram ``[n_mels, frames]`` from raw audio via the shared STFT."""
    win_size = n_fft if win_size is None else win_size
    spec = stft(audio, n_fft=n_fft, hop_size=hop_size, win_size=win_size, pad_mode=pad_mode)
    mag = np.abs(spec) ** power
    fb = mel_filterbank(sample_rate, n_fft, n_mels, fmin=fmin, fmax=fmax, norm=norm)
    return (fb @ mag).astype(np.float32)


def log_melspectrogram(
    audio: np.ndarray, sample_rate: int, n_fft: int = 1024, hop_size: int = 256,
    win_size: int | None = None, n_mels: int = 80,
    fmin: float = 0.0, fmax: float | None = None, norm: MelNorm = "slaney",
    power: float = 2.0, clip_val: float = 1e-5,
) -> np.ndarray:
    """Log-mel spectrogram ``[n_mels, frames]``, clamped before the log to avoid ``-inf``."""
    mel = melspectrogram(
        audio, sample_rate, n_fft=n_fft, hop_size=hop_size, win_size=win_size,
        n_mels=n_mels, fmin=fmin, fmax=fmax, norm=norm, power=power,
    )
    return np.log(np.maximum(mel, clip_val)).astype(np.float32)
