"""Log-domain spectral distances: LSD, MSD (mel-spectral distance), and mel-L1.

LSD follows the standard log-spectral-distortion definition used across speech
enhancement/vocoder literature: per-frame RMS of the dB difference between power
spectra, averaged over frames. MSD is the analogous quantity computed on the log-mel
spectrogram instead of the linear STFT magnitude. ``mel_l1`` is the mean absolute
difference between log-mel spectrograms, the vocoder-training metric most neural
TTS/vocoder papers report.
"""
from __future__ import annotations

import numpy as np

from speechonnxmetrics._dsp.mel import log_melspectrogram
from speechonnxmetrics._dsp.stft import stft
from speechonnxmetrics.intrusive._common import AudioLike, IntrusiveMetricError, check_finite_nonzero, load_pair

_CLIP = 1e-8


def _require_one_frame(ref_a: np.ndarray, deg_a: np.ndarray, n_fft: int) -> None:
    if deg_a.size < n_fft or ref_a.size < n_fft:
        raise IntrusiveMetricError(f"signal shorter than one {n_fft}-sample analysis frame")


def lsd(
    deg: AudioLike, sr: int, *, ref: AudioLike, ref_sr: int | None = None,
    n_fft: int = 1024, hop_size: int = 256,
) -> float:
    """Log-spectral distance in dB (``0`` for identical signals; higher is worse)."""
    deg_a, ref_a, common_sr = load_pair(deg, sr, ref, ref_sr)
    check_finite_nonzero(ref_a, deg_a)
    _require_one_frame(ref_a, deg_a, n_fft)
    ref_pow = np.abs(stft(ref_a, n_fft=n_fft, hop_size=hop_size, win_size=n_fft)) ** 2
    deg_pow = np.abs(stft(deg_a, n_fft=n_fft, hop_size=hop_size, win_size=n_fft)) ** 2
    n = min(ref_pow.shape[1], deg_pow.shape[1])
    log_ratio = 10.0 * np.log10(np.maximum(ref_pow[:, :n], _CLIP) / np.maximum(deg_pow[:, :n], _CLIP))
    return float(np.mean(np.sqrt(np.mean(log_ratio ** 2, axis=0))))


def msd(
    deg: AudioLike, sr: int, *, ref: AudioLike, ref_sr: int | None = None,
    n_mels: int = 80, n_fft: int = 1024, hop_size: int = 256,
) -> float:
    """Mel-spectral distance in dB: :func:`lsd`'s definition on the log-mel
    spectrogram instead of the linear STFT magnitude."""
    deg_a, ref_a, common_sr = load_pair(deg, sr, ref, ref_sr)
    check_finite_nonzero(ref_a, deg_a)
    _require_one_frame(ref_a, deg_a, n_fft)
    ref_log_mel = log_melspectrogram(ref_a, common_sr, n_fft=n_fft, hop_size=hop_size, n_mels=n_mels)
    deg_log_mel = log_melspectrogram(deg_a, common_sr, n_fft=n_fft, hop_size=hop_size, n_mels=n_mels)
    n = min(ref_log_mel.shape[1], deg_log_mel.shape[1])
    # log_melspectrogram is natural-log; convert the per-bin difference to dB.
    diff_db = (ref_log_mel[:, :n] - deg_log_mel[:, :n]).astype(np.float64) * (10.0 / np.log(10.0))
    return float(np.mean(np.sqrt(np.mean(diff_db ** 2, axis=0))))


def mel_l1(
    deg: AudioLike, sr: int, *, ref: AudioLike, ref_sr: int | None = None,
    n_mels: int = 80, n_fft: int = 1024, hop_size: int = 256,
) -> float:
    """Mean absolute difference between natural-log-mel spectrograms (``0`` for
    identical signals)."""
    deg_a, ref_a, common_sr = load_pair(deg, sr, ref, ref_sr)
    check_finite_nonzero(ref_a, deg_a)
    _require_one_frame(ref_a, deg_a, n_fft)
    ref_log_mel = log_melspectrogram(ref_a, common_sr, n_fft=n_fft, hop_size=hop_size, n_mels=n_mels)
    deg_log_mel = log_melspectrogram(deg_a, common_sr, n_fft=n_fft, hop_size=hop_size, n_mels=n_mels)
    n = min(ref_log_mel.shape[1], deg_log_mel.shape[1])
    return float(np.mean(np.abs(ref_log_mel[:, :n].astype(np.float64) - deg_log_mel[:, :n].astype(np.float64))))
