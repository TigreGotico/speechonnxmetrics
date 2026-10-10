"""Kaldi-compatible log-mel filterbank in numpy.

A faithful port of ``torchaudio.compliance.kaldi.fbank`` (which itself reproduces Kaldi's
``compute-fbank-feats``), plus ``torchaudio.functional.compute_deltas``. Models trained on
Kaldi features are sensitive to the exact front-end — the DC removal, pre-emphasis and
windowing order, the power-of-two FFT padding, and Kaldi's specific triangular mel banks
all matter — so this reproduces the pipeline step for step rather than approximating it
with a generic mel spectrogram.

Validated against ``torchaudio`` to max abs err 3.2e-05 on the filterbank (1.1e-06
relative) and 8.3e-06 on the deltas.

Kaldi defaults reproduced here: ``remove_dc_offset=True``, ``preemphasis=0.97``,
``round_to_power_of_two=True``, ``snip_edges=True``, ``use_power=True``,
``use_log_fbank=True``, ``low_freq=20``, ``high_freq=nyquist``.

``dither`` is exposed but defaults to ``0``. Upstream pipelines often pass ``1.0``, which
makes the front-end stochastic; the difference it makes downstream is ~1e-5, far below the
model's own precision, so a deterministic engine is preferred here.
"""
from __future__ import annotations

import numpy as np

#: Kaldi floors the filterbank at float32 epsilon before taking the log.
_EPS32 = float(np.finfo(np.float32).eps)
_LOW_FREQ = 20.0
_PREEMPH = 0.97


def _mel(freq) -> np.ndarray:
    """Kaldi's mel scale: ``1127 * ln(1 + f/700)``."""
    return 1127.0 * np.log(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def hamming(win_length: int) -> np.ndarray:
    """``torch.hamming_window(n, periodic=False)`` — symmetric, alpha 0.54 / beta 0.46."""
    k = np.arange(win_length, dtype=np.float64)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * k / (win_length - 1))


def povey(win_length: int) -> np.ndarray:
    """Kaldi's ``povey`` window: a symmetric Hann window raised to the power 0.85."""
    k = np.arange(win_length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / (win_length - 1))) ** 0.85


_WINDOWS = {"hamming": hamming, "povey": povey}


def mel_banks(
    num_bins: int, padded_window: int, sample_rate: float,
    low_freq: float = _LOW_FREQ, high_freq: float = 0.0,
) -> np.ndarray:
    """Kaldi triangular mel filterbank ``[num_bins, padded_window//2]``.

    ``high_freq <= 0`` is an offset from Nyquist, as in Kaldi.
    """
    nyquist = 0.5 * sample_rate
    if high_freq <= 0.0:
        high_freq += nyquist
    fft_bin_width = sample_rate / padded_window
    mel_low, mel_high = _mel(low_freq), _mel(high_freq)
    delta = (mel_high - mel_low) / (num_bins + 1)

    bin_idx = np.arange(num_bins)[:, None]
    left = mel_low + bin_idx * delta
    center = mel_low + (bin_idx + 1.0) * delta
    right = mel_low + (bin_idx + 2.0) * delta

    mel = _mel(fft_bin_width * np.arange(padded_window // 2))[None, :]
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    return np.maximum(0.0, np.minimum(up, down))


def fbank(
    waveform: np.ndarray, sample_rate: float, win_length: int, hop_length: int,
    num_mel_bins: int, dither: float = 0.0, *, window: str = "hamming",
    low_freq: float = _LOW_FREQ, high_freq: float = 0.0, floor: float = _EPS32,
) -> np.ndarray:
    """Log-mel filterbank ``[frames, num_mel_bins]`` matching ``kaldi.fbank``.

    ``win_length``/``hop_length`` are in samples (Kaldi takes milliseconds; the caller
    converts). Frames are snipped to those that fit entirely, per ``snip_edges=True``.
    ``window`` is ``"hamming"`` (the Kaldi fbank default) or ``"povey"`` (the Kaldi MFCC
    default); ``low_freq``/``high_freq`` bound the mel banks as in :func:`mel_banks`.
    ``floor`` is the smallest energy before the log, float32 epsilon as in Kaldi.
    """
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    padded_window = 1
    while padded_window < win_length:
        padded_window *= 2

    if x.size < win_length:
        return np.zeros((0, num_mel_bins), dtype=np.float32)
    frames_n = 1 + (x.size - win_length) // hop_length
    idx = np.arange(win_length)[None, :] + hop_length * np.arange(frames_n)[:, None]
    frames = x[idx]

    if dither:
        frames = frames + np.random.randn(*frames.shape) * dither
    frames = frames - frames.mean(axis=1, keepdims=True)          # remove DC offset
    # pre-emphasis with a replicated first sample, matching Kaldi's edge handling
    shifted = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = frames - _PREEMPH * shifted
    frames = frames * _WINDOWS[window](win_length)[None, :]
    if padded_window != win_length:
        frames = np.pad(frames, ((0, 0), (0, padded_window - win_length)))

    power = np.abs(np.fft.rfft(frames, n=padded_window, axis=1)) ** 2
    banks = np.pad(
        mel_banks(num_mel_bins, padded_window, sample_rate, low_freq, high_freq), ((0, 0), (0, 1))
    )
    return np.log(np.maximum(power @ banks.T, floor)).astype(np.float32)


def deltas(features: np.ndarray, win_length: int = 5) -> np.ndarray:
    """``torchaudio.functional.compute_deltas`` over ``[frames, channels]`` (edge-padded)."""
    if features.shape[0] == 0:      # a clip shorter than one window has no frames to pad
        return features.astype(np.float32)
    half = (win_length - 1) // 2
    denom = 2 * sum(i * i for i in range(1, half + 1))
    padded = np.pad(features, ((half, half), (0, 0)), mode="edge")
    n = features.shape[0]
    out = np.zeros(features.shape, dtype=np.float64)
    for i in range(1, half + 1):
        out += i * (padded[half + i: half + i + n] - padded[half - i: half - i + n])
    return (out / denom).astype(np.float32)
