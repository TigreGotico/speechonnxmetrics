"""Pure-numpy YIN pitch tracker (de Cheveigne & Kawahara, 2002).

Needed downstream for log-F0 RMSE and voiced/unvoiced-decision error metrics, which
must not depend on a librosa/torch pitch tracker. The per-frame difference function is
computed for every frame at once via a batched FFT autocorrelation (no per-sample
python loop); only the scalar threshold search — inherently sequential in tau per
frame — loops over frames.
"""
from __future__ import annotations

import numpy as np


class PitchError(ValueError):
    """Raised for inputs YIN cannot meaningfully track pitch on."""


def _frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    n_frames = 1 + (x.size - frame_length) // hop_length
    return np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, frame_length),
        strides=(x.strides[0] * hop_length, x.strides[0]),
    )


def _difference_function(frames: np.ndarray, tau_max: int) -> np.ndarray:
    n_frames, frame_length = frames.shape
    fft_size = 1
    while fft_size < 2 * frame_length:
        fft_size *= 2

    spec = np.fft.rfft(frames, fft_size, axis=1)
    acf = np.fft.irfft(spec * np.conj(spec), fft_size, axis=1)[:, :tau_max + 1]

    cumsum = np.concatenate(
        [np.zeros((n_frames, 1)), np.cumsum(frames ** 2, axis=1)], axis=1,
    )
    total = cumsum[:, -1:]
    tau_range = np.arange(tau_max + 1)
    energy_first = cumsum[:, frame_length - tau_range]
    energy_second = total - cumsum[:, tau_range]
    diff = energy_first + energy_second - 2.0 * acf
    diff[:, 0] = 0.0
    return np.maximum(diff, 0.0)


def _cumulative_mean_normalized_difference(diff: np.ndarray) -> np.ndarray:
    cmnd = np.ones_like(diff)
    running = np.cumsum(diff[:, 1:], axis=1)
    tau = np.arange(1, diff.shape[1])
    cmnd[:, 1:] = diff[:, 1:] * tau / np.maximum(running, 1e-12)
    return cmnd


def _pick_tau(cmnd_frame: np.ndarray, tau_min: int, tau_max: int, threshold: float) -> int | None:
    for tau in range(tau_min, tau_max):
        if cmnd_frame[tau] < threshold:
            while tau + 1 < tau_max and cmnd_frame[tau + 1] < cmnd_frame[tau]:
                tau += 1
            return tau
    best = tau_min + int(np.argmin(cmnd_frame[tau_min:tau_max]))
    return best if cmnd_frame[best] < 1.0 else None


def _parabolic_refine(cmnd_frame: np.ndarray, tau: int) -> float:
    if tau <= 0 or tau >= cmnd_frame.size - 1:
        return float(tau)
    s0, s1, s2 = cmnd_frame[tau - 1], cmnd_frame[tau], cmnd_frame[tau + 1]
    denom = 2.0 * s1 - s0 - s2
    if abs(denom) < 1e-12:
        return float(tau)
    shift = 0.5 * (s0 - s2) / denom
    return float(tau) + float(np.clip(shift, -1.0, 1.0))


def yin(
    audio: np.ndarray, sample_rate: int, fmin: float = 65.0, fmax: float = 2093.0,
    frame_length: int = 2048, hop_length: int = 256,
    threshold: float = 0.1, energy_threshold: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YIN pitch tracking: returns ``(f0_hz, voiced_flag, times)``, one value per frame.

    Unvoiced/silent frames get ``f0_hz == 0.0`` and ``voiced_flag == False`` — never
    ``NaN``. Raises :class:`PitchError` on non-finite input or a non-positive sample rate.
    """
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if sample_rate <= 0:
        raise PitchError(f"sample_rate must be positive, got {sample_rate}")
    if not np.all(np.isfinite(x)):
        raise PitchError("audio contains NaN/inf")
    if x.size < frame_length:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32)

    tau_min = max(1, int(sample_rate / fmax))
    tau_max = min(frame_length // 2, int(sample_rate / fmin))
    if tau_min >= tau_max:
        raise PitchError(f"fmin/fmax range [{fmin}, {fmax}] is empty at sample_rate={sample_rate}")

    frames = _frame_signal(x, frame_length, hop_length)
    n_frames = frames.shape[0]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    diff = _difference_function(frames, tau_max)
    cmnd = _cumulative_mean_normalized_difference(diff)

    f0 = np.zeros(n_frames, dtype=np.float32)
    voiced = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        if rms[i] < energy_threshold:
            continue
        tau = _pick_tau(cmnd[i], tau_min, tau_max, threshold)
        if tau is None:
            continue
        refined = _parabolic_refine(cmnd[i], tau)
        if refined <= 0:
            continue
        f0[i] = sample_rate / refined
        voiced[i] = True

    times = (np.arange(n_frames, dtype=np.float32) * hop_length + frame_length / 2) / sample_rate
    return f0, voiced, times.astype(np.float32)
