"""Audio loading/coercion: path, numpy array, or bytes -> float32 mono in [-1, 1].

Prefers ``soundfile`` when installed (optional ``audio`` extra) for broad codec
support, and falls back to the stdlib ``wave`` module for plain PCM WAV so the base
install stays dependency-light. Resampling, when a target rate is given, routes
through the single shared :func:`speechonnxmetrics._dsp.resample.kaiser_resample` —
never a second resampler.
"""
from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from speechonnxmetrics._dsp.resample import kaiser_resample


class AudioLoadError(ValueError):
    """Raised when audio cannot be loaded or coerced into float32 mono."""


def _normalize_dtype(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.float32:
        return x
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0)
    if x.dtype == np.int32:
        return (x.astype(np.float32) / 2147483648.0)
    if x.dtype == np.float64:
        return x.astype(np.float32)
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float32)
    raise AudioLoadError(f"unsupported audio dtype: {x.dtype}")


def _downmix(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1).astype(np.float32) if x.ndim == 2 else x


def _read_wave_bytes(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sampwidth)
    if dtype is None:
        raise AudioLoadError(f"unsupported WAV sample width: {sampwidth} bytes")
    samples = np.frombuffer(raw, dtype=dtype)
    if dtype == np.uint8:  # 8-bit PCM is unsigned, centered at 128
        samples = (samples.astype(np.float32) - 128.0) / 128.0
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)
    return samples, sr


def _load_bytes(data: bytes) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return x, sr
    except ImportError:
        return _read_wave_bytes(data)


def _load_path(path: str | Path) -> tuple[np.ndarray, int]:
    p = Path(path)
    if not p.is_file():
        raise AudioLoadError(f"audio file not found: {p}")
    try:
        import soundfile as sf
        x, sr = sf.read(str(p), dtype="float32", always_2d=False)
        return x, sr
    except ImportError:
        return _read_wave_bytes(p.read_bytes())


def load_audio(
    source: str | Path | bytes | np.ndarray, sample_rate: int | None = None,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """Load/coerce ``source`` into float32 mono ``[-1, 1]``; returns ``(audio, sample_rate)``.

    ``source`` may be a filesystem path, raw file bytes (e.g. a WAV blob), or an
    already-decoded numpy array — in which case ``sample_rate`` is required. Resamples
    to ``target_sr`` when given, via the shared kaiser-windowed sinc resampler.
    """
    if isinstance(source, np.ndarray):
        if sample_rate is None:
            raise AudioLoadError("sample_rate is required when source is a numpy array")
        x, sr = source, sample_rate
    elif isinstance(source, (bytes, bytearray)):
        x, sr = _load_bytes(bytes(source))
    elif isinstance(source, (str, Path)):
        x, sr = _load_path(source)
    else:
        raise AudioLoadError(f"unsupported audio source type: {type(source)}")

    if sr <= 0:
        raise AudioLoadError(f"sample_rate must be positive, got {sr}")
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        raise AudioLoadError("audio contains NaN/inf")

    x = _normalize_dtype(x)
    x = _downmix(x)

    if target_sr is not None and target_sr != sr:
        x = kaiser_resample(x, sr, target_sr)
        sr = target_sr
    return x.astype(np.float32, copy=False), sr
