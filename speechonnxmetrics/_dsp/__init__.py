"""Shared numpy DSP primitives (STFT, mel, Kaldi filterbank, pitch, DTW, resampling, audio I/O).

Kept out of onnxruntime graphs so every front-end/back-end transform is inspectable
and torch-free at runtime. There is exactly one STFT implementation in this package —
every metric and every ONNX frontend must route through it.
"""
from __future__ import annotations

from speechonnxmetrics._dsp.audio import AudioLoadError, load_audio
from speechonnxmetrics._dsp.dtw import DTWError, dtw
from speechonnxmetrics._dsp.fbank import deltas, fbank, hamming, mel_banks
from speechonnxmetrics._dsp.mel import log_melspectrogram, mel_filterbank, melspectrogram
from speechonnxmetrics._dsp.pitch import PitchError, yin
from speechonnxmetrics._dsp.resample import kaiser_resample
from speechonnxmetrics._dsp.stft import hamming_window, istft, stft, vorbis_window

__all__ = [
    "stft",
    "istft",
    "hamming_window",
    "vorbis_window",
    "fbank",
    "mel_banks",
    "hamming",
    "deltas",
    "kaiser_resample",
    "mel_filterbank",
    "melspectrogram",
    "log_melspectrogram",
    "yin",
    "PitchError",
    "dtw",
    "DTWError",
    "load_audio",
    "AudioLoadError",
]
