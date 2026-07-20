"""Shared numpy DSP primitives (STFT, Kaldi filterbank, kaiser resampling).

Kept out of onnxruntime graphs so every front-end/back-end transform is inspectable
and torch-free at runtime.
"""
from __future__ import annotations

from speechmetrics._dsp.fbank import deltas, fbank, hamming, mel_banks
from speechmetrics._dsp.resample import kaiser_resample
from speechmetrics._dsp.stft import hamming_window, istft, stft, vorbis_window

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
]
