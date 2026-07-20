"""Short-Time Objective Intelligibility (STOI) and Extended STOI (ESTOI).

Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2010). "A Short-Time
Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech."
ICASSP. Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2011). "An
Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech."
IEEE TASLP. ESTOI: Jensen, J., & Taal, C. H. (2016). "An Algorithm for Predicting the
Intelligibility of Speech Masked by Modulated Noise Maskers." IEEE/ACM TASLP.

Both resample to the standard 10 kHz analysis rate, discard silent frames (energy
more than 40 dB below the reference's peak frame), decompose into 15 one-third-octave
bands over 384 ms (30-frame) segments via the shared STFT
(:func:`speechonnxmetrics._dsp.stft.stft`), and correlate normalized short-time band
envelopes. STOI clips and normalizes each band independently and averages the
per-band, per-segment correlations. ESTOI instead row/column-normalizes whole
384 ms x 15-band segments and correlates them as flattened vectors, which makes it
more sensitive to across-band envelope structure (e.g. noise that fills spectral gaps).
"""
from __future__ import annotations

import numpy as np

from speechonnxmetrics._dsp.stft import stft
from speechonnxmetrics.intrusive._common import AudioLike, IntrusiveMetricError, check_finite_nonzero, load_pair

_FS = 10000
_N_FRAME = 256  # analysis window, samples at 10 kHz
_NFFT = 512
_HOP = _N_FRAME // 2  # 128 samples == 12.8 ms
_PAD_LEFT = (_NFFT - _N_FRAME) // 2
_NUMBAND = 15
_MINFREQ = 150.0
_N_SEG = 30  # 30 frames * 12.8 ms hop == 384 ms
_BETA = -15.0  # dB lower bound used to clip the degraded envelope
_DYN_RANGE = 40.0  # dB VAD threshold below the reference's peak-energy frame
_EPS = np.finfo(float).eps


def _thirdoct(fs: int, nfft: int, num_bands: int, min_freq: float) -> np.ndarray:
    """1/3-octave band matrix ``[num_bands, nfft//2+1]`` selecting FFT bins per band."""
    f = np.linspace(0.0, fs, nfft + 1)[: nfft // 2 + 1]
    k = np.arange(num_bands, dtype=np.float64)
    freq_low = min_freq * 2.0 ** ((2.0 * k - 1.0) / 6.0)
    freq_high = min_freq * 2.0 ** ((2.0 * k + 1.0) / 6.0)
    obm = np.zeros((num_bands, f.size))
    for i in range(num_bands):
        fl = int(np.argmin(np.abs(f - freq_low[i])))
        fh = int(np.argmin(np.abs(f - freq_high[i])))
        obm[i, fl:fh] = 1.0
    return obm


def _analysis_window() -> np.ndarray:
    """The MATLAB/Octave-compatible Hann window STOI is defined with — ``hanning(N+2)``
    with its (zero) endpoints trimmed — zero-padded to ``_NFFT`` at the same offset the
    shared STFT centers an explicit window within ``n_fft``."""
    n = np.arange(1, _N_FRAME + 1)
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (_N_FRAME + 1))
    out = np.zeros(_NFFT)
    out[_PAD_LEFT:_PAD_LEFT + _N_FRAME] = win
    return out


def _frame(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    starts = np.arange(0, max(x.size - frame_len, 0), hop)
    if starts.size == 0:
        return np.zeros((0, frame_len))
    idx = starts[:, None] + np.arange(frame_len)[None, :]
    return x[idx]


def _overlap_add(frames: np.ndarray, hop: int) -> np.ndarray:
    n_frames, frame_len = frames.shape
    if n_frames == 0:
        return np.zeros(0)
    out = np.zeros((n_frames - 1) * hop + frame_len)
    for i in range(n_frames):
        out[i * hop: i * hop + frame_len] += frames[i]
    return out


def _remove_silent_frames(ref: np.ndarray, deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    win = np.hanning(_N_FRAME + 2)[1:-1]
    ref_frames = _frame(ref, _N_FRAME, _HOP) * win
    deg_frames = _frame(deg, _N_FRAME, _HOP) * win
    if ref_frames.shape[0] == 0:
        raise IntrusiveMetricError(f"signal shorter than one {_N_FRAME}-sample STOI analysis frame")
    energies = 20.0 * np.log10(np.linalg.norm(ref_frames, axis=1) + _EPS)
    mask = (energies.max() - _DYN_RANGE - energies) < 0
    if not mask.any():
        raise IntrusiveMetricError("no frames above the STOI VAD dynamic-range threshold")
    return _overlap_add(ref_frames[mask], _HOP), _overlap_add(deg_frames[mask], _HOP)


def _band_envelopes(x: np.ndarray, obm: np.ndarray, window: np.ndarray) -> np.ndarray:
    """1/3-octave band envelopes ``[bands, frames]``, frame-aligned with pystoi's
    ``x[i:i+N_FRAME]`` loop by front-padding ``x`` by ``_PAD_LEFT`` samples — the shared
    STFT centers its window inside the ``_NFFT`` block at that same offset, so the
    padding cancels the offset and frame ``k`` covers exactly ``x[k*hop : k*hop+N_FRAME]``.
    """
    padded = np.concatenate([np.zeros(_PAD_LEFT), x])
    spec = stft(padded, n_fft=_NFFT, hop_size=_HOP, win_size=_N_FRAME, center=False, window=window)
    mag2 = np.abs(spec).astype(np.float64) ** 2
    return np.sqrt(obm @ mag2 + _EPS)


def _segments(x: np.ndarray, n_seg: int) -> np.ndarray:
    """Sliding windows of ``n_seg`` consecutive frames: ``[n_segments, bands, n_seg]``."""
    n_bands, n_frames = x.shape
    n_segments = n_frames - n_seg + 1
    if n_segments < 1:
        raise IntrusiveMetricError(f"not enough frames ({n_frames}) for one {n_seg}-frame STOI segment")
    idx = np.arange(n_seg)[None, :] + np.arange(n_segments)[:, None]
    return x[:, idx].transpose(1, 0, 2)


def _band_matrices(deg: AudioLike, sr: int, ref: AudioLike, ref_sr: int | None) -> tuple[np.ndarray, np.ndarray]:
    deg_a, ref_a, _ = load_pair(deg, sr, ref, ref_sr, target_sr=_FS)
    check_finite_nonzero(ref_a, deg_a)
    ref_sil, deg_sil = _remove_silent_frames(ref_a, deg_a)
    if ref_sil.size < _N_FRAME or deg_sil.size < _N_FRAME:
        raise IntrusiveMetricError("not enough voiced signal remains after STOI's VAD for one analysis frame")
    obm = _thirdoct(_FS, _NFFT, _NUMBAND, _MINFREQ)
    window = _analysis_window()
    x_tob = _band_envelopes(ref_sil, obm, window)
    y_tob = _band_envelopes(deg_sil, obm, window)
    n_frames = min(x_tob.shape[1], y_tob.shape[1])
    return x_tob[:, :n_frames], y_tob[:, :n_frames]


def stoi(deg: AudioLike, sr: int, *, ref: AudioLike, ref_sr: int | None = None) -> float:
    """STOI score, roughly in ``[0, 1]`` (higher means more intelligible)."""
    x_tob, y_tob = _band_matrices(deg, sr, ref, ref_sr)
    x_seg, y_seg = _segments(x_tob, _N_SEG), _segments(y_tob, _N_SEG)  # [segs, bands, N_SEG]

    norm = np.linalg.norm(x_seg, axis=2, keepdims=True) / (np.linalg.norm(y_seg, axis=2, keepdims=True) + _EPS)
    y_norm = y_seg * norm
    clip_value = 10.0 ** (-_BETA / 20.0)
    y_prime = np.minimum(y_norm, x_seg * (1.0 + clip_value))

    y_prime = y_prime - y_prime.mean(axis=2, keepdims=True)
    x_c = x_seg - x_seg.mean(axis=2, keepdims=True)
    y_prime = y_prime / (np.linalg.norm(y_prime, axis=2, keepdims=True) + _EPS)
    x_c = x_c / (np.linalg.norm(x_c, axis=2, keepdims=True) + _EPS)

    return float(np.mean(np.sum(x_c * y_prime, axis=2)))


def estoi(deg: AudioLike, sr: int, *, ref: AudioLike, ref_sr: int | None = None) -> float:
    """Extended STOI score, roughly in ``[-1, 1]`` (higher means more intelligible);
    more sensitive than :func:`stoi` to envelope structure shared across bands."""
    x_tob, y_tob = _band_matrices(deg, sr, ref, ref_sr)
    x_seg, y_seg = _segments(x_tob, _N_SEG), _segments(y_tob, _N_SEG)  # [segs, bands, N_SEG]

    def _row_col_normalize(seg: np.ndarray) -> np.ndarray:
        row = seg - seg.mean(axis=2, keepdims=True)
        row = row / (np.linalg.norm(row, axis=2, keepdims=True) + _EPS)
        col = row - row.mean(axis=1, keepdims=True)
        return col / (np.linalg.norm(col, axis=1, keepdims=True) + _EPS)

    x_n, y_n = _row_col_normalize(x_seg), _row_col_normalize(y_seg)
    return float(np.sum(x_n * y_n) / (_N_SEG * x_n.shape[0]))
