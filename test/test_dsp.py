"""Tests for the shared numpy DSP layer, oracled against scipy where one exists.

scipy is a TEST-ONLY dependency (the `test` extra) — never imported at runtime by
`speechonnxmetrics`. Every adversarial case documented in the DSP task spec is covered
explicitly rather than relying on incidental crashes.
"""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest
import scipy.signal

from speechonnxmetrics._dsp.audio import AudioLoadError, load_audio
from speechonnxmetrics._dsp.dtw import DTWError, dtw
from speechonnxmetrics._dsp.mel import log_melspectrogram, mel_filterbank, melspectrogram
from speechonnxmetrics._dsp.pitch import PitchError, yin
from speechonnxmetrics._dsp.resample import kaiser_resample
from speechonnxmetrics._dsp.stft import istft, stft

SR = 16000


def _sine(freq: float, duration: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _sawtooth(freq: float, duration: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return (amp * scipy.signal.sawtooth(2 * np.pi * freq * t)).astype(np.float64)


# --------------------------------------------------------------------------- STFT/ISTFT

class TestSTFT:
    def test_roundtrip_reconstruction_error_below_tolerance(self):
        # length is an exact multiple of hop_size so every sample is fully covered by
        # the frame grid — with center=True and a non-aligned length, the tail beyond
        # the last full frame is legitimately unreconstructed (same as torch.stft).
        x = _sine(440, 1.024)[:16384].astype(np.float32)
        spec = stft(x, n_fft=1024, hop_size=256, win_size=1024)
        rec = istft(spec, n_fft=1024, hop_size=256, win_size=1024, length=len(x))
        assert np.max(np.abs(rec - x)) < 1e-3

    def test_shape_agrees_with_scipy(self):
        x = _sine(440, 0.5)
        spec = stft(x, n_fft=512, hop_size=128, win_size=512, center=False)
        _, _, sspec = scipy.signal.stft(
            x, nperseg=512, noverlap=512 - 128, nfft=512, boundary=None, padded=False,
        )
        # scipy uses a different normalization convention (energy-preserving vs raw FFT
        # sum); shapes must match exactly, values only up to a real scale factor.
        assert spec.shape == sspec.shape
        ratio = np.abs(spec[1:, 1:-1]) / np.maximum(np.abs(sspec[1:, 1:-1]), 1e-12)
        assert np.std(ratio) / np.mean(ratio) < 0.05

    def test_signal_shorter_than_one_frame_raises_or_empty(self):
        x = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError):
            stft(x, n_fft=1024, hop_size=256, win_size=1024, center=False)

    def test_n_fft_larger_than_signal_with_center_padding_still_works(self):
        x = _sine(440, 0.01).astype(np.float32)  # 160 samples
        spec = stft(x, n_fft=1024, hop_size=256, win_size=1024, center=True)
        assert spec.shape[0] == 1024 // 2 + 1
        assert np.all(np.isfinite(spec))

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError):
            stft(np.zeros(0, dtype=np.float32), n_fft=64, hop_size=16, win_size=64, center=False)


# --------------------------------------------------------------------------------- mel

class TestMel:
    def test_filterbank_rows_sum_sensibly_and_no_empty_filter(self):
        fb = mel_filterbank(SR, n_fft=1024, n_mels=80)
        assert fb.shape == (80, 513)
        assert np.all(fb.sum(axis=1) > 0), "a silently empty top filter is a classic bug"
        assert np.all(np.isfinite(fb))

    def test_htk_and_slaney_differ(self):
        slaney = mel_filterbank(SR, 1024, 80, norm="slaney")
        htk = mel_filterbank(SR, 1024, 80, norm="htk")
        assert not np.allclose(slaney, htk)

    def test_log_melspectrogram_finite_and_shaped(self):
        x = _sine(440, 0.5).astype(np.float32)
        mel = log_melspectrogram(x, SR, n_fft=1024, hop_size=256, n_mels=80)
        assert mel.shape[0] == 80
        assert np.all(np.isfinite(mel))

    def test_silence_does_not_produce_neg_inf(self):
        x = np.zeros(SR, dtype=np.float32)
        mel = log_melspectrogram(x, SR, n_fft=1024, hop_size=256, n_mels=40)
        assert np.all(np.isfinite(mel))

    def test_invalid_n_fft_raises(self):
        with pytest.raises(ValueError):
            mel_filterbank(SR, n_fft=0, n_mels=80)

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError):
            mel_filterbank(0, n_fft=1024, n_mels=80)


# ----------------------------------------------------------------------------- resample

class TestResample:
    @pytest.mark.parametrize("orig_sr,new_sr", [(44100, 16000), (8000, 48000), (16000, 8000), (22050, 22050)])
    def test_output_length_matches_expected(self, orig_sr, new_sr):
        x = _sine(220, 0.5, sr=orig_sr).astype(np.float32)
        out = kaiser_resample(x, orig_sr, new_sr)
        expected_len = int(np.ceil(new_sr * len(x) / orig_sr))
        assert out.shape[0] == expected_len

    def test_agrees_with_scipy_resample_poly_within_tolerance(self):
        x = _sine(220, 0.5, sr=44100).astype(np.float32)
        out = kaiser_resample(x, 44100, 16000)
        ref = scipy.signal.resample_poly(x, 16000, 44100)
        n = min(len(out), len(ref))
        # different filter designs (kaiser-sinc vs scipy's default) — compare shape of
        # the resampled waveform via correlation, not sample-exact values.
        corr = np.corrcoef(out[:n], ref[:n])[0, 1]
        assert corr > 0.95

    def test_equal_rates_is_a_noop_passthrough(self):
        x = _sine(220, 0.1).astype(np.float32)
        out = kaiser_resample(x, 16000, 16000)
        assert out is x or np.array_equal(out, x)

    def test_empty_input_returns_empty(self):
        out = kaiser_resample(np.zeros(0, dtype=np.float32), 44100, 16000)
        assert out.shape[0] == 0


# --------------------------------------------------------------------------------- YIN

class TestYIN:
    @pytest.mark.parametrize("freq", [80.0, 150.0, 220.0, 400.0])
    def test_recovers_known_sine_f0_within_cents(self, freq):
        x = _sine(freq, 1.0).astype(np.float32)
        f0, voiced, _ = yin(x, SR, fmin=50, fmax=500)
        voiced_f0 = f0[voiced]
        assert voiced_f0.size > 0
        cents = 1200 * np.log2(np.median(voiced_f0) / freq)
        assert abs(cents) < 20  # within ~20 cents

    @pytest.mark.parametrize("freq", [80.0, 220.0, 400.0])
    def test_recovers_known_sawtooth_f0_within_cents(self, freq):
        x = _sawtooth(freq, 1.0).astype(np.float32)
        f0, voiced, _ = yin(x, SR, fmin=50, fmax=500)
        voiced_f0 = f0[voiced]
        assert voiced_f0.size > 0
        cents = 1200 * np.log2(np.median(voiced_f0) / freq)
        assert abs(cents) < 30

    def test_silence_is_unvoiced_not_nan(self):
        x = np.zeros(SR, dtype=np.float32)
        f0, voiced, _ = yin(x, SR)
        assert not np.any(voiced)
        assert np.all(np.isfinite(f0))
        assert np.all(f0 == 0.0)

    def test_all_zeros_short_signal_unvoiced(self):
        x = np.zeros(100, dtype=np.float32)
        f0, voiced, times = yin(x, SR, frame_length=2048)
        assert f0.size == 0 and voiced.size == 0 and times.size == 0

    def test_nan_input_raises(self):
        x = _sine(220, 0.1)
        x[10] = np.nan
        with pytest.raises(PitchError):
            yin(x, SR)

    def test_inf_input_raises(self):
        x = _sine(220, 0.1)
        x[10] = np.inf
        with pytest.raises(PitchError):
            yin(x, SR)

    def test_zero_sample_rate_raises(self):
        with pytest.raises(PitchError):
            yin(_sine(220, 0.1), 0)

    def test_negative_sample_rate_raises(self):
        with pytest.raises(PitchError):
            yin(_sine(220, 0.1), -16000)

    def test_empty_signal_returns_empty_arrays(self):
        f0, voiced, times = yin(np.zeros(0, dtype=np.float32), SR)
        assert f0.size == 0 and voiced.size == 0 and times.size == 0


# --------------------------------------------------------------------------------- DTW

class TestDTW:
    def test_identical_sequences_diagonal_path_zero_cost(self):
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        path, cost = dtw(x, x)
        assert cost == pytest.approx(0.0)
        assert np.array_equal(path, np.array([[0, 0], [1, 1], [2, 2], [3, 3]]))

    def test_hand_computed_small_case(self):
        # x = [1, 3], y = [1, 2, 4]; local dist |a-b|
        # d(0,0)=0 d(0,1)=1 d(0,2)=3
        # d(1,0)=2 d(1,1)=1 d(1,2)=1
        # cost[0,0]=0 cost[0,1]=1 cost[0,2]=4
        # cost[1,0]=2 cost[1,1]=min(1,2,0)+1=1 cost[1,2]=min(4,1,1)+1=2
        x = np.array([1.0, 3.0])
        y = np.array([1.0, 2.0, 4.0])
        path, cost = dtw(x, y)
        assert cost == pytest.approx(2.0)
        # two paths tie for the minimum cost here ((0,0)->(0,1)->(1,2) and
        # (0,0)->(1,1)->(1,2)); either is a correct optimal alignment.
        assert path.tolist() in ([[0, 0], [0, 1], [1, 2]], [[0, 0], [1, 1], [1, 2]])

    def test_empty_sequence_raises(self):
        with pytest.raises(DTWError):
            dtw(np.zeros(0), np.array([1.0, 2.0]))

    def test_mismatched_feature_dims_raise(self):
        with pytest.raises(DTWError):
            dtw(np.zeros((3, 2)), np.zeros((3, 4)))

    def test_nan_input_raises(self):
        x = np.array([1.0, np.nan])
        with pytest.raises(DTWError):
            dtw(x, np.array([1.0, 2.0]))

    def test_custom_distance_function(self):
        x = np.array([0.0, 0.0])
        y = np.array([1.0, 1.0])
        path, cost = dtw(x, y, distance=lambda a, b: np.abs(a[:, None, 0] - b[None, :, 0]) ** 2)
        assert cost == pytest.approx(2.0)


# ------------------------------------------------------------------------------- audio

class _MemoryWav:
    """A tiny in-memory WAV blob, for testing without a checked-in fixture file."""

    def __init__(self, samples: np.ndarray, sr: int, n_channels: int = 1):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(samples.tobytes())
        self.bytes = buf.getvalue()


class TestAudio:
    def test_int16_int32_float64_normalize_identically(self):
        base = _sine(440, 0.1, amp=0.5)
        i16 = (base * 32767).astype(np.int16)
        i32 = (base * 2147483647).astype(np.int32)
        f64 = base.astype(np.float64)
        x16, sr16 = load_audio(i16, sample_rate=SR)
        x32, sr32 = load_audio(i32, sample_rate=SR)
        xf64, srf64 = load_audio(f64, sample_rate=SR)
        assert sr16 == sr32 == srf64 == SR
        assert np.max(np.abs(x16 - xf64.astype(np.float32))) < 2e-3
        assert np.max(np.abs(x32 - xf64.astype(np.float32))) < 1e-4

    def test_stereo_downmix(self):
        left = _sine(440, 0.1)
        right = _sine(440, 0.1, amp=0.25)
        stereo = np.stack([left, right], axis=1).astype(np.float32)
        x, sr = load_audio(stereo, sample_rate=SR)
        assert x.ndim == 1
        assert np.allclose(x, (left + right) / 2, atol=1e-5)

    def test_real_wav_file_roundtrip(self, tmp_path):
        samples = (_sine(440, 0.2) * 32767).astype(np.int16)
        wav = _MemoryWav(samples, SR)
        p = tmp_path / "tone.wav"
        p.write_bytes(wav.bytes)
        x, sr = load_audio(p)
        assert sr == SR
        assert x.dtype == np.float32
        assert np.max(np.abs(x - samples.astype(np.float32) / 32768.0)) < 1e-4

    def test_wav_bytes_roundtrip(self):
        samples = (_sine(440, 0.2) * 32767).astype(np.int16)
        wav = _MemoryWav(samples, SR)
        x, sr = load_audio(wav.bytes)
        assert sr == SR
        assert x.shape[0] == samples.shape[0]

    def test_resample_on_load(self):
        samples = (_sine(440, 0.2) * 32767).astype(np.int16)
        wav = _MemoryWav(samples, SR)
        x, sr = load_audio(wav.bytes, target_sr=8000)
        assert sr == 8000
        assert x.shape[0] == pytest.approx(samples.shape[0] * 8000 / SR, abs=1)

    def test_missing_file_raises(self):
        with pytest.raises(AudioLoadError):
            load_audio("/nonexistent/path/does-not-exist.wav")

    def test_numpy_source_without_sample_rate_raises(self):
        with pytest.raises(AudioLoadError):
            load_audio(np.zeros(100, dtype=np.float32))

    def test_nan_audio_raises(self):
        x = _sine(440, 0.1).astype(np.float32)
        x[5] = np.nan
        with pytest.raises(AudioLoadError):
            load_audio(x, sample_rate=SR)

    def test_zero_sample_rate_raises(self):
        with pytest.raises(AudioLoadError):
            load_audio(np.zeros(100, dtype=np.float32), sample_rate=0)

    def test_unsupported_source_type_raises(self):
        with pytest.raises(AudioLoadError):
            load_audio(12345)
