"""Tests for the intrusive (reference-based) metrics.

STOI/ESTOI parity is checked against a fixture of pystoi outputs committed at
``test/fixtures/stoi_fixture.json`` (regenerate with
``test/fixtures/generate_stoi_fixture.py``) — pystoi is a test-only oracle, never a
declared runtime dependency (see ``pyproject.toml``'s ``test`` extra).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from speechonnxmetrics.intrusive import (
    IntrusiveMetricError,
    estoi,
    lsd,
    log_f0_rmse,
    mcd,
    mel_l1,
    msd,
    sdr,
    si_sdr,
    snr,
    stoi,
    vuv_error,
)

SR = 16000
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "stoi_fixture.json").read_text())

# STOI parity tolerance: the shared _dsp.stft frames on a slightly different sample
# grid than pystoi's own ad hoc analysis loop (compensated for with a front-padding
# trick — see stoi.py's _band_envelopes docstring — but not exact at the boundary
# frames), so full float parity is not expected; correlation-based scores stay close.
_STOI_ATOL = 0.02
# ESTOI correlates whole 15-band x 30-frame segments, so it is more sensitive than
# STOI to the same small framing-offset residual — a looser tolerance is needed.
_ESTOI_ATOL = 0.1


def _speechlike(duration: float, sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    f0 = 110 + 30 * np.sin(2 * np.pi * (0.4 + 0.1 * rng.random()) * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    harmonics = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * (2.5 + rng.random()) * t)
    return (0.2 * harmonics * env).astype(np.float32)


def _noisy(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.size).astype(np.float32)
    noise *= np.sqrt(np.mean(x ** 2) / (10 ** (snr_db / 10))) / (np.std(noise) + 1e-9)
    return x + noise


def _sine(freq: float, duration: float, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * duration)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------- STOI

class TestSTOI:
    @pytest.mark.parametrize("case", FIXTURE["cases"])
    def test_matches_pystoi_fixture(self, case):
        x = _speechlike(2.0, FIXTURE["sr"], case["seed"])
        y = x if case["snr_db"] is None else _noisy(x, case["snr_db"], case["seed"] + 100)
        assert stoi(y, FIXTURE["sr"], ref=x) == pytest.approx(case["stoi"], abs=_STOI_ATOL)
        assert estoi(y, FIXTURE["sr"], ref=x) == pytest.approx(case["estoi"], abs=_ESTOI_ATOL)

    def test_identical_signals_give_one(self):
        x = _speechlike(2.0, SR, 0)
        assert stoi(x, SR, ref=x) == pytest.approx(1.0, abs=1e-6)

    def test_stereo_input_is_downmixed(self):
        x = _speechlike(2.0, SR, 0)
        y = _noisy(x, 10, 1)
        stereo_x = np.stack([x, x], axis=1)
        stereo_y = np.stack([y, y], axis=1)
        assert stoi(stereo_y, SR, ref=stereo_x) == pytest.approx(stoi(y, SR, ref=x), abs=1e-6)

    def test_all_zero_reference_raises(self):
        y = _speechlike(2.0, SR, 0)
        with pytest.raises(IntrusiveMetricError):
            stoi(y, SR, ref=np.zeros_like(y))

    def test_nan_in_reference_raises(self):
        x = _speechlike(2.0, SR, 0).copy()
        x[10] = np.nan
        y = _speechlike(2.0, SR, 0)
        with pytest.raises(IntrusiveMetricError):
            stoi(y, SR, ref=x)

    def test_signal_shorter_than_one_analysis_frame_raises(self):
        x = _sine(220, 0.005)  # far below one 256-sample @10kHz STOI frame
        with pytest.raises(IntrusiveMetricError):
            stoi(x, SR, ref=x)

    def test_sample_rate_mismatch_is_resampled_not_errored(self):
        x = _speechlike(2.0, SR, 0)
        y = _noisy(x, 10, 1)
        x_8k = x[::2].copy()
        assert stoi(y, SR, ref=x_8k, ref_sr=SR // 2) == pytest.approx(stoi(y, SR, ref=x), abs=0.05)

    def test_length_mismatch_within_tolerance_truncates_with_warning(self):
        x = _speechlike(2.0, SR, 0)
        y = np.concatenate([x, x[: int(0.2 * x.size)]])  # 1.2x -> above warn ratio, below raise ratio
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stoi(y, SR, ref=x)
        assert any("truncat" in str(w.message) for w in caught)

    def test_length_mismatch_beyond_2x_raises(self):
        x = _speechlike(0.5, SR, 0)
        y = _speechlike(2.0, SR, 0)
        with pytest.raises(IntrusiveMetricError):
            stoi(y, SR, ref=x)


# ------------------------------------------------------------------------------ SI-SDR

class TestSISDR:
    def test_scale_invariance(self):
        rng = np.random.default_rng(0)
        ref = rng.standard_normal(SR).astype(np.float32)
        deg = ref + 0.1 * rng.standard_normal(SR).astype(np.float32)
        base = si_sdr(deg, SR, ref=ref)
        for c in (0.01, 0.5, 2.0, 100.0, -3.0):
            assert si_sdr((c * deg).astype(np.float32), SR, ref=ref) == pytest.approx(base, abs=1e-6)

    def test_hand_computed_orthogonal_case(self):
        # ref and error are orthogonal unit-ish vectors -> target energy == deg energy
        # projected onto ref, exactly computable by hand.
        ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        deg = np.array([2.0, 1.0, 0.0, 0.0], dtype=np.float32)  # alpha=2, residual=[0,1,0,0]
        # target energy = alpha^2 * ||ref||^2 = 4, residual energy = 1 -> 10*log10(4) dB
        assert si_sdr(deg, SR, ref=ref) == pytest.approx(10 * np.log10(4.0), abs=1e-6)

    def test_perfect_estimate_is_capped_not_infinite(self):
        ref = np.random.default_rng(1).standard_normal(1000).astype(np.float32)
        value = si_sdr((5.0 * ref).astype(np.float32), SR, ref=ref)
        assert np.isfinite(value)
        assert value == pytest.approx(100.0)

    def test_identical_inputs(self):
        ref = np.random.default_rng(2).standard_normal(1000).astype(np.float32)
        assert np.isfinite(si_sdr(ref, SR, ref=ref))

    def test_all_zero_reference_raises(self):
        deg = np.ones(1000, dtype=np.float32)
        with pytest.raises(IntrusiveMetricError):
            si_sdr(deg, SR, ref=np.zeros(1000, dtype=np.float32))

    def test_all_zero_degraded_is_finite(self):
        ref = np.random.default_rng(3).standard_normal(1000).astype(np.float32)
        value = si_sdr(np.zeros(1000, dtype=np.float32), SR, ref=ref)
        assert np.isfinite(value)

    def test_nan_in_degraded_raises(self):
        ref = np.ones(1000, dtype=np.float32)
        deg = ref.copy()
        deg[5] = np.inf
        with pytest.raises(IntrusiveMetricError):
            si_sdr(deg, SR, ref=ref)

    def test_single_sample_inputs(self):
        assert np.isfinite(si_sdr(np.array([2.0], dtype=np.float32), SR, ref=np.array([1.0], dtype=np.float32)))

    def test_stereo_input(self):
        rng = np.random.default_rng(4)
        ref = rng.standard_normal(1000).astype(np.float32)
        deg = ref + 0.1 * rng.standard_normal(1000).astype(np.float32)
        stereo_ref = np.stack([ref, ref], axis=1)
        stereo_deg = np.stack([deg, deg], axis=1)
        assert si_sdr(stereo_deg, SR, ref=stereo_ref) == pytest.approx(si_sdr(deg, SR, ref=ref), abs=1e-5)

    def test_length_mismatch_beyond_2x_raises(self):
        ref = np.ones(1000, dtype=np.float32)
        deg = np.ones(100, dtype=np.float32)
        with pytest.raises(IntrusiveMetricError):
            si_sdr(deg, SR, ref=ref)


class TestSDRAndSNR:
    def test_sdr_hand_computed(self):
        ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        deg = np.array([1.0, 2.0, 4.0], dtype=np.float32)  # error energy = 1, signal energy = 14
        assert sdr(deg, SR, ref=ref) == pytest.approx(10 * np.log10(14.0), abs=1e-6)

    def test_snr_matches_sdr_single_reference(self):
        rng = np.random.default_rng(5)
        ref = rng.standard_normal(1000).astype(np.float32)
        deg = ref + 0.2 * rng.standard_normal(1000).astype(np.float32)
        assert snr(deg, SR, ref=ref) == pytest.approx(sdr(deg, SR, ref=ref), abs=1e-9)

    def test_identical_inputs_capped(self):
        ref = np.random.default_rng(6).standard_normal(1000).astype(np.float32)
        assert sdr(ref, SR, ref=ref) == pytest.approx(100.0)

    def test_all_zero_reference_raises(self):
        with pytest.raises(IntrusiveMetricError):
            sdr(np.ones(100, dtype=np.float32), SR, ref=np.zeros(100, dtype=np.float32))


# -------------------------------------------------------------------------------- MCD

class TestMCD:
    def test_identical_signals_give_zero(self):
        x = _speechlike(1.0, SR, 0)
        assert mcd(x, SR, ref=x) == pytest.approx(0.0, abs=1e-6)
        assert mcd(x, SR, ref=x, align="frame") == pytest.approx(0.0, abs=1e-6)

    def test_dtw_and_frame_aligned_agree_when_already_aligned(self):
        x = _speechlike(1.0, SR, 0)
        y = _noisy(x, 15, 1)
        assert mcd(y, SR, ref=x, align="dtw") == pytest.approx(mcd(y, SR, ref=x, align="frame"), rel=0.4)

    def test_known_reference_pair_value(self):
        # A one-semitone pitch shift is a small, stable spectral-envelope perturbation
        # relative to the gross spectral mismatch of additive white noise at 0 dB SNR —
        # a hand-verifiable ordering, not a literature-standard SPTK mcep magnitude
        # (this module's MCEP comes from a DCT of the log-mel-STFT spectrogram, not a
        # smoothed cepstral envelope, so its absolute scale is not directly comparable).
        x = _sine(220, 1.0)
        pitch_shifted = mcd(_sine(233.08, 1.0), SR, ref=x)
        noisy = mcd(_noisy(x, 0, 1), SR, ref=x)
        assert 0.0 < pitch_shifted < noisy

    def test_noisier_signal_gives_larger_mcd(self):
        x = _speechlike(1.0, SR, 0)
        mild = mcd(_noisy(x, 30, 1), SR, ref=x)
        harsh = mcd(_noisy(x, 0, 1), SR, ref=x)
        assert harsh > mild

    def test_invalid_align_raises(self):
        x = _speechlike(1.0, SR, 0)
        with pytest.raises(ValueError):
            mcd(x, SR, ref=x, align="bogus")

    def test_all_zero_reference_raises(self):
        x = _speechlike(1.0, SR, 0)
        with pytest.raises(IntrusiveMetricError):
            mcd(x, SR, ref=np.zeros_like(x))

    def test_signal_shorter_than_one_frame_raises(self):
        x = np.zeros(100, dtype=np.float32) + 1.0
        with pytest.raises(IntrusiveMetricError):
            mcd(x, SR, ref=x)

    def test_include_c0_changes_result(self):
        x = _speechlike(1.0, SR, 0)
        y = _noisy(x, 10, 1)
        assert mcd(y, SR, ref=x, include_c0=True) != pytest.approx(mcd(y, SR, ref=x, include_c0=False))


# ------------------------------------------------------------------- log_f0_rmse / vuv

class TestF0Metrics:
    def test_identical_tone_zero_rmse_zero_vuv_error(self):
        x = _sine(220, 1.0)
        assert log_f0_rmse(x, SR, ref=x) == pytest.approx(0.0, abs=1e-3)
        assert vuv_error(x, SR, ref=x) == pytest.approx(0.0, abs=1e-6)

    def test_known_octave_shift(self):
        x = _sine(220, 1.0)
        y = _sine(440, 1.0)
        assert log_f0_rmse(y, SR, ref=x) == pytest.approx(np.log(2.0), abs=0.05)

    def test_silence_vs_tone_gives_full_vuv_error(self):
        x = _sine(220, 1.0)
        y = np.zeros_like(x) + 1e-6  # near-silent but non-zero, avoids the ref-silence guard
        assert vuv_error(y, SR, ref=x) == pytest.approx(1.0, abs=0.05)

    def test_all_zero_reference_raises(self):
        x = _sine(220, 1.0)
        with pytest.raises(IntrusiveMetricError):
            log_f0_rmse(x, SR, ref=np.zeros_like(x))
        with pytest.raises(IntrusiveMetricError):
            vuv_error(x, SR, ref=np.zeros_like(x))

    def test_no_common_voiced_frame_raises(self):
        tone = _sine(220, 1.0)
        silence = np.zeros_like(tone) + 1e-6
        with pytest.raises(IntrusiveMetricError):
            log_f0_rmse(silence, SR, ref=tone)

    def test_signal_shorter_than_one_frame_raises(self):
        x = np.ones(50, dtype=np.float32) * 0.1
        with pytest.raises(IntrusiveMetricError):
            log_f0_rmse(x, SR, ref=x)
        with pytest.raises(IntrusiveMetricError):
            vuv_error(x, SR, ref=x)


# ---------------------------------------------------------------------- spectral (LSD/MSD/mel_l1)

class TestSpectralDistances:
    @pytest.mark.parametrize("metric", [lsd, msd, mel_l1])
    def test_identical_signals_give_zero(self, metric):
        x = _speechlike(1.0, SR, 0)
        assert metric(x, SR, ref=x) == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize("metric", [lsd, msd, mel_l1])
    def test_monotonic_increase_with_noise(self, metric):
        x = _speechlike(1.0, SR, 0)
        none = metric(x, SR, ref=x)
        mild = metric(_noisy(x, 30, 1), SR, ref=x)
        harsh = metric(_noisy(x, 0, 1), SR, ref=x)
        assert none < mild < harsh

    @pytest.mark.parametrize("metric", [lsd, msd, mel_l1])
    def test_all_zero_reference_raises(self, metric):
        x = _speechlike(1.0, SR, 0)
        with pytest.raises(IntrusiveMetricError):
            metric(x, SR, ref=np.zeros_like(x))

    @pytest.mark.parametrize("metric", [lsd, msd, mel_l1])
    def test_nan_in_degraded_raises(self, metric):
        x = _speechlike(1.0, SR, 0)
        y = x.copy()
        y[0] = np.nan
        with pytest.raises(IntrusiveMetricError):
            metric(y, SR, ref=x)

    @pytest.mark.parametrize("metric", [lsd, msd, mel_l1])
    def test_signal_shorter_than_one_frame_raises(self, metric):
        x = np.ones(50, dtype=np.float32) * 0.1
        with pytest.raises(IntrusiveMetricError):
            metric(x, SR, ref=x)

    @pytest.mark.parametrize("metric", [lsd, msd, mel_l1])
    def test_single_sample_inputs_raise(self, metric):
        x = np.array([0.5], dtype=np.float32)
        with pytest.raises(IntrusiveMetricError):
            metric(x, SR, ref=x)

    @pytest.mark.parametrize("metric", [lsd, msd, mel_l1])
    def test_stereo_input(self, metric):
        x = _speechlike(1.0, SR, 0)
        y = _noisy(x, 15, 1)
        stereo_x, stereo_y = np.stack([x, x], axis=1), np.stack([y, y], axis=1)
        assert metric(stereo_y, SR, ref=stereo_x) == pytest.approx(metric(y, SR, ref=x), abs=1e-4)


# -------------------------------------------------------------------- shared adversarial matrix

ALL_METRICS = [stoi, estoi, si_sdr, sdr, snr, mcd, log_f0_rmse, vuv_error, lsd, msd, mel_l1]
# log_f0_rmse is undefined (by design, see TestF0Metrics.test_no_common_voiced_frame_raises)
# when the degraded signal has no voiced frame at all, as an all-zero signal never does.
_SUPPORTS_ALL_ZERO_DEGRADED = [m for m in ALL_METRICS if m is not log_f0_rmse]


class TestSharedAdversarialCases:
    @pytest.mark.parametrize("metric", _SUPPORTS_ALL_ZERO_DEGRADED)
    def test_all_zero_degraded_does_not_raise_or_nan(self, metric):
        x = _speechlike(1.0, SR, 0)
        value = metric(np.zeros_like(x), SR, ref=x)
        assert np.isfinite(value)

    def test_all_zero_degraded_log_f0_rmse_raises(self):
        x = _speechlike(1.0, SR, 0)
        with pytest.raises(IntrusiveMetricError):
            log_f0_rmse(np.zeros_like(x), SR, ref=x)

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_sample_rate_mismatch_resamples_instead_of_raising(self, metric):
        x = _speechlike(1.0, SR, 0)
        y = _noisy(x, 15, 1)
        x_8k = x[::2].copy()
        assert np.isfinite(metric(y, SR, ref=x_8k, ref_sr=SR // 2))

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_length_mismatch_beyond_2x_raises(self, metric):
        x = _speechlike(0.5, SR, 0)
        y = _speechlike(2.0, SR, 0)
        with pytest.raises(IntrusiveMetricError):
            metric(y, SR, ref=x)
