"""Intrusive (reference-based) metrics: STOI, ESTOI, SI-SDR, SDR, SNR, MCD, log-F0
RMSE, V/UV error, LSD, MSD, Mel-L1.

Every metric shares the signature ``metric(deg, sr, *, ref, ref_sr=None)`` — see
:mod:`speechonnxmetrics.intrusive._common` for the shared length/rate-matching policy
and :class:`speechonnxmetrics.intrusive._common.IntrusiveMetricError` for the error
type raised on unscorable inputs.
"""
from __future__ import annotations

from speechonnxmetrics.intrusive._common import IntrusiveMetricError
from speechonnxmetrics.intrusive.cepstral import log_f0_rmse, mcd, vuv_error
from speechonnxmetrics.intrusive.sdr import sdr, si_sdr, snr
from speechonnxmetrics.intrusive.spectral import lsd, mel_l1, msd
from speechonnxmetrics.intrusive.stoi import estoi, stoi

__all__ = [
    "IntrusiveMetricError",
    "stoi",
    "estoi",
    "si_sdr",
    "sdr",
    "snr",
    "mcd",
    "log_f0_rmse",
    "vuv_error",
    "lsd",
    "msd",
    "mel_l1",
]
