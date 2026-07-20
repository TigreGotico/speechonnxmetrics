"""Registry: lookup, filtering, and the audio-vs-text metric split."""
from __future__ import annotations

import subprocess
import sys

import pytest

import speechonnxmetrics.registry as registry


def test_known_audio_metrics_registered():
    names = {e.name for e in registry.list_metrics(kind="audio")}
    assert {"stoi", "estoi", "si_sdr", "sdr", "snr", "mcd", "log_f0_rmse", "vuv_error", "lsd", "msd", "mel_l1"} <= names


def test_known_text_metrics_registered():
    names = {e.name for e in registry.list_metrics(kind="text")}
    assert names == {"wer", "cer", "mer", "wil", "wip"}


def test_get_is_case_insensitive():
    assert registry.get("STOI") is registry.get("stoi")


def test_get_unknown_metric_lists_available():
    with pytest.raises(KeyError, match="unknown metric"):
        registry.get("not-a-real-metric")


def test_list_metrics_filters_by_intrusive():
    assert all(e.intrusive for e in registry.list_metrics(intrusive=True))


def test_list_metrics_filters_by_requires_download():
    assert all(not e.requires_download for e in registry.list_metrics(requires_download=False))
    # the MOS predictors are the download-backed metrics; everything else is pure numpy
    assert {e.name for e in registry.list_metrics(requires_download=True)} == {
        "dnsmos", "dnsmos_p808", "sigmos", "utmos", "nisqa",
    }


def test_registry_import_does_not_import_onnxruntime():
    """Importing the package must not pull in onnxruntime, torch, or the network.

    Checked in a subprocess: asserting on this process's ``sys.modules`` only holds
    if no earlier test imported onnxruntime, which makes the result depend on test
    ordering rather than on the package.
    """
    probe = (
        "import sys, speechonnxmetrics; "
        "print(int('onnxruntime' in sys.modules), int('torch' in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True,
    ).stdout.split()
    assert out == ["0", "0"], f"import pulled in heavy deps: onnxruntime={out[0]} torch={out[1]}"
