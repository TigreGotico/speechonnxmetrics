"""Registry: lookup, filtering, and the audio-vs-text metric split."""
from __future__ import annotations

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
    # every currently-registered metric is pure numpy, none require a download
    assert all(not e.requires_download for e in registry.list_metrics(requires_download=False))
    assert registry.list_metrics(requires_download=True) == []


def test_registry_import_does_not_import_onnxruntime():
    import sys

    assert "onnxruntime" not in sys.modules
