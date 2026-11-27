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


def _in_range(value, rng):
    low, high = rng if rng is not None else (None, None)
    return (low is None or value >= low) and (high is None or value <= high)


def test_cer_range_admits_insertion_heavy_error_rate():
    from speechonnxmetrics.asr import cer

    # 5 insertions over a 2-character reference: 5 / 2 = 2.5.
    value = cer("ab", "abcdefg")
    assert value == 2.5
    assert _in_range(value, registry.get("cer").range)


def test_estoi_range_is_a_correlation():
    # ESTOI is a mean sample correlation (Jensen & Taal 2016), so it spans [-1, 1].
    assert registry.get("estoi").range == (-1.0, 1.0)


def test_wip_direction_matches_its_values():
    from speechonnxmetrics.asr import wip

    perfect = wip("the cat sat", "the cat sat")
    wrong = wip("the cat sat", "a dog ran")
    assert perfect == 1.0 and wrong == 0.0
    assert registry.get("wip").higher_is_better is (perfect > wrong)
