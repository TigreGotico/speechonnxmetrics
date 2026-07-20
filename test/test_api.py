"""score()/score_batch(): flattening, batch session reuse, and failure isolation."""
from __future__ import annotations

import os

import numpy as np
import pytest

import speechonnxmetrics.registry as registry
from speechonnxmetrics import score, score_batch
from speechonnxmetrics.base import ModelEntry, OnnxMetric

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "audio")
SOURCE_WAV = os.path.join(FIXTURES, "source.wav")


class _FakeSession:
    def __init__(self, outputs):
        self._outputs = outputs

    def run(self, output_names, feed):
        return self._outputs


class _CountingMetric(OnnxMetric):
    """Fake ONNX metric that never touches onnxruntime, counting session creations."""

    name = "fake_counting"
    intrusive = False

    def __init__(self, outputs=(np.array([0.42], dtype=np.float32),), fail_on: set[int] | None = None):
        super().__init__(model=ModelEntry(alias="fake", hf_repo="r", hf_file="f", sample_rate=None))
        self.creations = 0
        self._outputs = outputs
        self._fail_on = fail_on or set()
        self._calls = 0

    def _create_session(self):
        self.creations += 1
        return _FakeSession(self._outputs)

    def _frontend(self, audio, sr):
        return {}

    def _postprocess(self, outputs):
        self._calls += 1
        if self._calls in self._fail_on:
            raise RuntimeError(f"boom on call {self._calls}")
        return float(outputs[0][0])


class _DictMetric(_CountingMetric):
    name = "fake_dict"

    def _postprocess(self, outputs):
        return {"sig": 1.0, "bak": 2.0, "ovrl": 3.0}


@pytest.fixture
def register_fake():
    registered = []

    def _register(entry: registry.RegistryEntry):
        registry.register(entry)
        registered.append(entry.name)
        return entry

    yield _register
    for name in registered:
        registry._REGISTRY.pop(name.lower(), None)


def test_score_flattens_dict_returning_metric(register_fake):
    metric = _DictMetric()
    register_fake(registry.RegistryEntry(name="fake_dict", kind="audio", intrusive=False, requires_download=True, fn=metric))
    result = score(SOURCE_WAV, ["fake_dict"])
    assert result == {"fake_dict.sig": 1.0, "fake_dict.bak": 2.0, "fake_dict.ovrl": 3.0}


def test_score_batch_creates_session_exactly_once(register_fake):
    metric = _CountingMetric()
    register_fake(registry.RegistryEntry(name="fake_counting", kind="audio", intrusive=False, requires_download=True, fn=metric))
    score_batch([SOURCE_WAV, SOURCE_WAV, SOURCE_WAV], ["fake_counting"])
    assert metric.creations == 1


def test_score_batch_isolates_per_item_failure(register_fake):
    metric = _CountingMetric(fail_on={2})
    register_fake(registry.RegistryEntry(name="fake_counting", kind="audio", intrusive=False, requires_download=True, fn=metric))
    results = score_batch([SOURCE_WAV, SOURCE_WAV, SOURCE_WAV], ["fake_counting"])
    assert results[0]["fake_counting"] == pytest.approx(0.42)
    assert results[1]["fake_counting"] is None
    assert "fake_counting" in results[1]["_errors"]
    assert results[2]["fake_counting"] == pytest.approx(0.42)


def test_score_batch_none_audio_isolated_not_batch_aborting(register_fake):
    metric = _CountingMetric()
    register_fake(registry.RegistryEntry(name="fake_counting", kind="audio", intrusive=False, requires_download=True, fn=metric))
    results = score_batch([None, SOURCE_WAV], ["fake_counting"])
    assert results[0]["fake_counting"] is None
    assert "fake_counting" in results[0]["_errors"]
    assert results[1]["fake_counting"] == pytest.approx(0.42)


def test_intrusive_metric_without_ref_raises_named_error():
    with pytest.raises(ValueError, match="stoi.*requires refs"):
        score_batch([SOURCE_WAV], ["stoi"])


def test_unknown_metric_lists_available():
    with pytest.raises(KeyError, match="unknown metric"):
        score_batch([SOURCE_WAV], ["not-a-metric"])


def test_empty_metric_list_raises():
    with pytest.raises(ValueError, match="metrics must not be empty"):
        score_batch([SOURCE_WAV], [])


def test_duplicate_metric_names_raise():
    with pytest.raises(ValueError, match="duplicate metric name"):
        score_batch([SOURCE_WAV], ["stoi", "stoi"], refs=[SOURCE_WAV])


def test_duplicate_metric_names_case_insensitive_raise():
    with pytest.raises(ValueError, match="duplicate metric name"):
        score_batch([SOURCE_WAV], ["stoi", "STOI"], refs=[SOURCE_WAV])


def test_mismatched_audios_and_refs_length_raises():
    with pytest.raises(ValueError, match="refs .* audios"):
        score_batch([SOURCE_WAV, SOURCE_WAV], ["stoi"], refs=[SOURCE_WAV])


def test_score_batch_with_real_intrusive_metric():
    results = score_batch([SOURCE_WAV], ["stoi"], refs=[SOURCE_WAV])
    assert results[0]["stoi"] == pytest.approx(1.0, abs=1e-6)
