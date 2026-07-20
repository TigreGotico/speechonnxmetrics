"""OnnxMetric: lazy, thread-safe session creation; context-manager close()."""
from __future__ import annotations

import os
import threading

import pytest

from speechonnxmetrics.base import ModelEntry, Metric, OnnxMetric

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "audio")
SOURCE_WAV = os.path.join(FIXTURES, "source.wav")


class _CountingMetric(OnnxMetric):
    name = "counting"
    intrusive = False

    def __init__(self):
        super().__init__(model=ModelEntry(alias="fake", hf_repo="r", hf_file="f", sample_rate=None))
        self.creations = 0

    def _create_session(self):
        self.creations += 1

        class _FakeSession:
            def run(self, output_names, feed):
                return [1.0]

        return _FakeSession()

    def _frontend(self, audio, sr):
        return {}

    def _postprocess(self, outputs):
        return 1.0


def test_construction_touches_neither_network_nor_disk(monkeypatch):
    called = []
    monkeypatch.setattr("speechonnxmetrics.resolver.resolve", lambda *a, **k: called.append(1))
    metric = _CountingMetric()
    assert metric._session is None
    assert called == []
    assert metric.creations == 0


def test_session_created_lazily_on_first_call():
    metric = _CountingMetric()
    assert metric.creations == 0
    metric(SOURCE_WAV, 10000)
    assert metric.creations == 1
    metric(SOURCE_WAV, 10000)
    assert metric.creations == 1  # reused, not recreated


def test_session_creation_is_thread_safe():
    metric = _CountingMetric()
    barrier = threading.Barrier(8)

    def _call():
        barrier.wait()
        metric(SOURCE_WAV, 10000)

    threads = [threading.Thread(target=_call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert metric.creations == 1


def test_close_releases_session_and_allows_recreation():
    metric = _CountingMetric()
    metric(SOURCE_WAV, 10000)
    assert metric.creations == 1
    metric.close()
    assert metric._session is None
    metric(SOURCE_WAV, 10000)
    assert metric.creations == 2


def test_context_manager_closes_on_exit():
    with _CountingMetric() as metric:
        metric(SOURCE_WAV, 10000)
        assert metric.creations == 1
    assert metric._session is None


def test_intrusive_metric_without_ref_raises():
    class _IntrusiveFake(_CountingMetric):
        name = "intrusive_fake"
        intrusive = True

    metric = _IntrusiveFake()
    with pytest.raises(ValueError, match="intrusive_fake.*requires ref"):
        metric(SOURCE_WAV, 10000)


def test_metric_protocol_runtime_checkable():
    metric = _CountingMetric()
    assert isinstance(metric, Metric)
