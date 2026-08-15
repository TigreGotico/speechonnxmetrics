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


# ---------------------------------------------------------------------- #
# GPU / execution-provider resolution
# ---------------------------------------------------------------------- #
from speechonnxmetrics.base import PROVIDERS_ENV_VAR, _env_providers, _resolve_providers  # noqa: E402


class _ProvidersProbe(OnnxMetric):
    """Fake metric that records the providers passed to ``ort.InferenceSession``
    instead of touching onnxruntime/the network."""

    name = "providers_probe"
    intrusive = False

    def __init__(self, **kwargs):
        super().__init__(model=ModelEntry(alias="fake", hf_repo="r", hf_file="f", sample_rate=None), **kwargs)

    def _frontend(self, audio, sr):
        return {}

    def _postprocess(self, outputs):
        return 1.0


class _FakeInferenceSession:
    def __init__(self, path, providers=None):
        self.providers = providers


class _FakeOrt:
    def __init__(self, available):
        self._available = available
        self.InferenceSession = _FakeInferenceSession

    def get_available_providers(self):
        return self._available


def _build_session(monkeypatch, metric, available):
    monkeypatch.setattr("speechonnxmetrics.resolver.resolve", lambda *a, **k: "fake.onnx")
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", _FakeOrt(available))
    return metric._create_session()


def test_env_var_unset_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv(PROVIDERS_ENV_VAR, raising=False)
    assert _env_providers() == ["CPUExecutionProvider"]


def test_env_var_parses_comma_separated_list(monkeypatch):
    monkeypatch.setenv(PROVIDERS_ENV_VAR, "CUDAExecutionProvider,CPUExecutionProvider")
    assert _env_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_env_var_garbage_falls_back_to_cpu(monkeypatch):
    monkeypatch.setenv(PROVIDERS_ENV_VAR, "   ,  ,")
    assert _env_providers() == ["CPUExecutionProvider"]


def test_resolve_providers_intersects_with_available():
    resolved = _resolve_providers(["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"])
    assert resolved == ["CPUExecutionProvider"]


def test_resolve_providers_drops_unavailable_keeps_order():
    resolved = _resolve_providers(
        ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert resolved == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_always_keeps_cpu_fallback():
    resolved = _resolve_providers(["CUDAExecutionProvider"], ["CPUExecutionProvider"])
    assert resolved == ["CPUExecutionProvider"]


def test_constructor_none_defers_to_env_var_at_session_build(monkeypatch):
    monkeypatch.setenv(PROVIDERS_ENV_VAR, "CUDAExecutionProvider,CPUExecutionProvider")
    metric = _ProvidersProbe()
    assert metric.providers is None  # not resolved at construction time
    session = _build_session(monkeypatch, metric, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert session.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_explicit_constructor_arg_beats_env_var(monkeypatch):
    monkeypatch.setenv(PROVIDERS_ENV_VAR, "CUDAExecutionProvider,CPUExecutionProvider")
    metric = _ProvidersProbe(providers=["CPUExecutionProvider"])
    session = _build_session(monkeypatch, metric, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert session.providers == ["CPUExecutionProvider"]


def test_unavailable_provider_falls_back_to_cpu_not_crash(monkeypatch):
    monkeypatch.delenv(PROVIDERS_ENV_VAR, raising=False)
    metric = _ProvidersProbe(providers=["CUDAExecutionProvider"])
    session = _build_session(monkeypatch, metric, ["CPUExecutionProvider"])
    assert session.providers == ["CPUExecutionProvider"]


def test_with_providers_returns_independent_clone():
    metric = _ProvidersProbe()
    clone = metric._with_providers(["CUDAExecutionProvider"])
    assert clone is not metric
    assert clone.providers == ["CUDAExecutionProvider"]
    assert metric.providers is None
    assert clone._session is None
    assert clone._lock is not metric._lock
