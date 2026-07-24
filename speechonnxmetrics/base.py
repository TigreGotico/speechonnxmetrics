"""``Metric`` protocol and the ``OnnxMetric`` base class for ONNX-backed metrics.

Every ONNX-backed metric (MOS predictors, speaker-similarity, …) subclasses
:class:`OnnxMetric` and implements two hooks: :meth:`OnnxMetric._frontend` turns a
loaded audio array into an ONNX input feed, :meth:`OnnxMetric._postprocess` turns the
raw ONNX outputs into a score. Everything else — resampling, session creation,
thread-safe lazy init, batching — lives here so adapters stay pure numeric code.

The onnxruntime :class:`~onnxruntime.InferenceSession` is never created in
``__init__``: constructing a metric must not touch the network or the disk. The
session is created lazily, behind a lock, on first use.
"""
from __future__ import annotations

import abc
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence, Union, runtime_checkable

import numpy as np

from speechonnxmetrics._dsp.audio import load_audio
from speechonnxmetrics._dsp.resample import kaiser_resample

AudioLike = Union[str, Path, bytes, np.ndarray]
Score = Union[float, dict[str, float]]


@runtime_checkable
class Metric(Protocol):
    """Shape every audio metric — ONNX-backed or pure-numpy — conforms to."""

    name: str
    #: native/expected sample rate; ``None`` means the metric accepts any rate.
    sample_rate: int | None
    #: whether scoring requires a reference signal.
    intrusive: bool
    #: ``(low, high)`` bound on the returned score, or ``None`` if unbounded.
    range: tuple[float, float] | None
    #: whether a larger score means better quality.
    higher_is_better: bool

    def __call__(
        self, deg: AudioLike, sr: int, *, ref: AudioLike | None = None, ref_sr: int | None = None
    ) -> Score: ...


@dataclass
class ModelEntry:
    """Describes one ONNX model backing a metric — resolution + provenance."""

    alias: str
    hf_repo: str
    hf_file: str
    revision: str | None = None
    license: str = ""
    #: native sample rate the model expects; input is resampled to this.
    #: ``None`` means the model is rate-adaptive and audio is passed through untouched.
    sample_rate: int | None = 16000
    description: str = ""
    #: model-specific extra fields (frontend params, output layout, …).
    extra: dict[str, Any] = field(default_factory=dict)


class OnnxMetric(abc.ABC):
    """Base class for ONNX-backed metrics.

    Subclasses set the class attributes ``name``, ``intrusive``, ``range``,
    ``higher_is_better``, provide a :class:`ModelEntry` via ``model``, and implement
    :meth:`_frontend` and :meth:`_postprocess`. The inference session is created
    lazily and thread-safely on first call, never in ``__init__``.
    """

    name: str
    intrusive: bool = False
    range: tuple[float, float] | None = None
    higher_is_better: bool = True

    def __init__(
        self,
        model: ModelEntry,
        *,
        providers: Sequence[str] | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.model = model
        self.providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        self.cache_dir = cache_dir
        self._session: Any = None
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int | None:
        return self.model.sample_rate

    @property
    def model_info(self) -> dict[str, str | None]:
        """Provenance of the backing ONNX model: ``repo_id``, ``filename`` and the
        pinned ``revision``. Downstream evaluators can record this per prediction row
        so a judge's exact weights are always reproducible from a result set alone."""
        return {
            "repo_id": self.model.hf_repo,
            "filename": self.model.hf_file,
            "revision": self.model.revision,
        }

    # ------------------------------------------------------------------ #
    # session management
    # ------------------------------------------------------------------ #
    @property
    def session(self) -> Any:
        """The lazily-created onnxruntime session, resolving/downloading the model
        on first access."""
        if self._session is None:
            with self._lock:
                if self._session is None:
                    self._session = self._create_session()
        return self._session

    def _create_session(self) -> Any:
        import onnxruntime as ort  # type: ignore[import-untyped]

        from speechonnxmetrics.resolver import resolve

        path = resolve(
            self.model.hf_file,
            hf_repo=self.model.hf_repo,
            revision=self.model.revision,
            cache_dir=self.cache_dir,
        )
        return ort.InferenceSession(path, providers=self.providers)

    def close(self) -> None:
        """Release the underlying onnxruntime session, if one was created."""
        with self._lock:
            self._session = None

    def __enter__(self) -> "OnnxMetric":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # subclass hooks
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _frontend(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        """Turn a mono float32 array at ``sr`` into an ONNX input feed."""

    @abc.abstractmethod
    def _postprocess(self, outputs: list[np.ndarray]) -> Score:
        """Turn raw ONNX outputs into a score (a float, or a dict for multi-head models)."""

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def _load(self, audio: AudioLike, sr: int | None) -> tuple[np.ndarray, int]:
        target = self.model.sample_rate
        x, out_sr = load_audio(audio, sample_rate=sr)
        if target is not None and out_sr != target:
            x = kaiser_resample(x, out_sr, target)
            out_sr = target
        return x, out_sr

    def __call__(
        self, deg: AudioLike, sr: int, *, ref: AudioLike | None = None, ref_sr: int | None = None
    ) -> Score:
        if self.intrusive and ref is None:
            raise ValueError(f"metric {self.name!r} is intrusive and requires ref=...")
        audio, out_sr = self._load(deg, sr)
        feed = self._frontend(audio, out_sr)
        outputs = self.session.run(None, feed)
        return self._postprocess(outputs)
