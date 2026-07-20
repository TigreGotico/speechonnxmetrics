"""Public scoring API: :func:`score`, :func:`score_batch`, :func:`list_metrics`.

Only audio metrics (``kind="audio"`` in the registry) are scored here — text metrics
(WER/CER/...) compare strings, not audio, and are called directly from
:mod:`speechonnxmetrics.asr` instead.

``score_batch`` is the hot path for eval loops: each requested metric is resolved
from the registry exactly once and reused across every item, so an
:class:`~speechonnxmetrics.base.OnnxMetric` builds its session on the first item and
every later item reuses it. A metric raising on one item is recorded as a failure for
that item only — the rest of the batch keeps running, because a single corrupt file
must not sink an hours-long eval run.
"""
from __future__ import annotations

from typing import Any, Sequence

from speechonnxmetrics._dsp.audio import AudioLoadError, load_audio
from speechonnxmetrics.registry import RegistryEntry
from speechonnxmetrics.registry import get as _get_entry
from speechonnxmetrics.registry import list_metrics as list_metrics  # noqa: F401 (re-export)

AudioLike = Any


def _resolve(names: Sequence[str]) -> list[RegistryEntry]:
    if not names:
        raise ValueError("metrics must not be empty")
    seen: set[str] = set()
    entries = []
    for name in names:
        key = name.lower()
        if key in seen:
            raise ValueError(f"duplicate metric name in request: {name!r}")
        seen.add(key)
        entry = _get_entry(name)
        if entry.kind != "audio":
            raise ValueError(f"metric {name!r} is a text metric — call speechonnxmetrics.asr.{entry.name} directly")
        entries.append(entry)
    return entries


def _flatten(name: str, value: float | dict[str, float]) -> dict[str, float]:
    if isinstance(value, dict):
        return {f"{name}.{k}": v for k, v in value.items()}
    return {name: value}


def _load_item(audio: AudioLike, sr: int | None) -> tuple[Any, int]:
    return load_audio(audio, sample_rate=sr)


def score_batch(
    audios: Sequence[AudioLike],
    metrics: Sequence[str],
    refs: Sequence[AudioLike] | None = None,
    sr: int | None = None,
) -> list[dict[str, Any]]:
    """Score every item in ``audios`` against every metric in ``metrics``.

    ``refs``, when given, must have the same length as ``audios``. Each metric name is
    resolved from the registry once, so an ONNX-backed metric's session is built on
    the first item and reused for the rest. A metric that raises on a given item does
    not abort the batch: that item's value for that metric is ``None`` and the
    exception message is recorded under ``"_errors"`` in the item's result dict.
    """
    if refs is not None and len(refs) != len(audios):
        raise ValueError(f"refs ({len(refs)}) and audios ({len(audios)}) must have the same length")
    entries = _resolve(metrics)
    for entry in entries:
        if entry.intrusive and refs is None:
            raise ValueError(f"metric {entry.name!r} is intrusive and requires refs=...")

    results: list[dict[str, Any]] = []
    for i, deg in enumerate(audios):
        ref = refs[i] if refs is not None else None
        row: dict[str, Any] = {}
        errors: dict[str, str] = {}
        try:
            deg_audio, deg_sr = _load_item(deg, sr)
            ref_audio, ref_sr = (_load_item(ref, sr) if ref is not None else (None, None))
        except AudioLoadError as exc:
            for entry in entries:
                row[entry.name] = None
                errors[entry.name] = str(exc)
            row["_errors"] = errors
            results.append(row)
            continue

        for entry in entries:
            try:
                value = entry.fn(deg_audio, deg_sr, ref=ref_audio, ref_sr=ref_sr)
                row.update(_flatten(entry.name, value))
            except Exception as exc:  # noqa: BLE001 - one bad item must not sink the batch
                row[entry.name] = None
                errors[entry.name] = str(exc)
        if errors:
            row["_errors"] = errors
        results.append(row)
    return results


def score(
    audio: AudioLike, metrics: Sequence[str], ref: AudioLike | None = None, sr: int | None = None,
) -> dict[str, Any]:
    """Score a single ``audio`` item against every metric in ``metrics``. See
    :func:`score_batch` for the batch form used by eval loops."""
    refs = [ref] if ref is not None else None
    return score_batch([audio], metrics, refs=refs, sr=sr)[0]
