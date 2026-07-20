"""Name -> metric registry.

Two kinds of metric live here, deliberately not squeezed into one signature:

* **audio** metrics conform to :class:`speechonnxmetrics.base.Metric` — they take
  ``(deg, sr, *, ref=None, ref_sr=None)`` and return a float or a dict of floats. Both
  the pure-numpy :mod:`speechonnxmetrics.intrusive` functions and any
  :class:`~speechonnxmetrics.base.OnnxMetric` subclass fit this shape.
* **text** metrics (WER/CER/MER/WIL/WIP from :mod:`speechonnxmetrics.asr`) score two
  strings/corpora, not audio. Forcing them through the audio ``Metric`` signature would
  mean either lying about ``sample_rate``/``ref`` semantics or inventing a fake audio
  wrapper around a text comparison — instead they are registered under
  ``kind="text"`` with their own ``(reference, hypothesis) -> float`` shape, kept in
  the same registry so ``list_metrics()``/the CLI can enumerate everything in one
  place while ``score()``/``score_batch()`` (audio-only) simply never dispatch to them.

Importing this module touches neither onnxruntime nor the network — it only imports
the pure-numpy metric implementations, so ``import speechonnxmetrics`` stays fast and
offline-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from speechonnxmetrics import asr, intrusive

AudioFn = Callable[..., Any]
TextFn = Callable[..., float]


@dataclass(frozen=True)
class RegistryEntry:
    """One registered metric plus the metadata the CLI/`list_metrics` need."""

    name: str
    kind: str  # "audio" | "text"
    intrusive: bool
    requires_download: bool
    fn: AudioFn | TextFn
    range: tuple[float, float] | None = None
    higher_is_better: bool = True
    description: str = ""


_REGISTRY: dict[str, RegistryEntry] = {}


def register(entry: RegistryEntry) -> RegistryEntry:
    """Register *entry* under its (lowercased) name; returns *entry*."""
    _REGISTRY[entry.name.lower()] = entry
    return entry


def get(name: str) -> RegistryEntry:
    """Look up a registered metric by name (case-insensitive).

    Raises :class:`KeyError` naming every available metric when ``name`` is unknown.
    """
    key = name.lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown metric {name!r}. Available metrics: {known or '(none registered)'}")
    return _REGISTRY[key]


def list_metrics(
    intrusive: bool | None = None, requires_download: bool | None = None, kind: str | None = None
) -> list[RegistryEntry]:
    """List registered metrics, optionally filtered by ``intrusive``/``requires_download``/``kind``."""
    entries: Iterable[RegistryEntry] = _REGISTRY.values()
    if intrusive is not None:
        entries = (e for e in entries if e.intrusive == intrusive)
    if requires_download is not None:
        entries = (e for e in entries if e.requires_download == requires_download)
    if kind is not None:
        entries = (e for e in entries if e.kind == kind)
    return sorted(entries, key=lambda e: e.name)


def _wrap_intrusive(fn: AudioFn, *, higher_is_better: bool, range_: tuple[float, float] | None) -> AudioFn:
    """Adapt a ``metric(deg, sr, *, ref, ref_sr=None)`` function to the ``Metric``
    ``__call__`` shape, where ``ref`` is optional in the signature but required at
    call time for an intrusive metric."""

    def call(deg: Any, sr: int, *, ref: Any = None, ref_sr: int | None = None) -> float:
        return fn(deg, sr, ref=ref, ref_sr=ref_sr)

    call.__name__ = getattr(fn, "__name__", "metric")
    return call


_INTRUSIVE_SPECS: list[tuple[str, AudioFn, bool, tuple[float, float] | None]] = [
    ("stoi", intrusive.stoi, True, (0.0, 1.0)),
    ("estoi", intrusive.estoi, True, (0.0, 1.0)),
    ("si_sdr", intrusive.si_sdr, True, None),
    ("sdr", intrusive.sdr, True, None),
    ("snr", intrusive.snr, True, None),
    ("mcd", intrusive.mcd, False, None),
    ("log_f0_rmse", intrusive.log_f0_rmse, False, None),
    ("vuv_error", intrusive.vuv_error, False, (0.0, 1.0)),
    ("lsd", intrusive.lsd, False, None),
    ("msd", intrusive.msd, False, None),
    ("mel_l1", intrusive.mel_l1, False, None),
]

for _name, _fn, _higher, _range in _INTRUSIVE_SPECS:
    register(
        RegistryEntry(
            name=_name,
            kind="audio",
            intrusive=True,
            requires_download=False,
            fn=_wrap_intrusive(_fn, higher_is_better=_higher, range_=_range),
            range=_range,
            higher_is_better=_higher,
            description=(_fn.__doc__ or "").strip().splitlines()[0] if _fn.__doc__ else "",
        )
    )

_TEXT_SPECS: list[tuple[str, TextFn]] = [
    ("wer", asr.wer),
    ("cer", asr.cer),
    ("mer", asr.mer),
    ("wil", asr.wil),
    ("wip", asr.wip),
]

for _name, _fn in _TEXT_SPECS:
    register(
        RegistryEntry(
            name=_name,
            kind="text",
            intrusive=True,
            requires_download=False,
            fn=_fn,
            range=(0.0, 1.0) if _name != "wer" else None,
            higher_is_better=False,
            description=(_fn.__doc__ or "").strip().splitlines()[0] if _fn.__doc__ else "",
        )
    )
