"""Lect fidelity: how far a speech clip realises one lect of a language against another.

Three parts:

* **expected phones**: a provider (:mod:`~speechonnxmetrics.lect_fidelity.providers`)
  gives each lect's candidate readings of the text, and the documented places where the
  two lects differ are the *discriminative sites*
  (:mod:`~speechonnxmetrics.lect_fidelity.sites`);
* **realised phones**: a phone recogniser backend turns the audio into IPA segments
  (:class:`AllosaurusBackend`, the default, or :class:`Wav2Vec2EspeakBackend`);
* **scoring**: each site is read as the lect whose candidates are nearer, or neither.
  The per-clip figures are the shares of sites per lect and a calibrated log-likelihood
  ratio with its decision; a posterior over the lects is given for pooled clips
  (:func:`pool`), with the pooled duration beside it.

A text with no discriminative site, or a clip with too little realised speech or far
more than its text accounts for, returns ``None``. Requires the optional ``lect`` extra (``pip install speechonnxmetrics[lect]``).
"""
from __future__ import annotations

from collections.abc import Sequence

from speechonnxmetrics.base import AudioLike
from speechonnxmetrics.lect_fidelity.backends import (
    BACKENDS,
    AllosaurusBackend,
    PhoneBackend,
    Wav2Vec2EspeakBackend,
)
from speechonnxmetrics.lect_fidelity.calibration import Calibration, fit, shipped
from speechonnxmetrics.lect_fidelity.providers import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    Orthography2ipaProvider,
    PhoneProvider,
    TugaphoneProvider,
    WordPhones,
    get_provider,
)
from speechonnxmetrics.lect_fidelity.scoring import (
    LectFidelityResult,
    PooledResult,
    SiteReading,
    pool,
    score_phones,
)
from speechonnxmetrics.lect_fidelity.sites import (
    SITE_CLASSES,
    Expectation,
    Site,
    discriminative_sites,
    expect,
)

DEFAULT_BACKEND = "allosaurus"


class LectFidelity:
    """Scores clips against a pair of lects with one backend and one provider.

    ``backend`` is a name from :data:`BACKENDS` or a :class:`PhoneBackend`; ``provider``
    a name from :data:`PROVIDERS` or a :class:`PhoneProvider`. ``calibration`` is
    ``"shipped"`` for the table measured for this combination, in either lect order,
    ``None`` for none (no log-likelihood ratio or decision), or a :class:`Calibration`.
    ``restrict_inventory`` decodes among the phones of the two lects only; left ``None``,
    it follows the calibration, which was measured one way or the other. Construction
    loads no model.
    """

    def __init__(
        self,
        lects: Sequence[str],
        backend: str | PhoneBackend = DEFAULT_BACKEND,
        *,
        provider: str | PhoneProvider = DEFAULT_PROVIDER,
        calibration: str | Calibration | None = "shipped",
        restrict_inventory: bool | None = None,
        **backend_kwargs: object,
    ) -> None:
        self.lects = tuple(lects)
        if len(self.lects) != 2 or self.lects[0] == self.lects[1]:
            raise ValueError(f"expected two distinct lects, got {lects!r}")
        if isinstance(backend, str):
            if backend not in BACKENDS:
                raise KeyError(f"unknown backend {backend!r}. Available: {', '.join(sorted(BACKENDS))}")
            backend = BACKENDS[backend](**backend_kwargs)
        self.backend = backend
        self.provider = get_provider(provider)
        if calibration == "shipped":
            calibration = shipped(self.lects, self.provider.name, self.backend.name)
        self.calibration = calibration
        if restrict_inventory is None:
            restrict_inventory = calibration.restrict_inventory if calibration is not None else False
        self.restrict_inventory = restrict_inventory

    def expect(self, text: str) -> Expectation:
        return expect(text, self.lects, self.provider)

    def inventory(self) -> frozenset[str]:
        """The union of the two lects' phone inventories."""
        return self.provider.inventory(self.lects[0]) | self.provider.inventory(self.lects[1])

    def score(self, audio: AudioLike, text: str, sr: int | None = None) -> LectFidelityResult | None:
        """Score one clip of ``text``; ``None`` without a site, without enough speech, or
        with far more speech than the text."""
        expectation = self.expect(text)
        if not expectation.sites:
            return None
        logits = self.backend.logits(audio, sr)
        inventory = self.inventory() if self.restrict_inventory else None
        phones = self.backend.phones_from_logits(logits, inventory)
        duration = _duration(audio, sr)
        return score_phones(phones, expectation, self.calibration, duration)

    def pool(self, results: Sequence[LectFidelityResult | None]) -> PooledResult:
        """Pool clip results of one voice with this scorer's calibration."""
        return pool([r for r in results if r is not None], self.calibration)


def _duration(audio: AudioLike, sr: int | None) -> float | None:
    from speechonnxmetrics._dsp.audio import load_audio

    x, rate = load_audio(audio, sample_rate=sr)
    return len(x) / rate if rate else None


def lect_fidelity(
    audio: AudioLike,
    text: str,
    lects: Sequence[str],
    *,
    sr: int | None = None,
    backend: str | PhoneBackend = DEFAULT_BACKEND,
    provider: str | PhoneProvider = DEFAULT_PROVIDER,
) -> LectFidelityResult | None:
    """One-shot :meth:`LectFidelity.score`."""
    return LectFidelity(lects, backend, provider=provider).score(audio, text, sr)


__all__ = [
    "BACKENDS",
    "PROVIDERS",
    "SITE_CLASSES",
    "AllosaurusBackend",
    "Calibration",
    "Expectation",
    "LectFidelity",
    "LectFidelityResult",
    "Orthography2ipaProvider",
    "PhoneBackend",
    "PhoneProvider",
    "PooledResult",
    "Site",
    "SiteReading",
    "TugaphoneProvider",
    "Wav2Vec2EspeakBackend",
    "WordPhones",
    "discriminative_sites",
    "expect",
    "fit",
    "lect_fidelity",
    "pool",
    "score_phones",
]
