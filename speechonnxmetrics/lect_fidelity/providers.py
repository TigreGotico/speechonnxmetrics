"""Expected phones: the readings a lect licenses for each word of a text.

A provider turns a text and a lect into one :class:`WordPhones` per orthographic word:
the word's alternative readings, best first, and the free variants of the lect's
segments. Site finding and scoring see only this interface, so any phonemiser can
supply the expected side. Two ship here, both over the ``orthography2ipa`` candidate
lattice:

* :class:`Orthography2ipaProvider` (the default) reads the lect spec as it is;
* :class:`TugaphoneProvider` adds ``tugaphone``'s stages for the Portuguese-family
  lects: number verbalisation, sense-based homograph marking and the ``tugalex``
  pronunciation lexicon.

The readings of a word are its reading inside the sentence, with cross-word sandhi, and
its reading in isolation; where a process is optional (European word-final reduced e
before a vowel-initial word, a sandhi voicing) the two differ, and both are licensed. The free
variants are the lect spec's ``allophones`` table. A provider writing a notation other
than IPA names it in :attr:`PhoneProvider.notation`, and scriptconv converts it.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from functools import lru_cache

from speechonnxmetrics.lect_fidelity.notation import to_ipa
from speechonnxmetrics.lect_fidelity.phones import canonical, segment

_EXTRA = {
    "orthography2ipa": "speechonnxmetrics[lect]",
    "tugaphone": "speechonnxmetrics[lect-tugaphone]",
}


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise ImportError(
            f"this expected-phone provider requires {module}: pip install {_EXTRA[module]}"
        ) from exc


@dataclass(frozen=True)
class WordPhones:
    """The expected phones of one word in one lect.

    ``readings`` are alternative segment sequences, best first; ``variants`` maps a
    segment to the segments the lect realises it as freely, itself included, with
    ``""`` standing for no segment.
    """

    word: str
    readings: tuple[tuple[str, ...], ...]
    variants: dict[str, frozenset[str]] = field(default_factory=dict)

    @property
    def primary(self) -> tuple[str, ...]:
        return self.readings[0]


class PhoneProvider(abc.ABC):
    """Text and lect in, the lect's candidate phones per word out."""

    #: the provider's name in :data:`PROVIDERS` and in calibration tables
    name: str
    #: the scriptconv notation the provider writes; ``"ipa"`` needs no conversion
    notation: str = "ipa"

    @abc.abstractmethod
    def candidates(self, text: str, lect: str) -> list[WordPhones]:
        """One :class:`WordPhones` per orthographic word of ``text``, in order."""

    @abc.abstractmethod
    def inventory(self, lect: str) -> frozenset[str]:
        """Every phone the lect may realise, as canonical IPA segments."""

    def _segments(self, phones: str) -> tuple[str, ...]:
        return tuple(segment(to_ipa(phones, self.notation)))


class _LatticeProvider(PhoneProvider):
    """A provider reading an ``orthography2ipa`` engine: sentence and isolated readings
    of each word, and the spec's ``allophones`` as free variants."""

    def __init__(self, *, free_variants: bool = True) -> None:
        self.free_variants = free_variants

    @abc.abstractmethod
    def engine(self, lect: str):
        """The ``orthography2ipa.G2P`` engine for ``lect``."""

    def _variants(self, lect: str) -> dict[str, frozenset[str]]:
        if not self.free_variants:
            return {}
        table = self.engine(lect).spec.allophones or {}
        out: dict[str, frozenset[str]] = {}
        for phone, realisations in table.items():
            if not realisations:
                continue
            key = canonical(to_ipa(phone, self.notation))
            out[key] = frozenset(canonical(to_ipa(r, self.notation)) for r in realisations) | {key}
        return out

    @lru_cache(maxsize=None)
    def inventory(self, lect: str) -> frozenset[str]:
        from orthography2ipa import emission_inventory

        phones = {
            canonical(s) for p in emission_inventory(self.engine(lect).spec) for s in self._segments(p) if s[0].isalpha()
        }
        for realisations in self._variants(lect).values():
            phones |= {v for v in realisations if v}
        return frozenset(phones)

    def candidates(self, text: str, lect: str) -> list[WordPhones]:
        engine = self.engine(lect)
        variants = self._variants(lect)
        out = []
        for word in engine.transcribe_detailed(text).words:
            readings = [self._segments(word.ipa)]
            isolated = self._segments(engine.transcribe_word(word.word))
            if isolated and isolated not in readings:
                readings.append(isolated)
            if readings[0]:
                out.append(WordPhones(word.word, tuple(readings), variants))
        return out


class Orthography2ipaProvider(_LatticeProvider):
    """The ``orthography2ipa`` lect specs as they are."""

    name = "orthography2ipa"

    def engine(self, lect: str):
        return _o2i_engine(lect)


class TugaphoneProvider(_LatticeProvider):
    """``tugaphone``: the ``orthography2ipa`` lattice with number verbalisation,
    homograph marking and the ``tugalex`` lexicon, for 41 Portuguese-family lects.

    ``tugaphone`` registers the lexicon with ``orthography2ipa`` for the whole process,
    so once this provider has run, :class:`Orthography2ipaProvider` in the same
    process reads the lexicon too. Use one provider per process, or clear the
    registration (``tugaphone.lattice_core.clear_caches()`` and
    ``orthography2ipa.clear_lexicons()``) between them."""

    name = "tugaphone"

    def engine(self, lect: str):
        return _tugaphone_engine(lect)


@lru_cache(maxsize=None)
def _o2i_engine(lect: str):
    return _require("orthography2ipa").G2P(lect)


@lru_cache(maxsize=None)
def _tugaphone_engine(lect: str):
    _require("tugaphone")
    from tugaphone.lattice_core import engine
    from tugaphone.registry import resolve_lect

    return engine(resolve_lect(lect))


PROVIDERS: dict[str, type[PhoneProvider]] = {
    "orthography2ipa": Orthography2ipaProvider,
    "tugaphone": TugaphoneProvider,
}
DEFAULT_PROVIDER = "orthography2ipa"


def get_provider(provider: str | PhoneProvider) -> PhoneProvider:
    """A provider instance from its name, or ``provider`` itself."""
    if isinstance(provider, PhoneProvider):
        return provider
    if provider not in PROVIDERS:
        raise KeyError(f"unknown provider {provider!r}. Available: {', '.join(sorted(PROVIDERS))}")
    return PROVIDERS[provider]()
