"""Recogniser symbols and expected phones, brought to one IPA convention.

A recogniser emits units from a fixed inventory, and an expected-phone provider writes
a phone notation. Before anything is compared, both are converted to IPA in the
convention :mod:`speechonnxmetrics.lect_fidelity.phones` segments. The conversion has
two kinds of row, kept apart because only one of them is a fact:

* **notation rows** spell the same phone in IPA. ASCII mnemonics (espeak-ng writes
  ``S``, ``N``, ``dZ``, ``t[``, ``u:`` where its IPA table has no entry) are read by
  :mod:`scriptconv`'s Kirshenbaum table, and expected phones in any notation
  :mod:`scriptconv` knows are converted by :func:`scriptconv.notation.convert`. Units
  that spell one affricate as two letters take the tie bar, and a few code points are
  replaced by their IPA ones (:data:`ALLOSAURUS_SPELLING`, :data:`ESPEAK_SPELLING`).
  The spelling rows are proposed to :mod:`scriptconv` as the ``allosaurus`` and
  ``espeak-ipa`` notations (TigreGotico/scriptconv#138); they are applied here until a
  scriptconv release carries them.
* **metric decisions** map a unit that has no exact IPA counterpart: a tone number, a
  vowel followed by espeak-ng's tone-3 mark, an r-coloured vowel, an English reduced
  vowel, a diphthong whose offglide this metric writes as a glide
  (:data:`ESPEAK_DECISIONS`). Each one is a choice made for this metric, and a unit no
  choice covers is listed as unmapped and dropped from the realised string.
"""
from __future__ import annotations

import unicodedata

from scriptconv.notation import convert, kirshenbaum_to_ipa

from speechonnxmetrics.lect_fidelity import inventories as inv

#: Allosaurus units that spell an IPA phone with a non-IPA code point or without the
#: tie bar of a single affricate unit.
ALLOSAURUS_SPELLING: dict[str, str] = {
    "dʒ": "d͡ʒ",
    "g": "ɡ",
    "gʲ": "ɡʲ",
    "gʲj": "ɡʲj",
    "gʷ": "ɡʷ",
    "gː": "ɡː",
    "kx": "k͡x",
    "pf": "p͡f",
    "ts": "t͡s",
    "tsʰ": "t͡sʰ",
    "tɕ": "t͡ɕ",
    "tɕʰ": "t͡ɕʰ",
    "tʂ": "t͡ʂ",
    "tʂʰ": "t͡ʂʰ",
    "tʃ": "t͡ʃ",
    "ә": "ə",  # U+04D9 CYRILLIC SMALL LETTER SCHWA
}

#: Allosaurus units with no IPA value: ``I`` is an output unit without a phone, and
#: ``ː`` is a bare length mark.
ALLOSAURUS_UNMAPPED: tuple[str, ...] = ("I", "ː")

#: espeak-ng phoneme units, after the Kirshenbaum reading of their ASCII mnemonics, that
#: spell a single affricate without its tie bar, a retroflex with espeak-ng's ``.``
#: mnemonic, or aspiration with a plain ``h``.
ESPEAK_SPELLING: dict[str, str] = {
    "ts": "t͡s",
    "tsʲ": "t͡sʲ",
    "tsː": "t͡sː",
    "tsh": "t͡sʰ",
    "dʒ": "d͡ʒ",
    "dʒʲ": "d͡ʒʲ",
    "dʒː": "d͡ʒː",
    "dzː": "d͡zː",
    "tʃ": "t͡ʃ",
    "tʃʲ": "t͡ʃʲ",
    "tʃː": "t͡ʃː",
    "tʃʰ": "t͡ʃʰ",
    "tɕ": "t͡ɕ",
    "tɕʲ": "t͡ɕʲ",
    "tɕh": "t͡ɕʰ",
    "dʑ": "d͡ʑ",
    "dʑʲ": "d͡ʑʲ",
    "pf": "p͡f",
    "s.": "ʂ",
    "ts.": "t͡ʂ",
    "ts.h": "t͡ʂʰ",
    "th": "tʰ",
    "kh": "kʰ",
    "ph": "pʰ",
}

#: espeak-ng units with no exact IPA counterpart, mapped by a decision of this metric.
#: Tone numbers and the tone-3 mark (espeak-ng writes Mandarin tone 3 as the digit 3,
#: which its own IPA table then reads as ``ɜ``) are dropped, since lect sites are
#: segmental; the r-coloured vowel becomes schwa plus approximant; the English reduced
#: vowel ``ᵻ`` becomes ``ɨ``; espeak-ng's apical ``i.`` becomes ``ɨ``; diphthong
#: offglides become glides, the convention of ``orthography2ipa``; and ``r.``, the
#: Mandarin rhotic, becomes ``ɻ``.
_ESPEAK_UNIT_DECISIONS: dict[str, str] = {
    "ɚ": "əɹ",
    "aɪɚ": "ajəɹ",
    "aɪ": "aj",
    "eɪ": "ej",
    "oʊ": "ow",
    "aʊ": "aw",
    "ɔɪ": "ɔj",
    "oɪ": "oj",
    "əɪ": "əj",
    "ɛɪ": "ɛj",
    "eʊ": "ew",
    "əʊ": "əw",
    "ɛʊ": "ɛw",
    "iʊ": "iw",
    "ɐ̃ʊ̃": "ɐ̃w̃",
    "ᵻ": "ɨ",
    "i.": "ɨ",
    "i.ː": "ɨː",
    "onɡ": "oŋ",
    "r.": "ɻ",
}


def _espeak_decisions() -> dict[str, str]:
    out = dict(_ESPEAK_UNIT_DECISIONS)
    for unit in inv.ESPEAK_INVENTORY:
        base = unit.rstrip("12345")
        if base == unit and len(unit) > 1 and unit.endswith("ɜ"):
            base = unit[:-1]
        if base and base != unit:
            out[unit] = out.get(base, base)
    return out


ESPEAK_DECISIONS: dict[str, str] = _espeak_decisions()

#: espeak-ng notations with no IPA reading and no decision: unconverted mnemonics
#: (``u"``, ``a.``, ``u.``, the ``^`` forms), the unknown-phone mark ``??``, a bare
#: palatalisation mark and a bare tone number.
ESPEAK_UNMAPPED: tuple[str, ...] = (
    "??", 'u"', "a.", "a.ː", "u.", "u.ː", "t^", "d^",
    "s^", "t^ː", "ɪ^", "ʲ", "yɛ5ʲ", "1",
)

#: Tokenizer symbols of the wav2vec2 vocabulary that are not phones; ``<pad>`` is the
#: CTC blank.
ESPEAK_SPECIAL: tuple[str, ...] = ("<pad>", "<s>", "</s>", "<unk>")


#: Phones of the European or Brazilian Portuguese inventory that a backend has no unit
#: for: Allosaurus has four nasal vowels and no nasal glides, and neither recogniser
#: writes the apical sibilants of northern European Portuguese. No site class needs
#: any of them.
GAPS: dict[str, frozenset[str]] = {
    "allosaurus": frozenset({"j̃", "w̃", "s̺", "z̺", "õ", "ĩ", "ũ", "ɐ̃", "ẽ", "ə̃", "ɨ̃", "ʌ̃"}),
    "wav2vec2_espeak": frozenset({"j̃", "s̺", "z̺", "ə̃", "ɨ̃", "ʌ̃", "ɦ", "ʀ"}),
}


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def allosaurus_to_ipa(unit: str) -> str | None:
    """The IPA of one Allosaurus unit, or ``None`` when it has no IPA value."""
    if unit in ALLOSAURUS_UNMAPPED:
        return None
    return _nfc(ALLOSAURUS_SPELLING.get(unit, unit))


def espeak_to_ipa(unit: str) -> str | None:
    """The IPA of one espeak-ng phoneme unit, or ``None`` when it has none.

    A metric decision wins over the notation; otherwise ASCII mnemonics are read by
    scriptconv's Kirshenbaum table and the spelling rows complete the conversion.
    """
    if unit in ESPEAK_UNMAPPED or unit in ESPEAK_SPECIAL:
        return None
    if unit in ESPEAK_DECISIONS:
        return _nfc(ESPEAK_DECISIONS[unit])
    if unit in ESPEAK_SPELLING:
        return _nfc(ESPEAK_SPELLING[unit])
    ipa = kirshenbaum_to_ipa(unit) if any(c.isascii() and not c.islower() for c in unit) else unit
    return _nfc(ESPEAK_SPELLING.get(ipa, ipa))


def unit_table(inventory, to_ipa) -> dict[str, str]:
    """Backend unit to IPA for every unit of ``inventory`` that has an IPA reading."""
    out = {}
    for unit in inventory:
        ipa = to_ipa(unit)
        if ipa is not None:
            out[unit] = ipa
    return out


def to_ipa(text: str, notation: str) -> str:
    """``text`` written in ``notation``, in NFC IPA; scriptconv converts every
    non-IPA notation it knows."""
    if notation != "ipa":
        text = convert(text, notation, "ipa", errors="strict")
    return _nfc(text)
