"""Phone recogniser backends: audio in, realised IPA segments out.

Every backend is an :class:`~speechonnxmetrics.base.OnnxMetric`, so it shares the
package's lazy session, provider selection and resampling. Its output is a list of IPA
segments, obtained by greedy CTC decoding and the backend's unit table from
:mod:`speechonnxmetrics.lect_fidelity.notation`.

Decoding can be restricted to the units whose IPA lies in a given phone inventory,
the union of the two lects' phones, so the recogniser chooses only among phones
either lect has; restricting to one lect would decide the measurement before it runs.
"""
from __future__ import annotations

import abc

import numpy as np

from speechonnxmetrics._dsp.fbank import fbank
from speechonnxmetrics.base import AudioLike, ModelEntry, OnnxMetric
from speechonnxmetrics.lect_fidelity import inventories as inv
from speechonnxmetrics.lect_fidelity.notation import allosaurus_to_ipa, espeak_to_ipa, unit_table
from speechonnxmetrics.lect_fidelity.phones import canonical, segment


def ctc_greedy(logits: np.ndarray, blank: int = 0) -> list[int]:
    """Best-path CTC decoding: argmax per frame, merge repeats, drop blanks."""
    best = np.asarray(logits).reshape(-1, np.asarray(logits).shape[-1]).argmax(axis=-1)
    out: list[int] = []
    prev = -1
    for unit in best.tolist():
        if unit != prev and unit != blank:
            out.append(unit)
        prev = unit
    return out


class PhoneBackend(OnnxMetric, abc.ABC):
    """A CTC phone recogniser whose output units are ``symbols``."""

    intrusive = False
    range = None
    higher_is_better = True
    #: output unit index -> backend symbol (``None`` for the blank)
    symbols: tuple[str | None, ...]
    #: backend symbol -> IPA string
    table: dict[str, str]

    def _decode(self, logits: np.ndarray, inventory: frozenset[str] | None) -> list[int]:
        logits = np.asarray(logits, dtype=np.float32).reshape(-1, np.asarray(logits).shape[-1])
        if inventory is not None:
            logits = np.where(self._allowed(inventory)[None, :], logits, -np.inf)
        return ctc_greedy(logits)

    def _allowed(self, inventory: frozenset[str]) -> np.ndarray:
        """Output units kept by an inventory restriction: the blank, and every unit
        whose IPA segments all lie in ``inventory`` (canonical forms)."""
        allowed = np.zeros(len(self.symbols), dtype=bool)
        allowed[0] = True
        for i, symbol in enumerate(self.symbols[1:], 1):
            ipa = self.table.get(symbol or "")
            if ipa and all(canonical(s) in inventory for s in segment(ipa)):
                allowed[i] = True
        return allowed

    def _to_ipa(self, units: list[int]) -> list[str]:
        return segment("".join(self.table.get(self.symbols[u] or "", "") + " " for u in units))

    def _postprocess(self, outputs: list[np.ndarray]) -> list[str]:
        return self._to_ipa(self._decode(outputs[0], None))

    def logits(self, audio: AudioLike, sr: int | None = None) -> np.ndarray:
        """The model's per-frame output scores for ``audio``."""
        x, out_sr = self._load(audio, sr)
        return self.session.run(None, self._frontend(x, out_sr))[0]

    def raw(self, audio: AudioLike, sr: int | None = None) -> list[str]:
        """The backend's own symbols for ``audio``, before mapping."""
        return [self.symbols[u] for u in ctc_greedy(self.logits(audio, sr))]

    def phones(
        self, audio: AudioLike, sr: int | None = None, inventory: frozenset[str] | None = None
    ) -> list[str]:
        """The realised IPA segments of ``audio``, decoded among the units whose IPA
        lies in ``inventory`` when one is given."""
        return self.phones_from_logits(self.logits(audio, sr), inventory)

    def phones_from_logits(self, logits: np.ndarray, inventory: frozenset[str] | None = None) -> list[str]:
        """:meth:`phones` from scores already computed by :meth:`logits`."""
        return self._to_ipa(self._decode(logits, inventory))


#: Kaldi MFCC parameters of the Allosaurus ``uni2005`` model (its ``pm_config.json``).
ALLOSAURUS_SR = 8000
_WIN, _HOP, _BANKS, _LOW, _HIGH, _LIFTER = 200, 80, 40, 40.0, 3800.0, 22
#: Allosaurus floors a silent band at float64 epsilon, not Kaldi's float32 epsilon; on
#: digital silence the two give log energies 20 apart, which the normalisation spreads
#: over every frame.
_FLOOR = float(np.finfo(np.float64).eps)


def _dct_ortho(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.sqrt(2.0 / n) * np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    m[0] /= np.sqrt(2.0)
    return m


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    """``audio`` in ``[-1, 1]`` stored as 16-bit PCM and read back, as Allosaurus reads its
    input from a WAV file. Allosaurus is sensitive at this level: on the twelve test clips
    the unquantised samples gave Allosaurus's own phones exactly on 3 clips, the 16-bit
    samples on 11."""
    pcm = (np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0) * 32767).astype(np.int16)
    return pcm.astype(np.float32) / 32768.0


def allosaurus_features(audio: np.ndarray) -> np.ndarray:
    """Allosaurus input features for 8 kHz mono audio in ``[-1, 1]``: ``(T/3, 120)``.

    40 Kaldi MFCCs (povey window, 40 mel banks over 40 to 3800 Hz, DCT-II, lifter 22,
    no energy coefficient) on 16-bit sample values, normalised to zero mean and unit
    variance per coefficient over the utterance, then each frame stacked with its two
    neighbours, wrapping at the edges, and every third stacked frame kept.
    """
    x = np.asarray(audio, dtype=np.float64).reshape(-1) * 32768.0
    if x.size < _WIN:
        x = np.pad(x, (0, _WIN - x.size))
    logmel = fbank(
        x, ALLOSAURUS_SR, _WIN, _HOP, _BANKS, window="povey", low_freq=_LOW, high_freq=_HIGH, floor=_FLOOR
    )
    cep = logmel.astype(np.float64) @ _dct_ortho(_BANKS).T
    cep *= 1.0 + (_LIFTER / 2.0) * np.sin(np.pi * np.arange(_BANKS) / _LIFTER)
    std = cep.std(axis=0)
    cep = (cep - cep.mean(axis=0)) / np.where(std > 0, std, 1.0)
    stacked = np.concatenate([np.roll(cep, 1, axis=0), cep, np.roll(cep, -1, axis=0)], axis=1)
    return stacked[::3].astype(np.float32)


class AllosaurusBackend(PhoneBackend):
    """Allosaurus ``uni2005``, a universal phone recogniser; GPL-3.0 weights. The
    default backend."""

    name = "allosaurus"
    symbols = (None,) + inv.ALLOSAURUS_INVENTORY
    table = unit_table(inv.ALLOSAURUS_INVENTORY, allosaurus_to_ipa)

    def __init__(self, model: str | None = None, **kwargs: object) -> None:
        entry = ModelEntry(
            alias="allosaurus",
            hf_repo="TigreGotico/allosaurus-onnx",
            hf_file=model or "allosaurus_uni2005.onnx",
            revision="1234e2ef6ee47275cd865e5836a974ecb98bcf6f",
            license="GPL-3.0",
            sample_rate=ALLOSAURUS_SR,
            description="Allosaurus uni2005 universal phone recogniser, acoustic model only",
        )
        super().__init__(entry, **kwargs)  # type: ignore[arg-type]

    def _frontend(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        return {"feats": allosaurus_features(to_pcm16(audio))[np.newaxis]}


class Wav2Vec2EspeakBackend(PhoneBackend):
    """``facebook/wav2vec2-xlsr-53-espeak-cv-ft``, XLSR-53 fine-tuned on espeak-ng
    phones over Common Voice; Apache-2.0 weights.

    It cannot judge Brazilian Portuguese: its Portuguese training labels came from
    espeak-ng's European voice, so it writes European phones (``ɨ``, coda ``ʃ``) for
    Brazilian speech. It is kept for comparison and for lect pairs where that bias is
    measured not to apply."""

    name = "wav2vec2_espeak"
    symbols = (None,) + inv.ESPEAK_INVENTORY[1:]
    table = unit_table(inv.ESPEAK_INVENTORY, espeak_to_ipa)

    def __init__(self, model: str | None = None, **kwargs: object) -> None:
        entry = ModelEntry(
            alias="wav2vec2_espeak",
            hf_repo="TigreGotico/wav2vec2-xlsr-53-espeak-cv-ft-onnx",
            hf_file=model or "wav2vec2_xlsr53_espeak_cv_ft.onnx",
            revision="b07c09a08486c38f4c736cb1735e030f40e37bb7",
            license="Apache-2.0",
            sample_rate=16000,
            description="XLSR-53 fine-tuned on espeak-ng phone labels over Common Voice",
        )
        super().__init__(entry, **kwargs)  # type: ignore[arg-type]

    def _frontend(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size < 400:
            x = np.pad(x, (0, 400 - x.size))
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        return {"input_values": x[np.newaxis].astype(np.float32)}


BACKENDS = {"allosaurus": AllosaurusBackend, "wav2vec2_espeak": Wav2Vec2EspeakBackend}
