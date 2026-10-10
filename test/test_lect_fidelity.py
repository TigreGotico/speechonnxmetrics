"""Tests for the lect-fidelity metric, on European against Brazilian Portuguese.

Test data, all public:

* sentences are written here; their expected phones come from ``orthography2ipa`` and
  ``tugaphone`` (both Apache-2.0);
* ``fixtures/tiny_allosaurus.onnx`` and ``fixtures/tiny_wav2vec2.onnx`` are one-layer
  graphs with the recognisers' input and output shapes, written by
  ``fixtures/generate_tiny_backends.py``; they exercise each backend's whole path
  without the real models;
* ``fixtures/allosaurus_features_reference.npz`` holds Allosaurus's own MFCC frontend
  on one clip, written by ``fixtures/generate_allosaurus_features.py``;
* ``fixtures/backend_units.json`` is the golden IPA of every output unit of both
  recognisers;
* the twelve clips in ``fixtures/audio/lect`` are synthesised by
  ``fixtures/generate_lect_audio.py`` with two Piper voices (see the manifest for their
  lineage), and ``fixtures/*_reference.json`` hold the reference recognisers' own
  outputs on them.

The tests that need the real ONNX exports run when ``SPEECHONNXMETRICS_LECT_MODELS``
names a directory holding them, and skip otherwise; everything else runs everywhere.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from speechonnxmetrics._dsp.audio import load_audio
from speechonnxmetrics._dsp.resample import kaiser_resample
from speechonnxmetrics.lect_fidelity import (
    SITE_CLASSES,
    AllosaurusBackend,
    LectFidelity,
    PhoneBackend,
    PhoneProvider,
    Wav2Vec2EspeakBackend,
    WordPhones,
    discriminative_sites,
    expect,
    fit,
    pool,
    score_phones,
)
from speechonnxmetrics.lect_fidelity import inventories as inv
from speechonnxmetrics.lect_fidelity import notation
from speechonnxmetrics.lect_fidelity.backends import allosaurus_features, ctc_greedy, to_pcm16
from speechonnxmetrics.lect_fidelity.calibration import Calibration, fit_class, load_table, shipped
from speechonnxmetrics.lect_fidelity.phones import align, canonical, distance, segment
from speechonnxmetrics.lect_fidelity.providers import Orthography2ipaProvider, TugaphoneProvider
from speechonnxmetrics.lect_fidelity.scoring import MAX_COVERAGE, MIN_COVERAGE, MIN_PHONES, NEITHER
from speechonnxmetrics.lect_fidelity.sites import POST_LEXICAL, PRE_LEXICAL, classify, roles

FIXTURES = Path(__file__).parent / "fixtures"
CLIPS = FIXTURES / "audio" / "lect"
MANIFEST = json.loads((CLIPS / "manifest.json").read_text())
UNITS = json.loads((FIXTURES / "backend_units.json").read_text())
LECTS = ("pt-PT", "pt-BR")
CLASSES = {c.name for c in SITE_CLASSES}


@pytest.fixture(autouse=True)
def _isolate_tugaphone_lexicon():
    """tugaphone registers its lexicon with orthography2ipa for the whole process; drop
    it after each test so the next one sees the bare specs."""
    yield
    import orthography2ipa
    from tugaphone.lattice_core import clear_caches

    from speechonnxmetrics.lect_fidelity import providers

    clear_caches()
    orthography2ipa.clear_lexicons()
    providers._tugaphone_engine.cache_clear()


# --- segmentation, distance, alignment -------------------------------------------------------


def test_segment_keeps_diacritics_and_tied_affricates_together():
    assert segment("ˈt͡ʃivi") == ["t͡ʃ", "i", "v", "i"]
    assert segment("ˈpɐ̃w̃ kʰa") == ["p", "ɐ̃", "w̃", "kʰ", "a"]
    assert segment("ˈkazɐz eʃ.ˈtɐ̃w̃") == ["k", "a", "z", "ɐ", "z", "e", "ʃ", "t", "ɐ̃", "w̃"]


def test_segment_joins_untied_portuguese_affricates():
    assert segment("ˈtaɦdʒɪ") == ["t", "a", "ɦ", "d͡ʒ", "ɪ"]
    assert segment("t ʃ i") == ["t͡ʃ", "i"]
    assert segment("ts") == ["t", "s"]


def test_distance_orders_near_and_far_segments():
    assert distance("s", "s") == 0.0
    assert distance("t͡ʃ", "tʃ") == 0.0
    assert distance("ʂ", "ʃ") < distance("ʂ", "a")
    assert distance("u", "w") < distance("ɫ", "w")
    assert distance("s̪", "s") < distance("s̪", "ʃ")
    assert distance("a", "k") == 1.0


def test_align_identity_costs_nothing_and_gaps_cost_one():
    cost, ops = align(["a", "b"], ["a", "b"])
    assert cost == 0.0 and [(o.a, o.b) for o in ops] == [(0, 0), (1, 1)]
    cost, ops = align(["a"], ["a", "i"])
    assert cost == 1.0 and (None, 1) in [(o.a, o.b) for o in ops]


# --- providers --------------------------------------------------------------------------------


@pytest.mark.parametrize("provider", [Orthography2ipaProvider(), TugaphoneProvider()])
def test_providers_give_the_same_words_in_both_lects(provider):
    text = "Mais tarde vamos dormir."
    words = {lect: provider.candidates(text, lect) for lect in LECTS}
    assert [w.word for w in words["pt-PT"]] == [w.word for w in words["pt-BR"]] == ["mais", "tarde", "vamos", "dormir"]
    for w in words["pt-PT"] + words["pt-BR"]:
        assert w.readings and all(w.readings)


def test_optional_processes_reach_the_candidate_set():
    # European word-final reduced e is deletable: the sentence and isolated readings
    # of "que" before "o" differ, and both are licensed
    e = expect("Ele disse que o Rio de Janeiro é bonito", LECTS)
    que = [s for s in e.sites if s.word == "que"]
    assert que and () in que[0].variants["pt-PT"] and ("ɨ",) in que[0].variants["pt-PT"]


def test_free_variants_come_from_the_spec():
    words = Orthography2ipaProvider().candidates("pedido", "pt-PT")
    assert {"ɨ", "ə"} <= words[0].variants["ɨ"]
    assert Orthography2ipaProvider(free_variants=False).candidates("pedido", "pt-PT")[0].variants == {}


@pytest.mark.parametrize("provider", [Orthography2ipaProvider(), TugaphoneProvider()])
def test_provider_inventories_hold_the_site_class_phones(provider):
    eu, br = provider.inventory("pt-PT"), provider.inventory("pt-BR")
    assert {"ʃ", "ɨ", "u", "ɫ", "t"} <= eu
    assert {"s", "i", "o", "w", "tʃ", "dʒ"} <= br


class _StubProvider(PhoneProvider):
    """A provider reading fixed transcriptions, to drive site finding directly."""

    name = "stub"

    def __init__(self, table):
        self.table = table

    def candidates(self, text, lect):
        return [
            WordPhones(word, tuple(tuple(segment(r)) for r in readings), {})
            for word, readings in zip(text.split(), self.table[lect])
        ]

    def inventory(self, lect):
        return frozenset(canonical(s) for rs in self.table[lect] for r in rs for s in segment(r))


def test_site_finding_depends_only_on_the_provider_interface():
    stub = _StubProvider({"pt-PT": [["maʃ"], ["paɫ"]], "pt-BR": [["mas"], ["paw"]]})
    sites = discriminative_sites("mas pal", LECTS, stub)
    assert [(s.word, s.cls) for s in sites] == [("mas", "coda_s"), ("pal", "coda_l")]
    assert sites[0].variants == {"pt-PT": frozenset({("ʃ",)}), "pt-BR": frozenset({("s",)})}


def test_free_variants_enter_the_candidate_set():
    class _Variants(_StubProvider):
        def candidates(self, text, lect):
            words = super().candidates(text, lect)
            return [WordPhones(w.word, w.readings, {"ɨ": frozenset({"ɨ", "ə"})}) for w in words]

    e = expect("pede", LECTS, _Variants({"pt-PT": [["pɛdɨ"]], "pt-BR": [["pɛdi"]]}))
    assert e.sites[0].variants["pt-PT"] == frozenset({("ɨ",), ("ə",)})
    assert score_phones(["p", "ɛ", "d", "ə"], e).readings[0].lect == "pt-PT"


def test_overlapping_candidate_sets_are_not_sites():
    stub = _StubProvider({"pt-PT": [["maʃ", "mas"]], "pt-BR": [["mas"]]})
    e = expect("mas", LECTS, stub)
    assert e.sites == () and e.excluded == {"overlapping": 1}


def test_fused_differences_are_not_sites():
    stub = _StubProvider({"pt-PT": [["taɾdɨ"]], "pt-BR": [["taɾd͡ʒi"]]})
    e = expect("tarde", LECTS, stub)
    assert e.sites == () and e.excluded == {"fused": 1}


def test_differences_outside_the_documented_classes_are_not_sites():
    stub = _StubProvider({"pt-PT": [["o"], ["kaɾu"]], "pt-BR": [["u"], ["kaxu"]]})
    e = expect("o carro", LECTS, stub)
    assert e.sites == () and e.excluded == {"unclassified": 2}


# --- site classes -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, word, cls",
    [
        ("Mais tarde vamos dormir.", "mais", "coda_s"),
        ("O papel é azul.", "azul", "coda_l"),
        ("Os meninos comem pão.", "meninos", "unstressed_e"),
        ("Tive de partir cedo.", "partir", "td_before_i"),
    ],
)
def test_known_sites_are_found_in_their_class(text, word, cls):
    assert (word, cls) in [(s.word, s.cls) for s in discriminative_sites(text, LECTS)]


@pytest.mark.parametrize("provider", ["orthography2ipa", "tugaphone"])
def test_every_site_is_in_a_documented_class_and_level(provider):
    text = "Os amigos chegaram tarde ao hotel, mas a tia disse que o pedido era sempre igual."
    e = expect(text, LECTS, provider)
    assert e.sites
    for site in e.sites:
        assert site.cls in CLASSES
        assert site.level == {c.name: c.level for c in SITE_CLASSES}[site.cls]
        assert not site.variants["pt-PT"] & site.variants["pt-BR"]
        for lect in LECTS:
            start, end = site.spans[lect]
            assert tuple(canonical(s) for s in e.phones[lect][start:end]) in site.variants[lect]


def test_classify_reads_the_european_side_first():
    assert classify("ʃ", "s", None).name == "coda_s"
    assert classify("s", "ʃ", None) is None
    assert classify("t", "t͡ʃ", "i").name == "td_before_i"
    assert classify("t", "t͡ʃ", "a") is None
    assert classify("", "i", None).name == "unstressed_e"


def test_the_lect_pair_must_be_european_against_brazilian():
    assert roles(("pt-BR-x-sp", "pt-PT-x-porto")) == ("pt-PT-x-porto", "pt-BR-x-sp")
    with pytest.raises(ValueError):
        expect("Pá.", ("pt-PT", "pt-AO"))
    with pytest.raises(ValueError):
        expect("Pá.", ("pt-PT",))


def test_a_text_the_lects_say_alike_has_no_site():
    assert discriminative_sites("Lá vai a vaca.", LECTS) == ()


# --- scoring ----------------------------------------------------------------------------------

TEXT = "Os meninos de Lisboa partiram mais cedo para o hospital."


@pytest.mark.parametrize("lect", LECTS)
def test_the_expected_string_of_a_lect_reads_all_its_sites(lect):
    e = expect(TEXT, LECTS)
    result = score_phones(e.phones[lect], e)
    assert result.n_decided == result.n_sites == len(e.sites) >= 4
    assert result.fidelity(lect) == 100.0
    assert result.shares == {lect: 100.0, LECTS[1 - LECTS.index(lect)]: 0.0, NEITHER: 0.0}


def test_one_site_in_the_other_lect_moves_one_share():
    e = expect(TEXT, LECTS)
    mixed = list(e.phones["pt-PT"])
    site = next(s for s in e.sites if s.cls == "coda_s")
    start, end = site.spans["pt-PT"]
    mixed[start:end] = ["s"]
    result = score_phones(mixed, e)
    n = len(e.sites)
    assert result.shares["pt-BR"] == pytest.approx(100.0 / n)
    assert result.readings[site.index].lect == "pt-BR"


@pytest.mark.parametrize("silence", [[], ["a"], ["a", "b", "c"]])
def test_silence_or_too_little_speech_returns_none(silence):
    e = expect(TEXT, LECTS)
    assert len(silence) < MIN_PHONES
    assert score_phones(silence, e) is None


def test_a_string_short_of_the_expected_coverage_returns_none():
    e = expect(TEXT, LECTS)
    expected = min(len(p) for p in e.phones.values())
    enough = math.ceil(MIN_COVERAGE * expected)
    assert enough - 1 >= MIN_PHONES
    assert score_phones(e.phones["pt-PT"][: enough - 1], e) is None
    assert score_phones(e.phones["pt-PT"][:enough], e) is not None


def test_far_more_speech_than_the_text_returns_none():
    e = expect(TEXT, LECTS)
    expected = min(len(p) for p in e.phones.values())
    assert score_phones(e.phones["pt-PT"] * 240, e) is None
    longest = int(MAX_COVERAGE * expected)
    padded = list(e.phones["pt-PT"]) + ["a"] * (longest - len(e.phones["pt-PT"]))
    assert score_phones(padded, e) is not None
    assert score_phones(padded + ["a"], e) is None


def test_silent_audio_returns_none_through_the_scorer():
    scorer = LectFidelity(LECTS, backend=_FixedBackend([]), calibration=None)
    assert scorer.score(np.zeros(48000, dtype=np.float32), TEXT, sr=16000) is None


def test_a_deletion_site_counts_only_when_its_neighbours_were_realised():
    stub = _StubProvider({"pt-PT": [["kɨ", "k"], ["o"], ["kaza"]], "pt-BR": [["ki"], ["o"], ["kaza"]]})
    e = expect("que o casa", LECTS, stub)
    assert [s.cls for s in e.sites] == ["unstressed_e"] and () in e.sites[0].variants["pt-PT"]
    heard = score_phones(["k", "o", "k", "a", "z", "a"], e)
    assert heard.readings[0].lect == "pt-PT"
    lost = score_phones(["o", "k", "a", "z", "a"], e)
    assert lost.readings[0].lect is None


def test_shares_report_and_levels_are_consistent():
    e = expect(TEXT, LECTS)
    result = score_phones(e.phones["pt-BR"], e)
    assert sum(result.shares.values()) == pytest.approx(100.0)
    levels = result.shares_by_level
    assert set(levels) == {PRE_LEXICAL, POST_LEXICAL}
    for level, shares in levels.items():
        if any(r.site.level == level for r in result.readings):
            assert shares["pt-BR"] == 100.0
    rows = result.report()
    assert len(rows) == result.n_sites
    assert {"class", "level", "word", "context", "expected.pt-PT", "expected.pt-BR", "realised", "reading"} <= set(rows[0])


# --- calibration and pooling ------------------------------------------------------------------


def _readings(e, lect, n):
    return [(lect, score_phones(e.phones[lect], e).readings) for _ in range(n)]


def test_fit_measures_each_class_and_gates_unusable_ones():
    e = expect(TEXT, LECTS)
    clips = _readings(e, "pt-PT", 20) + _readings(e, "pt-BR", 20)
    cal = fit(clips, LECTS, "orthography2ipa", "fixed")
    seen = {s.cls for s in e.sites}
    for name, c in cal.classes.items():
        if name in seen:
            assert c.usable and c.llr["pt-PT"] > 0 > c.llr["pt-BR"], name
        else:
            assert not c.usable
    pt = cal.log_likelihood_ratio(clips[0][1])
    br = cal.log_likelihood_ratio(clips[-1][1])
    assert pt > 0 > br
    assert cal.decide(pt) == "pt-PT" and cal.decide(br) == "pt-BR"


def test_a_class_read_alike_for_both_lects_carries_no_weight():
    e = expect(TEXT, LECTS)
    same = score_phones(e.phones["pt-BR"], e).readings
    cal = fit([("pt-PT", same)] * 20 + [("pt-BR", same)] * 20, LECTS, "orthography2ipa", "fixed")
    assert not any(c.usable for c in cal.classes.values())
    assert cal.log_likelihood_ratio(same) == 0.0


def test_a_class_whose_difference_is_not_significant_carries_no_weight():
    weak = fit_class({"pt-PT": {"pt-PT": 6, "pt-BR": 4}, "pt-BR": {"pt-PT": 4, "pt-BR": 6}}, LECTS)
    assert weak.llr["pt-PT"] > 0 > weak.llr["pt-BR"]
    assert not weak.usable
    strong = fit_class({"pt-PT": {"pt-PT": 60, "pt-BR": 40}, "pt-BR": {"pt-PT": 40, "pt-BR": 60}}, LECTS)
    assert strong.usable


def test_class_log_ratios_use_add_one_smoothing():
    c = fit_class({"pt-PT": {"pt-PT": 8, "pt-BR": 2}, "pt-BR": {"pt-PT": 1, "pt-BR": 9}}, LECTS)
    assert c.llr["pt-PT"] == pytest.approx(math.log(9 / 13) - math.log(2 / 13))
    assert c.llr["pt-BR"] == pytest.approx(math.log(3 / 13) - math.log(10 / 13))
    assert c.llr[NEITHER] == pytest.approx(0.0)


def test_the_shipped_classes_follow_from_their_counts():
    for key, cal in load_table().items():
        for name, c in cal.classes.items():
            refit = fit_class(c.counts, cal.lects)
            assert refit.usable == c.usable, (key, name)
            for label, value in c.llr.items():
                assert refit.llr[label] == pytest.approx(value, abs=1e-9), (key, name, label)


def _calibration(temperature=1.0, threshold=0.0):
    return Calibration(LECTS, "orthography2ipa", "fixed", {}, temperature, threshold)


def test_the_decision_point_is_the_calibrated_threshold():
    cal = _calibration(threshold=1.0)
    assert cal.decide(0.5) == "pt-BR" and cal.decide(1.5) == "pt-PT"
    assert cal.decide(1.5, n_clips=2) == "pt-BR" and cal.decide(2.5, n_clips=2) == "pt-PT"
    assert cal.posterior(1.0) == pytest.approx(0.5)


def test_the_temperature_scales_the_posterior():
    assert _calibration(temperature=1.0).posterior(1.0) == pytest.approx(1 / (1 + math.exp(-1.0)))
    assert _calibration(temperature=2.0).posterior(1.0) == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert _calibration(temperature=0.5, threshold=1.0).posterior(3.0, n_clips=2) == pytest.approx(
        1 / (1 + math.exp(-0.5))
    )


def test_both_lect_orders_reach_the_same_verdict():
    swapped = LECTS[::-1]
    for provider in ("orthography2ipa", "tugaphone"):
        for backend in ("allosaurus", "wav2vec2_espeak"):
            forward, backward = shipped(LECTS, provider, backend), shipped(swapped, provider, backend)
            assert backward is not None and backward.lects == swapped
            for llr in (-2.0, forward.threshold - 1e-3, forward.threshold + 1e-3, 2.0):
                assert forward.decide(llr) == backward.decide(-llr)
                assert forward.posterior(llr) == pytest.approx(1.0 - backward.posterior(-llr))
    scorer = LectFidelity(swapped)
    assert scorer.calibration.lects == swapped and scorer.restrict_inventory
    forward = shipped(LECTS, "orthography2ipa", "allosaurus")
    e_forward, e_backward = expect(TEXT, LECTS), expect(TEXT, swapped)
    mixed = list(e_forward.phones["pt-PT"])
    site = next(s for s in e_forward.sites if s.cls == "coda_s")
    start, end = site.spans["pt-PT"]
    mixed[start:end] = ["s"]
    for realised in (e_forward.phones["pt-PT"], e_forward.phones["pt-BR"], mixed):
        a = score_phones(realised, e_forward, forward, duration=3.0)
        b = score_phones(realised, e_backward, scorer.calibration, duration=3.0)
        assert a.decision == b.decision
        assert a.log_likelihood_ratio == pytest.approx(-b.log_likelihood_ratio)
        pa, pb = pool([a] * 4, forward), pool([b] * 4, scorer.calibration)
        assert pa.decision == pb.decision
        assert pa.posterior == pytest.approx(pb.posterior)


def test_pool_states_duration_and_posterior():
    e = expect(TEXT, LECTS)
    cal = fit(_readings(e, "pt-PT", 10) + _readings(e, "pt-BR", 10), LECTS, "orthography2ipa", "fixed")
    clips = [score_phones(e.phones["pt-PT"], e, cal, duration=4.0) for _ in range(8)]
    pooled = pool(clips, cal)
    assert pooled.n_clips == 8 and pooled.duration == 32.0
    assert pooled.decision == "pt-PT" and pooled.posterior["pt-PT"] > 0.5
    assert pooled.n_sites == 8 * len(e.sites)


def test_the_shipped_table_covers_both_providers_and_states_its_provenance():
    table = load_table()
    for provider in ("orthography2ipa", "tugaphone"):
        for backend in ("allosaurus", "wav2vec2_espeak"):
            cal = shipped(LECTS, provider, backend)
            assert cal is not None, (provider, backend)
            assert {"calibration_split", "inventory", "packages", "corpora"} <= set(cal.provenance)
            for c in cal.classes.values():
                assert set(c.counts) == set(LECTS)
    assert all(k.count("|") == 3 for k in table)


def test_the_shipped_calibration_is_disjoint_from_the_evaluation():
    table = load_table()
    assert len(table) == 4
    for cal in table.values():
        corpora = cal.provenance["corpora"]
        assert corpora["calibration"] and corpora["evaluation"]
        assert set(corpora["calibration"]).isdisjoint(corpora["evaluation"])


# --- backend unit tables ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", [AllosaurusBackend, Wav2Vec2EspeakBackend])
def test_every_backend_unit_matches_the_golden_map(backend):
    golden = UNITS[backend.name]
    inventory = inv.ALLOSAURUS_INVENTORY if backend is AllosaurusBackend else inv.ESPEAK_INVENTORY
    assert list(golden) == list(inventory)
    assert {unit: backend.table.get(unit) for unit in inventory} == golden


def test_backend_tables_follow_the_inventories():
    assert AllosaurusBackend.symbols[0] is None and len(AllosaurusBackend.symbols) == 230
    assert Wav2Vec2EspeakBackend.symbols[0] is None and len(Wav2Vec2EspeakBackend.symbols) == 392


def test_espeak_ascii_mnemonics_are_read_by_scriptconv():
    assert notation.espeak_to_ipa("S") == "ʃ"
    assert notation.espeak_to_ipa("dZ") == "d͡ʒ"
    assert notation.espeak_to_ipa("t[") == "t̪"
    assert notation.espeak_to_ipa("u:") == "uː"


def test_metric_decisions_are_only_for_units_without_an_exact_reading():
    assert not set(notation.ESPEAK_DECISIONS) & set(notation.ESPEAK_SPELLING)
    assert notation.ESPEAK_DECISIONS["ᵻ"] == "ɨ" and notation.ESPEAK_DECISIONS["iɜ"] == "i"


@pytest.mark.parametrize("backend", [AllosaurusBackend, Wav2Vec2EspeakBackend])
@pytest.mark.parametrize("provider", [Orthography2ipaProvider(), TugaphoneProvider()])
def test_the_backend_covers_both_lects_phones(backend, provider):
    emitted = {canonical(s) for ipa in backend.table.values() for s in segment(ipa)}
    union = provider.inventory("pt-PT") | provider.inventory("pt-BR")
    gaps = notation.GAPS[backend.name]
    assert union - emitted <= gaps, sorted(union - emitted - gaps)
    for cls in SITE_CLASSES:
        assert {p for p in cls.european | cls.brazilian if p} <= emitted, cls.name


# --- backends: frontends and the whole path on tiny graphs ------------------------------------


def test_ctc_greedy_merges_repeats_and_drops_blanks():
    logits = np.eye(4)[[0, 2, 2, 0, 2, 3, 3, 1]][None]
    assert ctc_greedy(logits) == [2, 2, 3, 1]


def test_allosaurus_features_match_allosaurus():
    ref = np.load(FIXTURES / "allosaurus_features_reference.npz")
    feats = allosaurus_features(ref["pcm"].astype(np.float32) / 32768.0)
    assert feats.shape == ref["feats"].shape
    assert np.abs(feats - ref["feats"]).max() < 1e-3


def _tiny_weights(name):
    rng = np.random.default_rng(20261009)
    if name == "allosaurus":
        return rng.standard_normal((120, 230)).astype(np.float32)
    return rng.standard_normal((392, 1, 80)).astype(np.float32)


def test_allosaurus_backend_path_on_a_tiny_graph():
    backend = AllosaurusBackend(model=str(FIXTURES / "tiny_allosaurus.onnx"))
    clip = str(CLIPS / "pt-PT_0.wav")
    x, sr = load_audio(clip)
    feats = allosaurus_features(to_pcm16(kaiser_resample(x, sr, 8000)))
    units = ctc_greedy((feats @ _tiny_weights("allosaurus"))[None])
    assert units and backend.raw(clip) == [AllosaurusBackend.symbols[u] for u in units]
    want = segment("".join(AllosaurusBackend.table.get(AllosaurusBackend.symbols[u], "") + " " for u in units))
    assert backend.phones(clip) == want


def test_wav2vec2_backend_path_on_a_tiny_graph():
    backend = Wav2Vec2EspeakBackend(model=str(FIXTURES / "tiny_wav2vec2.onnx"))
    clip = str(CLIPS / "pt-BR_0.wav")
    x, sr = load_audio(clip)
    x = kaiser_resample(x, sr, 16000) if sr != 16000 else x
    x = ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)
    w = _tiny_weights("wav2vec2")[:, 0, :]
    frames = np.lib.stride_tricks.sliding_window_view(x, 80)[::320]
    units = ctc_greedy((frames @ w.T)[None])
    assert units and backend.raw(clip) == [Wav2Vec2EspeakBackend.symbols[u] for u in units]


def test_inventory_restriction_decodes_only_the_lects_phones():
    backend = AllosaurusBackend(model=str(FIXTURES / "tiny_allosaurus.onnx"))
    clip = str(CLIPS / "pt-PT_0.wav")
    inventory = frozenset({"a", "i", "u", "s", "t", "k"})
    phones = backend.phones(clip, inventory=inventory)
    assert phones and set(map(canonical, phones)) <= inventory


def test_construction_loads_no_model_and_allosaurus_is_the_default():
    scorer = LectFidelity(LECTS)
    assert isinstance(scorer.backend, AllosaurusBackend) and scorer.backend._session is None
    assert scorer.backend.model_info["repo_id"] == "TigreGotico/allosaurus-onnx"
    assert scorer.provider.name == "orthography2ipa"
    assert scorer.calibration is not None and scorer.calibration.backend == "allosaurus"


class _FixedBackend(PhoneBackend):
    """A backend that realises a fixed phone string and counts its calls."""

    name = "fixed"

    def __init__(self, phones):
        self._phones = phones
        self.calls = 0

    def logits(self, audio, sr=None):
        self.calls += 1
        return np.zeros((1, 1, 1), dtype=np.float32)

    def phones_from_logits(self, logits, inventory=None):
        return list(self._phones)

    def _frontend(self, audio, sr):  # pragma: no cover - never reached
        raise AssertionError

    def _postprocess(self, outputs):  # pragma: no cover - never reached
        raise AssertionError


def test_scorer_uses_the_backend_and_returns_none_without_sites():
    e = expect(TEXT, LECTS)
    backend = _FixedBackend(e.phones["pt-BR"])
    scorer = LectFidelity(LECTS, backend=backend, calibration=None)
    audio = np.zeros(16000, dtype=np.float32)
    assert scorer.score(audio, "Lá vai a vaca.", sr=16000) is None
    assert backend.calls == 0
    result = scorer.score(audio, TEXT, sr=16000)
    assert backend.calls == 1 and result.fidelity("pt-BR") == 100.0 and result.duration == 1.0


def test_unknown_backend_or_provider_is_refused():
    with pytest.raises(KeyError):
        LectFidelity(LECTS, backend="nope")
    with pytest.raises(KeyError):
        LectFidelity(LECTS, provider="nope")


# --- model tests: parity with the reference recognisers ----------------------------------------
#
# The Allosaurus export is 44 MB and is fetched from its pinned Hub revision, so these run
# in CI. The wav2vec2 export is 1.26 GB; its tests run only when
# ``SPEECHONNXMETRICS_LECT_MODELS`` names a directory holding it.

MODEL_DIR = os.environ.get("SPEECHONNXMETRICS_LECT_MODELS", "")
ESPEAK_ONNX = Path(MODEL_DIR) / "wav2vec2_xlsr53_espeak_cv_ft.onnx"
needs_wav2vec2 = pytest.mark.skipif(
    not (MODEL_DIR and ESPEAK_ONNX.is_file()),
    reason="SPEECHONNXMETRICS_LECT_MODELS does not name a directory holding the wav2vec2 export",
)


def _allosaurus() -> AllosaurusBackend:
    local = Path(MODEL_DIR) / "allosaurus_uni2005.onnx"
    return AllosaurusBackend(model=str(local)) if MODEL_DIR and local.is_file() else AllosaurusBackend()


def _levenshtein(a, b) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


@pytest.mark.models
def test_allosaurus_onnx_matches_allosaurus_through_the_default_path():
    # the reference phones are Allosaurus's own on each clip resampled to 8 kHz by this
    # package and stored as 16-bit PCM; the backend is called the way users call it
    reference = json.loads((FIXTURES / "allosaurus_reference.json").read_text())
    backend = _allosaurus()
    errors = total = 0
    for name, want in reference.items():
        errors += _levenshtein(backend.raw(str(CLIPS / name)), want)
        total += len(want)
    assert errors <= 2, f"{errors} of {total} phones differ from Allosaurus"


@pytest.mark.models
@needs_wav2vec2
def test_wav2vec2_onnx_matches_the_torch_model():
    reference = json.loads((FIXTURES / "wav2vec2_espeak_reference.json").read_text())
    backend = Wav2Vec2EspeakBackend(model=str(ESPEAK_ONNX))
    for name, want in reference.items():
        assert backend.raw(str(CLIPS / name)) == want, name


@pytest.mark.models
def test_the_shipped_calibration_ranks_the_synthesised_voices():
    scorer = LectFidelity(LECTS, backend=_allosaurus())
    pooled = {}
    for lect in LECTS:
        results = [scorer.score(str(CLIPS / c["file"]), c["text"]) for c in MANIFEST if c["lect"] == lect]
        pooled[lect] = scorer.pool(results)
    assert pooled["pt-PT"].log_likelihood_ratio > pooled["pt-BR"].log_likelihood_ratio


@pytest.mark.models
def test_a_clip_tiled_far_past_its_text_gets_no_verdict():
    clip = next(c for c in MANIFEST if c["lect"] == "pt-PT")
    x, sr = load_audio(str(CLIPS / clip["file"]))
    scorer = LectFidelity(LECTS, backend=_allosaurus())
    assert scorer.score(x, clip["text"], sr=sr) is not None
    tiled = np.tile(x, int(np.ceil(600 * sr / len(x))))
    assert scorer.score(tiled, clip["text"], sr=sr) is None
