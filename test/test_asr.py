import json
import math
import unicodedata
from pathlib import Path

import pytest

from speechonnxmetrics.asr import (
    BASIC,
    STRICT,
    AsrMetrics,
    EmptyReferenceError,
    Normalizer,
    align,
    cer,
    collapse_whitespace,
    compute,
    expand_contractions,
    lowercase,
    mer,
    remove_diacritics,
    strip_filler_words,
    strip_punctuation,
    wer,
    wil,
    wip,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "jiwer_wer_fixture.json").read_text())


# --- 1. parity against the committed jiwer fixture -------------------------------------------

@pytest.mark.parametrize("case", FIXTURE.values(), ids=FIXTURE.keys())
def test_jiwer_parity(case):
    ref, hyp = case["ref"], case["hyp"]
    if case["wer"] == "ERROR:ValueError":
        with pytest.raises(EmptyReferenceError):
            wer(ref, hyp)
        return
    assert wer(ref, hyp) == pytest.approx(case["wer"], abs=1e-9)
    assert cer(ref, hyp) == pytest.approx(case["cer"], abs=1e-9)
    assert mer(ref, hyp) == pytest.approx(case["mer"], abs=1e-9)
    assert wil(ref, hyp) == pytest.approx(case["wil"], abs=1e-9)


# --- 2. corpus aggregation vs per-utterance averaging must differ --------------------------

def test_corpus_aggregation_not_average_of_per_utterance_rates():
    refs = ["a", "a b c d e f g h i j"]
    hyps = ["b", "a b c d e f g h i j"]  # utterance 1: 100% wrong (1/1); utterance 2: perfect (0/10)

    per_utterance_average = (wer(refs[0], hyps[0]) + wer(refs[1], hyps[1])) / 2
    assert per_utterance_average == pytest.approx(0.5)

    corpus_wer = wer(refs, hyps)
    # aggregate: 1 error / 11 reference words total
    assert corpus_wer == pytest.approx(1 / 11)
    assert corpus_wer != pytest.approx(per_utterance_average)


def test_corpus_aggregation_matches_manual_totals():
    refs = ["the cat sat", "a dog ran fast"]
    hyps = ["the cat sat", "a cat ran"]
    result = compute(refs, hyps)
    assert isinstance(result, AsrMetrics)
    assert result.ref_length == 7  # 3 + 4
    assert result.substitutions == 1  # dog -> cat
    assert result.deletions == 1  # fast dropped
    assert result.wer == pytest.approx(2 / 7)


# --- 3. adversarial cases ---------------------------------------------------------------------

def test_empty_reference_raises_typed_error():
    with pytest.raises(EmptyReferenceError):
        wer("", "hello")


def test_whitespace_only_reference_raises_typed_error():
    with pytest.raises(EmptyReferenceError):
        wer("   ", "hello")


def test_empty_hypothesis_is_total_deletion_not_an_error():
    assert wer("hello world", "") == pytest.approx(1.0)


def test_both_empty_raises_typed_error():
    with pytest.raises(EmptyReferenceError):
        wer("", "")


def test_none_reference_raises_type_error():
    with pytest.raises(TypeError):
        wer(None, "hello")


def test_none_hypothesis_raises_type_error():
    with pytest.raises(TypeError):
        wer("hello", None)


def test_str_list_type_mismatch_raises_type_error():
    with pytest.raises(TypeError):
        wer("hello world", ["hello world"])


def test_non_string_elements_in_corpus_list_raise_type_error():
    with pytest.raises(TypeError):
        wer(["hello", 42], ["hello", "world"])


def test_mismatched_corpus_lengths_raise_value_error():
    with pytest.raises(ValueError):
        wer(["a", "b"], ["a"])


def test_extremely_long_single_token():
    token = "a" * 100_000
    assert wer(token, token) == pytest.approx(0.0)
    assert wer(token, token + "b") == pytest.approx(1.0)


def test_unicode_normalization_form_mismatch_does_not_crash():
    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", "café")
    assert nfc != nfd  # distinct byte sequences, same rendered text
    rate = wer(nfc, nfd)
    assert math.isfinite(rate)
    assert 0.0 <= rate <= 1.0


# --- 4. every metric returns a finite float or raises ----------------------------------------

FINITE_CASES = [
    ("hello world", "hello world"),
    ("hello world", "goodbye moon entirely different"),
    ("a", "a a a a a a a a a a"),
    ("a a a a a a a a a a", "a"),
    ("répété répété", "repete repete"),
]


@pytest.mark.parametrize("ref,hyp", FINITE_CASES)
@pytest.mark.parametrize("metric", [wer, cer, mer, wil, wip])
def test_metrics_are_always_finite(metric, ref, hyp):
    value = metric(ref, hyp)
    assert math.isfinite(value)


def test_compute_result_fields_all_finite():
    result = compute("the quick brown fox", "the quick fox jumped")
    for value in (result.wer, result.mer, result.wil, result.wip):
        assert math.isfinite(value)
    assert result.wip == pytest.approx(1.0 - result.wil)


# --- align() primitive -------------------------------------------------------------------------

def test_align_equal_sequences():
    ops = align("a b c", "a b c")
    assert [op.tag for op in ops] == ["equal", "equal", "equal"]


def test_align_covers_full_span_and_counts_match_compute():
    ref, hyp = "the cat sat on the mat", "the dog sat on mat"
    ops = align(ref, hyp)
    result = compute(ref, hyp)
    counted = {"equal": 0, "sub": 0, "del": 0, "ins": 0}
    for op in ops:
        counted[op.tag] += 1
    assert counted["equal"] == result.hits
    assert counted["sub"] == result.substitutions
    assert counted["del"] == result.deletions
    assert counted["ins"] == result.insertions


def test_align_pure_insertion():
    ops = align("", "a b c")
    assert all(op.tag == "ins" for op in ops)
    assert len(ops) == 3


def test_align_pure_deletion():
    ops = align("a b c", "")
    assert all(op.tag == "del" for op in ops)
    assert len(ops) == 3


# --- normalizer tests --------------------------------------------------------------------------

def test_lowercase():
    assert lowercase("HELLO World") == "hello world"


def test_strip_punctuation():
    assert strip_punctuation("hello, world!") == "hello  world "


def test_collapse_whitespace():
    assert collapse_whitespace("hello   world\t\nfoo") == "hello world foo"


def test_remove_diacritics():
    assert remove_diacritics("café naïve résumé") == "cafe naive resume"


def test_expand_contractions():
    assert expand_contractions("I can't won't don't") == "I cannot will not do not"


def test_strip_filler_words():
    assert strip_filler_words("um so uh this is like the plan") == "so this is like the plan"


NORMALIZERS = [lowercase, strip_punctuation, collapse_whitespace, remove_diacritics, expand_contractions, strip_filler_words]


@pytest.mark.parametrize("fn", NORMALIZERS)
def test_normalizer_idempotency(fn):
    samples = ["Hello, World! café naïve", "  um uh this   is a test  ", "won't can't don't"]
    for text in samples:
        once = fn(text)
        twice = fn(once)
        assert once == twice


def test_normalizer_class_chains_steps_in_order():
    normalizer = Normalizer((lowercase, strip_punctuation, collapse_whitespace))
    assert normalizer("Hello,   World!") == "hello world"


def test_basic_preset():
    assert BASIC("  Hello   World  ") == "hello world"


def test_strict_preset_full_pipeline():
    assert STRICT("Um, café naïve — I can't believe it!") == "cafe naive i cannot believe it"


def test_normalizer_default_is_none_no_silent_normalization():
    # differing case with no normalizer must NOT be treated as equal
    assert wer("Hello World", "hello world") == pytest.approx(1.0)
    assert wer("Hello World", "hello world", normalizer=BASIC) == pytest.approx(0.0)
