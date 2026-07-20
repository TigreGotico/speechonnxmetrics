import builtins
import sys

import numpy as np
import pytest

from speechonnxmetrics.speaker import (
    SpeakerSimilarity,
    eer,
    equal_error_threshold,
    min_dcf,
)

# --- EER / min_dcf on hand-computed score sets ----------------------------------------------


def test_eer_zero_on_perfectly_separable_scores():
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert eer(scores, labels) == pytest.approx(0.0)


def test_eer_half_on_fully_overlapping_distributions():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0])
    assert eer(scores, labels) == pytest.approx(0.5)


def test_equal_error_threshold_lies_at_separation_point():
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    labels = np.array([1, 1, 1, 0, 0, 0])
    threshold = equal_error_threshold(scores, labels)
    assert 0.3 <= threshold <= 0.7


def test_min_dcf_perfectly_separable_is_near_zero():
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert min_dcf(scores, labels, p_target=0.01) == pytest.approx(0.0, abs=1e-9)


def test_min_dcf_requires_p_target_in_open_unit_interval():
    scores = np.array([0.9, 0.1])
    labels = np.array([1, 0])
    with pytest.raises(ValueError):
        min_dcf(scores, labels, p_target=0.0)
    with pytest.raises(ValueError):
        min_dcf(scores, labels, p_target=1.0)


# --- symmetry / monotonicity properties -----------------------------------------------------


def test_eer_symmetry_under_score_and_label_negation():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=40)
    labels = rng.integers(0, 2, size=40)
    if labels.sum() in (0, 40):
        labels[0] = 1 - labels[0]
    direct = eer(scores, labels)
    flipped = eer(-scores, 1 - labels)
    assert direct == pytest.approx(flipped)


def test_eer_monotonicity_as_separation_increases():
    rng = np.random.default_rng(1)
    labels = np.array([1] * 25 + [0] * 25)

    def scores_with_gap(gap):
        genuine = rng.normal(loc=gap / 2, scale=1.0, size=25)
        impostor = rng.normal(loc=-gap / 2, scale=1.0, size=25)
        return np.concatenate([genuine, impostor])

    small_gap_eer = eer(scores_with_gap(0.5), labels)
    large_gap_eer = eer(scores_with_gap(6.0), labels)
    assert large_gap_eer <= small_gap_eer


# --- adversarial cases -----------------------------------------------------------------------


def test_eer_rejects_all_same_label():
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 1, 1])
    with pytest.raises(ValueError, match="genuine.*impostor|impostor.*genuine"):
        eer(scores, labels)


def test_eer_rejects_empty_arrays():
    with pytest.raises(ValueError, match="empty"):
        eer(np.array([]), np.array([]))


def test_eer_rejects_single_sample():
    with pytest.raises(ValueError, match="more than one"):
        eer(np.array([0.5]), np.array([1]))


def test_eer_rejects_nan_scores():
    with pytest.raises(ValueError, match="NaN|inf"):
        eer(np.array([0.1, np.nan, 0.3]), np.array([1, 0, 1]))


def test_eer_rejects_inf_scores():
    with pytest.raises(ValueError, match="NaN|inf"):
        eer(np.array([0.1, np.inf, 0.3]), np.array([1, 0, 1]))


def test_eer_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="binary"):
        eer(np.array([0.1, 0.5, 0.9]), np.array([0, 1, 2]))


def test_eer_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="shapes|length"):
        eer(np.array([0.1, 0.2, 0.3]), np.array([1, 0]))


def test_min_dcf_rejects_same_adversarial_inputs_as_eer():
    with pytest.raises(ValueError):
        min_dcf(np.array([0.1, 0.2]), np.array([1, 1]))


def test_equal_error_threshold_rejects_same_adversarial_inputs_as_eer():
    with pytest.raises(ValueError):
        equal_error_threshold(np.array([]), np.array([]))


# --- optional speakeronnx dependency: clear ImportError when missing -----------------------


def test_missing_speakeronnx_raises_actionable_import_error(monkeypatch):
    for mod in list(sys.modules):
        if mod == "speakeronnx" or mod.startswith("speakeronnx."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    for mod in list(sys.modules):
        if mod == "speechonnxmetrics.speaker" or mod.startswith("speechonnxmetrics.speaker."):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "speakeronnx" or name.startswith("speakeronnx."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"pip install speechonnxmetrics\[speaker\]"):
        import speechonnxmetrics.speaker  # noqa: F401


# --- construction is lazy: no network/disk touched until first use -------------------------


def test_speaker_similarity_construction_does_not_load_embedder():
    scorer = SpeakerSimilarity()
    assert "embedder" not in scorer.__dict__
