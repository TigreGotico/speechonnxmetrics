"""Tests for the no-reference MOS metrics.

Split in two: everything structural runs offline against a stub inference session, so
``pytest test/`` is green with no network and no model cache. The parity tests — the
ones that actually prove the frontends are right — are marked ``models`` and skip only
when the weights are genuinely unobtainable here (not cached and no HuggingFace token).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from speechonnxmetrics import api, registry
from speechonnxmetrics._dsp.audio import AudioLoadError
from speechonnxmetrics.mos import DNSMOS, NISQA, SIGMOS, UTMOS, DNSMOSP808
from speechonnxmetrics.mos.dnsmos import WINDOW_SAMPLES, segment
from speechonnxmetrics.mos.nisqa import DIMENSIONS as NISQA_DIMENSIONS
from speechonnxmetrics.mos.nisqa import melspec
from speechonnxmetrics.mos.nisqa import segment as nisqa_segment
from speechonnxmetrics.mos.sigmos import DIMENSIONS, features
from speechonnxmetrics.mos.utmos import MIN_SAMPLES

FIXTURES = Path(__file__).parent / "fixtures"
AUDIO = FIXTURES / "audio"
DEMO = Path("/home/miro/AgentWorkspaces/ml/voiceclonnx/demo")

#: Achieved parity against each reference implementation, with headroom.
#: DNSMOS/UTMOS are exact to float32; SIGMOS is looser because the reference computes
#: its STFT in float32 and then raises it to the power -0.35, which amplifies rounding
#: error in near-silent bins by three orders of magnitude. Feeding our session the
#: reference's own features reproduces its scores to 2.2e-4, so the residual is
#: reference-side precision, not a frontend difference.
TOL = {"dnsmos": 1e-5, "dnsmos_p808": 1e-5, "sigmos": 2e-3, "utmos": 1e-4}

#: NISQA agrees with the torch reference to 2.6e-06 through our frontend; 1e-4 leaves
#: room for float32 ordering differences without hiding a frontend parameter error.
NISQA_TOL = 1e-4


def _weights_available() -> bool:
    """Whether the ONNX weights can actually be obtained in this environment."""
    from huggingface_hub import constants, get_token

    if get_token():
        return True
    from speechonnxmetrics.resolver import get_cache_dir

    hf_cache = Path(get_cache_dir()) / "hf"
    if any(hf_cache.glob("**/*.onnx")):
        return True
    shared = Path(constants.HF_HUB_CACHE)
    return any(
        any(shared.glob(f"**/{repo}-onnx/**/*.onnx"))
        for repo in ("dnsmos", "sigmos", "utmos", "nisqa")
    )


needs_weights = pytest.mark.skipif(
    not _weights_available(),
    reason="MOS weights are neither cached nor downloadable here (no HF token)",
)


class StubSession:
    """Minimal ``InferenceSession`` stand-in: records the feed, returns fixed outputs."""

    def __init__(self, outputs: list[np.ndarray], input_names: tuple[str, ...] = ("input",)) -> None:
        self._outputs = outputs
        self._input_names = input_names
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, _out_names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feed)
        return self._outputs

    def get_inputs(self) -> list[object]:
        return [type("Inp", (), {"name": n})() for n in self._input_names]


def _stub(metric, outputs, input_names=("input",)):
    session = StubSession(outputs, input_names)
    metric._session = session
    return session


# ------------------------------------------------------------------ #
# revision pinning — every downloaded model must be pinned to an immutable
# commit SHA, never a mutable branch/tag: an upstream update to a model repo
# must never silently change scores already reported. No network access.
# ------------------------------------------------------------------ #

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@pytest.mark.parametrize(
    "metric_cls,kwargs",
    [
        (DNSMOS, {}),
        (DNSMOS, {"personalized": True}),
        (DNSMOSP808, {}),
        (SIGMOS, {}),
        (UTMOS, {}),
        (NISQA, {}),
    ],
)
def test_mos_model_entry_is_pinned_to_a_commit_sha(metric_cls, kwargs):
    metric = metric_cls(**kwargs)
    revision = metric.model.revision
    assert revision is not None, f"{metric.name} has no pinned revision"
    assert _SHA_RE.match(revision), f"{metric.name} revision {revision!r} is not a 40-hex-char commit SHA"


def test_registry_mos_metrics_are_all_pinned():
    for entry in registry.list_metrics(requires_download=True):
        model = entry.fn.model  # each registered MOS metric is an OnnxMetric instance
        revision = model.revision
        assert revision is not None, f"{entry.name} has no pinned revision"
        assert _SHA_RE.match(revision), f"{entry.name} revision {revision!r} is not a 40-hex-char commit SHA"


def test_model_info_exposes_provenance():
    metric = UTMOS()
    info = metric.model_info
    assert info == {
        "repo_id": "TigreGotico/utmos-onnx",
        "filename": "utmos22_strong.onnx",
        "revision": metric.model.revision,
    }
    assert _SHA_RE.match(info["revision"])


# ------------------------------------------------------------------ #
# registration and construction
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("name", ["dnsmos", "dnsmos_p808", "sigmos", "utmos", "nisqa"])
def test_registered_as_download_backed_audio_metrics(name: str) -> None:
    entry = registry.get(name)
    assert entry.kind == "audio"
    assert entry.requires_download
    assert not entry.intrusive
    assert entry.range == (1.0, 5.0)
    assert entry.higher_is_better


def test_construction_creates_no_session() -> None:
    for metric in (DNSMOS(), DNSMOSP808(), SIGMOS(), UTMOS()):
        assert metric._session is None


def test_session_is_created_once_and_lazily() -> None:
    metric = UTMOS()
    session = _stub(metric, [np.array([4.0], dtype=np.float32)], ("wave",))
    metric(np.zeros(MIN_SAMPLES, dtype=np.float32), 16000)
    metric(np.zeros(MIN_SAMPLES, dtype=np.float32), 16000)
    assert metric.session is session
    assert len(session.feeds) == 2


def test_personalized_dnsmos_uses_its_own_weights_and_name() -> None:
    plain, personalized = DNSMOS(), DNSMOS(personalized=True)
    assert plain.name == "dnsmos" and personalized.name == "dnsmos_personalized"
    assert plain.model.hf_file != personalized.model.hf_file


# ------------------------------------------------------------------ #
# DNSMOS segmentation
# ------------------------------------------------------------------ #
def test_segment_pads_short_audio_by_tiling() -> None:
    # 1000 samples double until they exceed one window, reaching 256000 samples, which
    # then yields floor(16) - 9.01 + 1 = 7 windows — the reference's behaviour exactly
    windows = segment(np.arange(1000, dtype=np.float32))
    assert windows.shape == (7, WINDOW_SAMPLES)
    # tiling repeats the signal, so the pattern recurs every 1000 samples
    assert np.array_equal(windows[0, :1000], windows[0, 1000:2000])


def test_segment_of_exactly_one_window() -> None:
    x = np.random.default_rng(0).normal(size=WINDOW_SAMPLES).astype(np.float32)
    windows = segment(x)
    assert windows.shape == (1, WINDOW_SAMPLES)
    assert np.array_equal(windows[0], x)


def test_segment_of_long_audio_hops_by_one_second_and_drops_the_tail() -> None:
    x = np.arange(20 * 16000, dtype=np.float32)
    windows = segment(x)
    assert windows.shape == (11, WINDOW_SAMPLES)  # floor(20) - 9.01 + 1
    assert windows[1][0] - windows[0][0] == 16000


def test_segment_of_a_single_sample_still_yields_full_windows() -> None:
    windows = segment(np.array([0.5], dtype=np.float32))
    assert windows.shape[1] == WINDOW_SAMPLES and windows.shape[0] >= 1
    assert np.all(windows == 0.5)


def test_segment_rejects_empty_audio() -> None:
    with pytest.raises(ValueError, match="empty audio"):
        segment(np.array([], dtype=np.float32))


def test_dnsmos_averages_the_polynomial_mapped_segment_scores() -> None:
    metric = DNSMOS()
    raw = np.array([[3.0, 4.0, 3.5], [3.4, 4.2, 3.1]], dtype=np.float32)
    _stub(metric, [raw])
    scores = metric(np.zeros(20 * 16000, dtype=np.float32), 16000)
    from speechonnxmetrics.mos.dnsmos import POLYFIT

    for i, key in enumerate(("sig", "bak", "ovrl")):
        assert scores[key] == pytest.approx(float(np.mean(np.polyval(POLYFIT[key], raw[:, i]))))


def test_dnsmos_polynomial_mapping_is_not_the_identity() -> None:
    """A missing score-fitting step would silently return the raw network scale."""
    metric = DNSMOS()
    raw = np.array([[3.0, 3.0, 3.0]], dtype=np.float32)
    _stub(metric, [raw])
    scores = metric(np.zeros(WINDOW_SAMPLES, dtype=np.float32), 16000)
    assert all(abs(v - 3.0) > 0.05 for v in scores.values())


def test_personalized_dnsmos_uses_different_polynomials() -> None:
    raw = np.array([[3.0, 4.0, 3.5]], dtype=np.float32)
    plain, personalized = DNSMOS(), DNSMOS(personalized=True)
    _stub(plain, [raw])
    _stub(personalized, [raw])
    audio = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    assert plain(audio, 16000) != personalized(audio, 16000)


def test_p808_frontend_shape_matches_the_model_input() -> None:
    metric = DNSMOSP808()
    session = _stub(metric, [np.array([[3.5], [3.7]], dtype=np.float32)])
    metric(np.zeros(11 * 16000, dtype=np.float32), 16000)
    feats = session.feeds[0]["input_1"]
    assert feats.shape == (2, 900, 120)
    assert feats.dtype == np.float32


def test_p808_averages_over_segments() -> None:
    metric = DNSMOSP808()
    _stub(metric, [np.array([[3.0], [4.0]], dtype=np.float32)])
    assert metric(np.zeros(11 * 16000, dtype=np.float32), 16000) == pytest.approx(3.5)


# ------------------------------------------------------------------ #
# SIGMOS frontend
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("n_samples", [1, 480, 481, 48000, 5 * 48000])
def test_sigmos_features_shape(n_samples: int) -> None:
    feats = features(np.zeros(n_samples, dtype=np.float32))
    assert feats.ndim == 4
    assert feats.shape[0] == 1 and feats.shape[1] == 3 and feats.shape[3] == 481
    assert feats.dtype == np.float32


def test_sigmos_features_frame_count_follows_the_hop() -> None:
    short = features(np.zeros(48000, dtype=np.float32)).shape[2]
    long = features(np.zeros(96000, dtype=np.float32)).shape[2]
    assert long - short == 48000 // 480


def test_sigmos_features_reject_empty_audio() -> None:
    with pytest.raises(ValueError, match="empty audio"):
        features(np.array([], dtype=np.float32))


def test_sigmos_returns_all_seven_dimensions_in_order() -> None:
    metric = SIGMOS()
    _stub(metric, [np.arange(7, dtype=np.float32).reshape(1, 7) + 1.0])
    scores = metric(np.zeros(48000, dtype=np.float32), 48000)
    assert list(scores) == list(DIMENSIONS)
    assert scores["col"] == 1.0 and scores["ovrl"] == 7.0


# ------------------------------------------------------------------ #
# UTMOS frontend
# ------------------------------------------------------------------ #
def test_utmos_feeds_a_batched_raw_waveform() -> None:
    metric = UTMOS()
    session = _stub(metric, [np.array([4.1], dtype=np.float32)], ("wave",))
    audio = np.linspace(-0.5, 0.5, 16000).astype(np.float32)
    assert metric(audio, 16000) == pytest.approx(4.1)
    wave = session.feeds[0]["wave"]
    assert wave.shape == (1, 16000) and wave.dtype == np.float32
    assert np.allclose(wave[0], audio)


@pytest.mark.parametrize("n_samples", [1, 100, MIN_SAMPLES - 1])
def test_utmos_pads_audio_below_the_convolutional_receptive_field(n_samples: int) -> None:
    metric = UTMOS()
    session = _stub(metric, [np.array([3.0], dtype=np.float32)], ("wave",))
    metric(np.full(n_samples, 0.1, dtype=np.float32), 16000)
    assert session.feeds[0]["wave"].shape == (1, MIN_SAMPLES)


def test_utmos_rejects_empty_audio() -> None:
    metric = UTMOS()
    _stub(metric, [np.array([3.0], dtype=np.float32)], ("wave",))
    with pytest.raises(ValueError, match="empty audio"):
        metric(np.array([], dtype=np.float32), 16000)


# ------------------------------------------------------------------ #
# adversarial: nothing may return NaN silently
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("bad", [np.array([np.nan, 0.1]), np.array([np.inf, 0.1])])
def test_non_finite_audio_raises_a_typed_error(bad: np.ndarray) -> None:
    for metric in (DNSMOS(), DNSMOSP808(), SIGMOS(), UTMOS()):
        with pytest.raises(AudioLoadError, match="NaN/inf"):
            metric(bad.astype(np.float32), metric.sample_rate)


@pytest.mark.parametrize(
    "metric_factory,outputs",
    [
        (DNSMOS, [np.array([[np.nan, 1.0, 1.0]], dtype=np.float32)]),
        (DNSMOSP808, [np.array([[np.nan]], dtype=np.float32)]),
        (SIGMOS, [np.full((1, 7), np.nan, dtype=np.float32)]),
        (UTMOS, [np.array([np.nan], dtype=np.float32)]),
    ],
)
def test_a_non_finite_model_output_raises_rather_than_returning_nan(metric_factory, outputs) -> None:
    metric = metric_factory()
    _stub(metric, outputs, ("wave",))
    with pytest.raises(ValueError, match="non-finite"):
        metric(np.zeros(48000, dtype=np.float32), metric.sample_rate)


def test_all_zero_audio_scores_without_nan() -> None:
    metric = DNSMOS()
    _stub(metric, [np.array([[3.0, 3.0, 3.0]], dtype=np.float32)])
    scores = metric(np.zeros(5 * 16000, dtype=np.float32), 16000)
    assert all(np.isfinite(v) for v in scores.values())


def test_wrong_sample_rate_is_resampled_not_rejected() -> None:
    metric = DNSMOS()
    session = _stub(metric, [np.array([[3.0, 3.0, 3.0]], dtype=np.float32)])
    metric(np.zeros(44100 * 12, dtype=np.float32), 44100)  # 12 s at a non-native rate
    feats = session.feeds[0]["input_1"]
    assert feats.shape == (3, WINDOW_SAMPLES)  # resampled to 16 kHz -> floor(12)-9.01+1


def test_stereo_audio_is_downmixed() -> None:
    metric = UTMOS()
    session = _stub(metric, [np.array([3.0], dtype=np.float32)], ("wave",))
    stereo = np.stack([np.full(16000, 0.2), np.full(16000, 0.4)], axis=1).astype(np.float32)
    metric(stereo, 16000)
    wave = session.feeds[0]["wave"]
    assert wave.shape == (1, 16000)
    assert np.allclose(wave[0], 0.3)


def test_empty_audio_is_rejected_by_every_metric() -> None:
    empty = np.array([], dtype=np.float32)
    for metric in (DNSMOS(), DNSMOSP808(), SIGMOS(), UTMOS()):
        _stub(metric, [np.zeros((1, 7), dtype=np.float32)], ("wave",))
        with pytest.raises(ValueError):
            metric(empty, metric.sample_rate)


# ------------------------------------------------------------------ #
# dict flattening through the public score() machinery
# ------------------------------------------------------------------ #
def test_dict_returning_metrics_flatten_through_score(monkeypatch: pytest.MonkeyPatch) -> None:
    metric = registry.get("sigmos").fn
    _stub(metric, [np.arange(7, dtype=np.float32).reshape(1, 7) + 1.0])
    try:
        row = api.score(np.zeros(48000, dtype=np.float32), ["sigmos"], sr=48000)
    finally:
        metric._session = None
    assert set(row) == {f"sigmos.{d}" for d in DIMENSIONS}
    assert row["sigmos.col"] == 1.0


def test_scalar_metrics_do_not_flatten() -> None:
    metric = registry.get("utmos").fn
    _stub(metric, [np.array([4.2], dtype=np.float32)], ("wave",))
    try:
        row = api.score(np.zeros(16000, dtype=np.float32), ["utmos"], sr=16000)
    finally:
        metric._session = None
    assert row == {"utmos": pytest.approx(4.2)}


# ------------------------------------------------------------------ #
# parity against the reference implementations
# ------------------------------------------------------------------ #
def _clip_path(key: str) -> Path:
    return DEMO / Path(key).name if key.startswith("demo/") else Path(__file__).parent.parent / key


with open(FIXTURES / "mos_fixture.json") as _f:
    PARITY = json.load(_f)


@pytest.mark.models
@needs_weights
@pytest.mark.parametrize("key", sorted(PARITY))
def test_parity_against_reference_implementations(key: str) -> None:
    """Score real clips through our frontends and compare to the reference values.

    Expected values come from ``speechmos`` (DNSMOS), the released ``sigmos.py``
    estimator (SIGMOS) and ``torch.hub`` ``tarepan/SpeechMOS`` (UTMOS).
    """
    path = _clip_path(key)
    if not path.is_file():
        pytest.skip(f"clip not present in this checkout: {path}")
    expected = PARITY[key]

    dnsmos = DNSMOS(personalized=True)(str(path), None)
    for dim, want in expected["dnsmos_personalized"].items():
        assert dnsmos[dim] == pytest.approx(want, abs=TOL["dnsmos"])

    assert DNSMOSP808()(str(path), None) == pytest.approx(
        expected["dnsmos_p808"], abs=TOL["dnsmos_p808"]
    )
    assert UTMOS()(str(path), None) == pytest.approx(expected["utmos"], abs=TOL["utmos"])

    sigmos = SIGMOS()(str(path), None)
    for dim, want in expected["sigmos"].items():
        assert sigmos[dim] == pytest.approx(want, abs=TOL["sigmos"])


@pytest.mark.models
@needs_weights
def test_scores_stay_inside_the_mos_range_on_real_audio() -> None:
    clip = str(AUDIO / "source.wav")
    for metric in (DNSMOS(personalized=True), DNSMOSP808(), SIGMOS(), UTMOS()):
        value = metric(clip, None)
        values = list(value.values()) if isinstance(value, dict) else [value]
        assert all(1.0 <= v <= 5.0 for v in values), (metric.name, values)


# ------------------------------------------------------------------ #
# NISQA
# ------------------------------------------------------------------ #
def test_nisqa_is_rate_adaptive_rather_than_pinned_to_one_rate() -> None:
    assert NISQA().sample_rate is None


def test_nisqa_licence_records_the_noncommercial_weights() -> None:
    assert "NonCommercial" in NISQA().model.license


def test_nisqa_dimension_order_matches_the_reference_columns() -> None:
    """``NISQA_lib`` emits mos/noi/dis/col/loud; the class docstring's col/dis swap is
    a documentation bug and must not be copied here."""
    assert NISQA_DIMENSIONS == ("mos", "noi", "dis", "col", "loud")


@pytest.mark.parametrize(
    "n_frames,expected", [(15, 1), (18, 1), (19, 2), (23, 3), (100, 22), (1000, 247)]
)
def test_nisqa_segment_count_follows_ceil_of_the_hop(n_frames: int, expected: int) -> None:
    segments = nisqa_segment(np.zeros((48, n_frames), dtype=np.float64))
    assert segments.shape == (expected, 1, 48, 15)
    assert segments.dtype == np.float32


def test_nisqa_segments_are_consecutive_windows_four_frames_apart() -> None:
    spec = np.tile(np.arange(23, dtype=np.float64), (48, 1))
    segments = nisqa_segment(spec)
    assert np.array_equal(segments[0, 0, 0], np.arange(15))
    assert np.array_equal(segments[1, 0, 0], np.arange(4, 19))


def test_nisqa_segment_rejects_a_spectrogram_below_one_window() -> None:
    with pytest.raises(ValueError, match="too short"):
        nisqa_segment(np.zeros((48, 14), dtype=np.float64))


def test_nisqa_frontend_feeds_a_batched_segment_stack() -> None:
    metric = NISQA()
    session = _stub(metric, [np.full((1, 5), 4.0, dtype=np.float32)], ("segments",))
    metric(np.zeros(16000, dtype=np.float32), 16000)
    segments = session.feeds[0]["segments"]
    assert segments.ndim == 5 and segments.shape[2:] == (1, 48, 15)
    assert segments.shape[0] == 1 and segments.dtype == np.float32


def test_nisqa_returns_all_five_dimensions_in_order() -> None:
    metric = NISQA()
    _stub(metric, [np.arange(5, dtype=np.float32).reshape(1, 5) + 1.0], ("segments",))
    scores = metric(np.zeros(16000, dtype=np.float32), 16000)
    assert list(scores) == list(NISQA_DIMENSIONS)
    assert scores["mos"] == 1.0 and scores["loud"] == 5.0


def test_nisqa_flattens_through_score() -> None:
    metric = registry.get("nisqa").fn
    _stub(metric, [np.arange(5, dtype=np.float32).reshape(1, 5) + 1.0], ("segments",))
    try:
        row = api.score(np.zeros(16000, dtype=np.float32), ["nisqa"], sr=16000)
    finally:
        metric._session = None
    assert set(row) == {f"nisqa.{d}" for d in NISQA_DIMENSIONS}
    assert row["nisqa.mos"] == 1.0


def test_nisqa_hop_and_window_adapt_to_the_sample_rate_without_resampling() -> None:
    metric = NISQA()
    session = _stub(metric, [np.full((1, 5), 4.0, dtype=np.float32)], ("segments",))
    metric(np.zeros(44100, dtype=np.float32), 44100)  # one second at 44.1 kHz
    metric(np.zeros(16000, dtype=np.float32), 16000)  # one second at 16 kHz
    # a 10 ms hop at either rate means ~100 frames/s, hence the same segment count;
    # a silent resample to 16 kHz would still give this, so also check the frames
    counts = [feed["segments"].shape[1] for feed in session.feeds]
    assert counts[0] == counts[1]
    frames_44k = melspec(np.zeros(44100, dtype=np.float32), 44100).shape[1]
    assert frames_44k == 1 + 44100 // int(44100 * 0.01)


@pytest.mark.parametrize("n_samples", [1, 100, 2048, 2239])
def test_nisqa_rejects_audio_shorter_than_one_segment(n_samples: int) -> None:
    metric = NISQA()
    _stub(metric, [np.full((1, 5), 4.0, dtype=np.float32)], ("segments",))
    with pytest.raises(ValueError, match="too short"):
        metric(np.zeros(n_samples, dtype=np.float32), 16000)


def test_nisqa_scores_exactly_one_segment() -> None:
    metric = NISQA()
    session = _stub(metric, [np.full((1, 5), 4.0, dtype=np.float32)], ("segments",))
    # 15 frames at a 160-sample hop: 1 + n//160 == 15
    metric(np.zeros(14 * 160, dtype=np.float32), 16000)
    assert session.feeds[0]["segments"].shape[:2] == (1, 1)


def test_nisqa_scores_all_zero_audio_without_nan() -> None:
    spec = melspec(np.zeros(16000, dtype=np.float32), 16000)
    assert np.all(np.isfinite(spec))
    # amin=1e-4 floors the magnitude, so silence lands on a finite -80 dB plateau
    assert spec.max() == pytest.approx(-80.0)


def test_nisqa_rejects_empty_audio() -> None:
    with pytest.raises(ValueError, match="empty audio"):
        melspec(np.array([], dtype=np.float32), 16000)


@pytest.mark.parametrize("bad", [np.array([np.nan, 0.1]), np.array([np.inf, 0.1])])
def test_nisqa_rejects_non_finite_audio(bad: np.ndarray) -> None:
    with pytest.raises(AudioLoadError, match="NaN/inf"):
        NISQA()(bad.astype(np.float32), 16000)


def test_nisqa_non_finite_model_output_raises() -> None:
    metric = NISQA()
    _stub(metric, [np.full((1, 5), np.nan, dtype=np.float32)], ("segments",))
    with pytest.raises(ValueError, match="non-finite"):
        metric(np.zeros(16000, dtype=np.float32), 16000)


def test_nisqa_downmixes_stereo() -> None:
    metric = NISQA()
    session = _stub(metric, [np.full((1, 5), 4.0, dtype=np.float32)], ("segments",))
    stereo = np.stack([np.full(16000, 0.2), np.full(16000, 0.4)], axis=1).astype(np.float32)
    metric(stereo, 16000)
    mono = metric._frontend(np.full(16000, 0.3, dtype=np.float32), 16000)
    assert np.allclose(session.feeds[0]["segments"], mono["segments"])


with open(FIXTURES / "nisqa_fixture.json") as _f:
    NISQA_PARITY = json.load(_f)


def _nisqa_clip_path(key: str) -> Path:
    if not key.startswith("demo/"):
        return Path(__file__).parent.parent / key
    name = Path(key).name
    return DEMO / name if (DEMO / name).is_file() else DEMO / "outputs" / name


@pytest.mark.models
@needs_weights
@pytest.mark.parametrize("key", sorted(NISQA_PARITY))
def test_nisqa_parity_against_the_torch_reference(key: str) -> None:
    """Our numpy frontend vs upstream's librosa frontend, both into the NISQA model.

    Expected values are the torch ``NISQA_DIM`` forward pass over librosa-computed
    features. The clips span 10, 16 and 24 kHz, so the rate-adaptive hop/window are
    exercised too.
    """
    path = _nisqa_clip_path(key)
    if not path.is_file():
        pytest.skip(f"clip not present in this checkout: {path}")
    scores = NISQA()(str(path), None)
    for dim, want in NISQA_PARITY[key]["nisqa"].items():
        assert scores[dim] == pytest.approx(want, abs=NISQA_TOL), dim
