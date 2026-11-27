"""Family contract: bytes input, and Hub pinning of the phone recognisers."""
import re
import wave

import numpy as np
import pytest

from speechonnxmetrics import resolver
from speechonnxmetrics._dsp.audio import load_audio
from speechonnxmetrics.lect_fidelity.backends import BACKENDS

SHA = re.compile(r"[0-9a-f]{40}")


def _write_wav(path, samples, sr):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.astype("<i2").tobytes())


def test_wav_bytes_read_with_header_rate(tmp_path):
    path = tmp_path / "tone.wav"
    t = np.arange(2205) / 22050
    _write_wav(path, (0.4 * np.sin(2 * np.pi * 440 * t) * 32767), 22050)
    with wave.open(str(path), "rb") as wf:
        oracle_sr = wf.getframerate()
        oracle = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2") / 32768.0
    x, sr = load_audio(path.read_bytes())
    assert sr == oracle_sr == 22050
    np.testing.assert_allclose(x, oracle, atol=1e-6)


def test_non_wav_bytes_without_rate_raise():
    pcm = np.array([0, 1000, -1000, 2000], dtype="<i2").tobytes()
    with pytest.raises(ValueError):
        load_audio(pcm)


def test_raw_pcm_bytes_with_rate():
    pcm = np.array([0, 16384, -16384], dtype="<i2").tobytes()
    x, sr = load_audio(pcm, sample_rate=16000)
    assert sr == 16000
    np.testing.assert_array_equal(x, np.array([0.0, 0.5, -0.5], dtype=np.float32))


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_phone_backend_pins_a_commit_sha(name):
    assert SHA.fullmatch(BACKENDS[name]().model.revision or "")


class _Resolved(Exception):
    pass


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_phone_backend_session_resolves_the_pinned_revision(name, monkeypatch):
    def fake_resolve(ref, **kw):
        raise _Resolved(kw["revision"])

    monkeypatch.setattr(resolver, "resolve", fake_resolve)
    with pytest.raises(_Resolved) as got:
        BACKENDS[name]()._create_session()
    assert SHA.fullmatch(got.value.args[0])
