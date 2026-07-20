"""One-off generator for test/fixtures/stoi_fixture.json — not part of the test suite.

Every fixture case is scored with pystoi at exactly the audio's native rate (10 kHz,
the STOI standard analysis rate). This is deliberate: our own resampler
(``_dsp.resample.kaiser_resample``, a torchaudio-faithful design) and pystoi's own
resampler (``pystoi.utils.resample_oct``, an Octave/MATLAB-compatible design) are
different filters that agree closely but not to machine precision — feeding
already-16kHz audio through both and comparing final scores conflates "does our STOI
math agree with pystoi's" with "do two unrelated resamplers agree with each other".
Committing fixture audio pre-resampled to 10 kHz removes that confound; parity here
tests the STOI/ESTOI algorithm itself (VAD, band decomposition, correlation), which
agrees with pystoi to ~1e-6.

Run with pystoi installed (`uv pip install --python ~/.venvs/ovos/bin/python pystoi`,
test-only, never a declared dependency) to regenerate the committed fixture:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/.venvs/ovos/bin/python test/fixtures/generate_stoi_fixture.py
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
from pystoi import stoi as pystoi_stoi

FS = 10000
AUDIO_DIR = Path(__file__).parent / "audio"
OUT = Path(__file__).parent / "stoi_fixture.json"


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        assert wf.getframerate() == FS, (path, wf.getframerate())
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0


def _synthetic_speechlike(duration: float, sr: int, seed: int) -> np.ndarray:
    """A stationary harmonic stack — not representative of real speech (see
    test_intrusive.py), kept only as a pure-arithmetic regression anchor."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    f0 = 110 + 30 * np.sin(2 * np.pi * (0.4 + 0.1 * rng.random()) * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    harmonics = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * (2.5 + rng.random()) * t)
    return 0.2 * harmonics * env


def _noisy(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.size)
    noise *= np.sqrt(np.mean(x ** 2) / (10 ** (snr_db / 10))) / (np.std(noise) + 1e-9)
    return x + noise


def main() -> None:
    cases = []

    # Real degraded pairs: source.wav vs genuine codec/voice-conversion outputs of
    # the same underlying content (trimmed 2s clips from speechonnxmetrics's sibling
    # voiceclonnx demo, resampled once to 10 kHz — see the audio/ directory).
    source = _read_wav(AUDIO_DIR / "source.wav")
    for name in ("facodec_aria", "bicodec_aria"):
        deg = _read_wav(AUDIO_DIR / f"{name}.wav")
        cases.append({
            "kind": "real", "ref_file": "source.wav", "deg_file": f"{name}.wav",
            "stoi": float(pystoi_stoi(source, deg, FS, extended=False)),
            "estoi": float(pystoi_stoi(source, deg, FS, extended=True)),
        })

    # Synthetic-noise-over-real-speech: controlled SNR corruption of the real clip,
    # giving a clean degradation curve on genuine speech content.
    for snr_db in (40, 20, 10, 0):
        deg = _noisy(source, snr_db, seed=snr_db + 1000)
        cases.append({
            "kind": "real_noisy", "ref_file": "source.wav", "snr_db": snr_db,
            "stoi": float(pystoi_stoi(source, deg, FS, extended=False)),
            "estoi": float(pystoi_stoi(source, deg, FS, extended=True)),
        })
    cases.append({
        "kind": "real_identical", "ref_file": "source.wav",
        "stoi": float(pystoi_stoi(source, source, FS, extended=False)),
        "estoi": float(pystoi_stoi(source, source, FS, extended=True)),
    })

    # A couple of pure-synthetic cases as arithmetic regression anchors (generated
    # directly at 10 kHz, no resampling involved on either side).
    for seed in (0, 1):
        x = _synthetic_speechlike(2.0, FS, seed)
        y = _noisy(x, 20, seed + 100)
        cases.append({
            "kind": "synthetic", "seed": seed, "snr_db": 20,
            "stoi": float(pystoi_stoi(x, y, FS, extended=False)),
            "estoi": float(pystoi_stoi(x, y, FS, extended=True)),
        })

    OUT.write_text(json.dumps({"sr": FS, "cases": cases}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
