"""One-off generator for test/fixtures/audio/*.wav — not part of the test suite.

Trims and downsamples three real speech clips from the sibling ``voiceclonnx`` repo's
demo assets (source speech plus two genuinely codec/voice-conversion-degraded
versions of the same content) to 2 s @ 10 kHz 16-bit PCM, for the STOI/ESTOI parity
fixture. scipy is the resampler here (test-only, matches the fixture generator's
"pystoi is the only oracle we compare against" contract; it never becomes a runtime
dependency of ``speechonnxmetrics`` itself — see ``pyproject.toml``'s ``test`` extra).

Requires the ``voiceclonnx`` repo checked out as a workspace sibling. Re-run only if
the source assets change:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/.venvs/ovos/bin/python test/fixtures/generate_stoi_audio.py
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf

DEMO = Path("/home/miro/AgentWorkspaces/ml/voiceclonnx/demo")
OUT_DIR = Path(__file__).parent / "audio"
SR_IN = 16000
SR_OUT = 10000
START_S, DURATION_S = 1.0, 2.0

CLIPS = {
    "source": DEMO / "source.wav",
    "facodec_aria": DEMO / "outputs" / "facodec__aria.wav",
    "bicodec_aria": DEMO / "outputs" / "bicodec__aria.wav",
}


def _load_trimmed(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    assert sr == SR_IN, (path, sr)
    start, end = int(START_S * sr), int((START_S + DURATION_S) * sr)
    return x[start:end]


def _resample(x: np.ndarray) -> np.ndarray:
    n_out = int(round(len(x) * SR_OUT / SR_IN))
    return sig.resample_poly(x, SR_OUT, SR_IN).astype(np.float32)[:n_out]


def _write_pcm16(path: Path, x: np.ndarray) -> None:
    samples = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR_OUT)
        wf.writeframes(samples.tobytes())


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name, src_path in CLIPS.items():
        clip = _resample(_load_trimmed(src_path))
        out_path = OUT_DIR / f"{name}.wav"
        _write_pcm16(out_path, clip)
        print(f"wrote {out_path} ({clip.size} samples @ {SR_OUT} Hz)")


if __name__ == "__main__":
    main()
