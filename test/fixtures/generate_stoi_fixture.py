"""One-off generator for test/fixtures/stoi_fixture.json — not part of the test suite.

Run with pystoi installed (`uv pip install --python ~/.venvs/ovos/bin/python pystoi`,
test-only, never a declared dependency) to regenerate the committed fixture:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/.venvs/ovos/bin/python test/fixtures/generate_stoi_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pystoi import stoi as pystoi_stoi

SR = 16000


def _speechlike(duration: float, sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    f0 = 110 + 30 * np.sin(2 * np.pi * (0.4 + 0.1 * rng.random()) * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    harmonics = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * (2.5 + rng.random()) * t)
    return (0.2 * harmonics * env).astype(np.float32)


def _noisy(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x.size).astype(np.float32)
    noise *= np.sqrt(np.mean(x ** 2) / (10 ** (snr_db / 10))) / (np.std(noise) + 1e-9)
    return x + noise


def main() -> None:
    cases = []
    for seed in range(3):
        x = _speechlike(2.0, SR, seed)
        for snr_db in (40, 20, 10, 0):
            y = _noisy(x, snr_db, seed + 100)
            cases.append({
                "seed": seed, "snr_db": snr_db,
                "stoi": float(pystoi_stoi(x, y, SR, extended=False)),
                "estoi": float(pystoi_stoi(x, y, SR, extended=True)),
            })
        # identical-signal case
        cases.append({
            "seed": seed, "snr_db": None,
            "stoi": float(pystoi_stoi(x, x, SR, extended=False)),
            "estoi": float(pystoi_stoi(x, x, SR, extended=True)),
        })

    out = Path(__file__).parent / "stoi_fixture.json"
    out.write_text(json.dumps({"sr": SR, "cases": cases}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
