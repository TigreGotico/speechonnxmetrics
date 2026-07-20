"""Intrusive (reference-based) metrics on a degraded/reference pair.

Intrusive metrics compare a degraded signal against a clean reference and are pure
numpy — no model download, fully offline. STOI measures intelligibility (0-1, higher
better), MCD the mel-cepstral distortion in dB (lower better), SI-SDR the
scale-invariant signal-to-distortion ratio in dB (higher better).

    python examples/reference_metrics.py
"""
from pprint import pprint

import speechonnxmetrics as s

# facodec_aria.wav is a neural-codec reconstruction of source.wav; source is the clean
# reference it should be compared against.
DEG = "test/fixtures/audio/facodec_aria.wav"
REF = "test/fixtures/audio/source.wav"

result = s.score(DEG, ["stoi", "estoi", "mcd", "si_sdr", "lsd"], ref=REF)
pprint(result)
