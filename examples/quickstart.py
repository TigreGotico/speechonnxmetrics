"""Score a single audio file on a few no-reference (intrusive-free) metrics.

No-reference metrics need only the signal itself — no clean reference — so this is
the simplest possible use. UTMOS/DNSMOS/NISQA download their ONNX weights on first
call (see tts_mos.py); here we use pure-numpy metrics that never touch the network,
plus one that would download. Run from the repo root:

    python examples/quickstart.py
"""
from pprint import pprint

import speechonnxmetrics as s

AUDIO = "test/fixtures/audio/source.wav"

# Pure-numpy, no download, no reference needed... but every no-reference metric here
# is a MOS predictor. For an offline-only quickstart, score a reference metric instead
# (see reference_metrics.py). UTMOS is the canonical one-number quality score:
print("UTMOS downloads ~30 MB on first run, then caches under XDG_DATA_HOME.")
result = s.score(AUDIO, ["utmos"])
pprint(result)
