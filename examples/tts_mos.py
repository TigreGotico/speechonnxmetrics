"""The canonical TTS-eval use case: no-reference neural MOS on synthesized speech.

UTMOS gives one naturalness number; DNSMOS returns a P.835 triplet (sig/bak/ovrl);
NISQA returns a five-dimension quality breakdown. ``score()`` flattens the dict-valued
metrics to ``dnsmos.sig``, ``nisqa.mos`` and so on.

Weights download on first run (~440 MB across all three MOS families) into
``$XDG_DATA_HOME/speechonnxmetrics`` and are cached thereafter.

NOTE: NISQA's weights are CC BY-NC-SA 4.0 (NonCommercial). Drop 'nisqa' below if your
use is commercial; every other metric here is MIT.

    python examples/tts_mos.py
"""
from pprint import pprint

import speechonnxmetrics as s

SYNTH = "test/fixtures/audio/facodec_aria.wav"  # stands in for a TTS/codec output

print("Scoring on UTMOS + DNSMOS + NISQA (downloading models on first run)...")
result = s.score(SYNTH, ["utmos", "dnsmos", "nisqa"])
pprint(result)
