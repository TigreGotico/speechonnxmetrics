"""Batch scoring with per-item failure isolation — the eval-loop hot path.

``score_batch`` resolves each metric from the registry once and reuses it across every
item, so an ONNX metric builds its session on the first clip and every later clip
reuses it. A clip that fails to load or score does not abort the run: its value is
``None`` and the reason is recorded under ``"_errors"`` for that item only. That is
what keeps an hours-long eval over thousands of clips from dying on one corrupt file.

    python examples/batch_eval.py
"""
from pprint import pprint

import speechonnxmetrics as s

# A TTS eval loop scores several synthesized clips. We deliberately include a path that
# does not exist to show the failure-isolation behaviour.
clips = [
    "test/fixtures/audio/source.wav",
    "test/fixtures/audio/facodec_aria.wav",
    "test/fixtures/audio/does_not_exist.wav",  # will fail — the batch keeps going
    "test/fixtures/audio/bicodec_aria.wav",
]

results = s.score_batch(clips, metrics=["utmos"])

for path, row in zip(clips, results):
    print(f"\n{path}")
    pprint(row)

ok = [r for r in results if "_errors" not in r]
print(f"\n{len(ok)}/{len(results)} clips scored; the rest are isolated in '_errors'.")
