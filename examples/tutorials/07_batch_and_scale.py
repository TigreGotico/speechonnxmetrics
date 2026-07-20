"""Tutorial 07 — batch scoring at scale, and reading a results table.

Real evaluation runs over hundreds or thousands of clips. score_batch is the hot path:

  * It resolves each metric from the registry ONCE and reuses it, so an ONNX model builds
    its session on the first clip and every later clip reuses it — one model load, not N.
  * It isolates failures: a clip that fails to load or score records None for that clip
    plus the reason under '_errors', and the batch keeps running. One corrupt file cannot
    sink an hours-long run.

We deliberately include a nonexistent path to show the isolation, then turn the results
into a small table and a mean — the shape a real eval loop produces.

    python examples/tutorials/07_batch_and_scale.py

First run downloads UTMOS weights, then caches them.
"""
import speechonnxmetrics as s

clips = [
    "test/fixtures/audio/source.wav",
    "test/fixtures/audio/facodec_aria.wav",
    "test/fixtures/audio/does_not_exist.wav",   # will fail — batch keeps going
    "test/fixtures/audio/bicodec_aria.wav",
]

print(__doc__)

results = s.score_batch(clips, metrics=["utmos"])

print(f"{'clip':40}  utmos   status")
print("-" * 62)
scored = []
for path, row in zip(clips, results):
    name = path.split("/")[-1]
    if "_errors" in row:
        print(f"{name:40}  {'--':>5}   FAILED: {list(row['_errors'].values())[0][:24]}...")
    else:
        print(f"{name:40}  {row['utmos']:5.2f}   ok")
        scored.append(row["utmos"])

print()
print(f"{len(scored)}/{len(clips)} clips scored; {len(clips) - len(scored)} isolated in '_errors'.")
if scored:
    print(f"Mean UTMOS over the scored clips: {sum(scored) / len(scored):.3f}")
print()
print("This is the backbone of an eval loop: feed a list, get a list of flat dicts, drop")
print("the '_errors' rows (or log them), and aggregate. Because failures are isolated, the")
print("run always finishes and you can inspect exactly which files went wrong afterwards.")
