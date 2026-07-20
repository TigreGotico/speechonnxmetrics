"""Tutorial 01 — what IS an objective speech metric?

An objective metric is a number a computer produces from audio that stands in for a
judgement a human would otherwise make by listening. Nothing here downloads a model:
we use an *intrusive* (reference-based) metric, which is pure numpy.

"Intrusive" means the metric needs two signals — a clean REFERENCE and the DEGRADED
signal you want to judge — and measures the difference between them. STOI (Short-Time
Objective Intelligibility) estimates how intelligible the degraded speech would be to a
listener, on a 0-to-1 scale where higher is better.

    python examples/tutorials/01_what_is_a_metric.py
"""
import speechonnxmetrics as s

# facodec_aria.wav is a neural-codec reconstruction of source.wav. source.wav is the
# clean original — the "right answer" we compare against.
REFERENCE = "test/fixtures/audio/source.wav"       # clean
DEGRADED = "test/fixtures/audio/facodec_aria.wav"   # what we want to judge

print(__doc__)

# score(audio, [metric_names], ref=...) returns a flat dict {metric: value}.
# STOI is intrusive, so we MUST pass ref=; leaving it out would raise.
result = s.score(DEGRADED, ["stoi"], ref=REFERENCE)
stoi = result["stoi"]

print(f"STOI of the codec output vs the clean source: {stoi:.4f}")
print()
print("How to read this number:")
print("  - STOI ranges 0..1; higher = more intelligible.")
print("  - ~1.0 would mean 'as intelligible as the clean reference'.")
print(f"  - {stoi:.2f} says the codec kept most, but not all, intelligibility.")
print()
print("Key idea: STOI did not 'listen' — it correlated short-time speech envelopes of")
print("the two signals. It APPROXIMATES a listening test; it does not replace one.")
print("Next: 02 shows what to do when no clean reference exists at all.")
