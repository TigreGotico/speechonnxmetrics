"""Tutorial 02 — reference-based vs no-reference metrics.

This is the single most important distinction in speech evaluation.

  * INTRUSIVE (reference-based) metrics need a clean "right answer" signal to compare
    against. STOI is one. They are precise and cheap, but only exist when you HAVE a
    reference.

  * NO-REFERENCE metrics judge a single signal on its own, with no reference. They work
    by running a neural model trained to predict human MOS ratings. UTMOS is one. This is
    the only option for synthesized speech, where no "true" waveform exists.

We run BOTH on the same clip to feel the difference.

    python examples/tutorials/02_reference_vs_noreference.py

First run downloads ~30 MB of UTMOS weights, then caches them.
"""
import speechonnxmetrics as s

CLIP = "test/fixtures/audio/facodec_aria.wav"
REFERENCE = "test/fixtures/audio/source.wav"

print(__doc__)

# 1) Intrusive: STOI. Needs ref=. Measures intelligibility RELATIVE to the reference.
stoi = s.score(CLIP, ["stoi"], ref=REFERENCE)["stoi"]
print(f"STOI (intrusive, needs a reference) : {stoi:.4f}   [0..1, higher better]")

# 2) No-reference: UTMOS. Notice there is NO ref= — it judges the clip alone.
utmos = s.score(CLIP, ["utmos"])["utmos"]
print(f"UTMOS (no-reference, judges alone)  : {utmos:.4f}   [1..5, higher better]")

print()
print("What just happened:")
print("  - STOI asked 'how close is this to the clean source?' — impossible without one.")
print("  - UTMOS asked 'how good would a listener rate this?' — no reference needed.")
print("  - They are on different scales (0..1 vs 1..5) and mean different things:")
print("    STOI = intelligibility vs a reference; UTMOS = predicted overall quality.")
print()
print("Rule of thumb: have a clean reference? intrusive metrics are available and precise.")
print("No reference (e.g. real TTS output)? you must predict the listener — use MOS models.")
