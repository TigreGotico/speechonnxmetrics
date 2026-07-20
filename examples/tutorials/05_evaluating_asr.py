"""Tutorial 05 — evaluating an ASR system, and why normalization matters.

Automatic speech recognition outputs TEXT, so audio metrics do not apply — you compare
the recognized transcript against a reference transcript. WER (word error rate) is the
universal default; CER (character error rate) works at the character level.

The subtle, score-changing point: this library NEVER normalizes text on its own. Casing,
punctuation and contractions all count as errors unless you pass a normalizer explicitly.
Choosing a normalizer can swing WER by points, so decide and document it before comparing
systems. Everything here is pure numpy — no downloads.

    python examples/tutorials/05_evaluating_asr.py
"""
from speechonnxmetrics import asr
from speechonnxmetrics.asr import BASIC, STRICT

reference = "I don't know, um, the answer."
hypothesis = "i do not know the answer"

print(__doc__)
print(f"reference : {reference!r}")
print(f"hypothesis: {hypothesis!r}\n")

# 1) Raw: no normalizer. Case, punctuation and "don't" vs "do not" ALL count as errors.
print("Raw (no normalizer) — every surface difference is an error:")
print(f"  WER = {asr.wer(reference, hypothesis):.3f}")
print(f"  CER = {asr.cer(reference, hypothesis):.3f}")

# 2) BASIC: lowercase + collapse whitespace. Removes casing noise only.
print("\nBASIC (lowercase + whitespace):")
print(f"  WER = {asr.wer(reference, hypothesis, normalizer=BASIC):.3f}")

# 3) STRICT: also expands contractions, strips punctuation, diacritics and filler words
#    ('um'). This is the fairest comparison of the actual words recognized.
print("\nSTRICT (contractions/punctuation/diacritics/fillers):")
print(f"  WER = {asr.wer(reference, hypothesis, normalizer=STRICT):.3f}")

# 4) The full breakdown: compute() returns every rate plus the raw edit counts in one pass.
print("\nFull breakdown via compute() with STRICT:")
print(" ", asr.compute(reference, hypothesis, normalizer=STRICT))

print()
print("Takeaways:")
print("  - The SAME transcripts scored very differently under different normalizers.")
print("  - The normalizer is part of your metric definition — report it, and keep it")
print("    fixed across the systems you compare.")
print("  - WER can exceed 1.0 (unbounded); use MER or WIL when you need a bounded 0..1 view.")
