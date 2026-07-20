"""WER/CER with explicit, opt-in text normalization.

The scoring functions never normalize on their own — you pass a ``normalizer=``
explicitly. Three presets ship: BASIC (lowercase + whitespace), STRICT (also expands
contractions, strips diacritics, punctuation and filler words), and LEGACY.

LEGACY reproduces the ``re.sub(r"[^a-z' ]", " ", text.lower()).split()`` tokenizer used
by some older call sites. It is provided only so migrating those sites is provably
value-preserving; it silently destroys every non-ASCII character (accented Latin,
Arabic, ...), so do not reach for it in new code.

    python examples/asr_wer.py
"""
from speechonnxmetrics import asr
from speechonnxmetrics.asr import BASIC, LEGACY, STRICT

reference = "The quick brown fox jumps over the lazy dog"
hypothesis = "the quick brown box jumps over a lazy dog"

print("Raw (no normalizer) — case/punctuation differences count as errors:")
print("  WER", asr.wer(reference, hypothesis))
print("  CER", asr.cer(reference, hypothesis))

print("\nBASIC normalizer (lowercase + collapse whitespace):")
print("  WER", asr.wer(reference, hypothesis, normalizer=BASIC))

print("\nSTRICT normalizer (contractions/diacritics/punctuation/fillers):")
print("  WER", asr.wer("I don't know, um, the answer.", "i do not know the answer",
                       normalizer=STRICT))

print("\nLEGACY normalizer (ASCII-only tokenizer; not for new code):")
print("  WER", asr.wer("Hello, WORLD!", "hello world", normalizer=LEGACY))

print("\nFull breakdown via compute():")
print(" ", asr.compute(reference, hypothesis, normalizer=BASIC))
