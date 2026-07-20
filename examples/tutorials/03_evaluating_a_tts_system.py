"""Tutorial 03 — evaluating a text-to-speech system.

A TTS system produces speech with no "correct" waveform to compare against, so intrusive
metrics do not apply. The standard approach is a panel of NO-REFERENCE MOS predictors:

  * UTMOS  -> one naturalness number (built for synthesized speech, VoiceMOS Challenge).
  * DNSMOS -> the ITU-T P.835 triplet: sig (speech), bak (background), ovrl (overall).
  * NISQA  -> overall MOS plus a degradation breakdown (noi/dis/col/loud).

Reporting several gives a fuller picture than any single number: they were trained on
different data and disagree in informative ways.

    python examples/tutorials/03_evaluating_a_tts_system.py

First run downloads the MOS weights (~440 MB across all three), then caches them.
NOTE: NISQA weights are CC BY-NC-SA 4.0 (NonCommercial). Drop 'nisqa' if your use is
commercial; UTMOS and DNSMOS are MIT.
"""
import speechonnxmetrics as s

# Several "synthesized" clips (here, neural-codec resyntheses standing in for TTS output).
CLIPS = {
    "facodec": "test/fixtures/audio/facodec_aria.wav",
    "bicodec": "test/fixtures/audio/bicodec_aria.wav",
    "source ": "test/fixtures/audio/source.wav",  # the real recording, as a sanity ceiling
}

print(__doc__)
print("Scoring each clip on UTMOS + DNSMOS + NISQA...\n")

for name, path in CLIPS.items():
    r = s.score(path, ["utmos", "dnsmos", "nisqa"])
    # score() flattens dict-valued metrics: dnsmos -> dnsmos.sig/.bak/.ovrl, etc.
    print(f"[{name}]  UTMOS={r['utmos']:.2f}   "
          f"DNSMOS(sig/bak/ovrl)={r['dnsmos.sig']:.2f}/{r['dnsmos.bak']:.2f}/{r['dnsmos.ovrl']:.2f}   "
          f"NISQA.mos={r['nisqa.mos']:.2f}")

print()
print("Interpreting the table:")
print("  - All scores are 1..5, higher = better. The real 'source' recording should sit")
print("    at or near the top on most dimensions — a useful sanity ceiling for the codecs.")
print("  - UTMOS is your headline naturalness number.")
print("  - DNSMOS.bak flags residual background/noise; DNSMOS.sig flags voice degradation.")
print("  - NISQA gives a second, independently-trained opinion; big UTMOS/NISQA gaps are")
print("    a cue to actually listen.")
print()
print("Remember: these are PREDICTIONS of a listening test. Use them to rank and to catch")
print("regressions; confirm the decisions that matter with real ears.")
