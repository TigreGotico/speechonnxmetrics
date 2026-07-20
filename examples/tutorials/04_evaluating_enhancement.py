"""Tutorial 04 — evaluating a speech-enhancement / denoising model.

Enhancement has a clean signal available (you often add the noise yourself), so you can
use BOTH kinds of metric:

  * SI-SDR (intrusive) — scale-invariant signal-to-distortion ratio in dB, the standard
    signal-fidelity measure. Needs the clean reference. Higher = cleaner.
  * DNSMOS (no-reference) — predicts how a listener would rate the result, and its P.835
    'bak' dimension specifically tracks residual background noise.

We fabricate a three-step story from one clean clip: clean -> noisy -> "enhanced"
(enhanced = a version with the added noise reduced, not perfectly). Watch both metrics
move in the same direction.

    python examples/tutorials/04_evaluating_enhancement.py

First run downloads DNSMOS weights (~cached after), then reuses them.
"""
import numpy as np

import speechonnxmetrics as s
from speechonnxmetrics._dsp.audio import load_audio

CLEAN_PATH = "test/fixtures/audio/source.wav"
clean, sr = load_audio(CLEAN_PATH)          # float32 mono in [-1, 1]
clean = np.asarray(clean, dtype=np.float64)

rng = np.random.default_rng(0)              # fixed seed -> reproducible numbers
noise = rng.standard_normal(clean.shape)
noise /= np.sqrt(np.mean(noise ** 2))       # unit-power noise

# Scale noise to a chosen level relative to the speech.
speech_rms = np.sqrt(np.mean(clean ** 2))
noisy = clean + 0.30 * speech_rms * noise            # heavily degraded
enhanced = clean + 0.08 * speech_rms * noise         # a "denoiser" that removed most noise

print(__doc__)
print(f"Loaded clean clip: {len(clean)} samples @ {sr} Hz\n")

for label, signal in [("clean (ceiling)", clean), ("noisy", noisy), ("enhanced", enhanced)]:
    # Intrusive SI-SDR vs the clean reference, and no-reference DNSMOS on the signal alone.
    r = s.score(signal.astype(np.float32), ["si_sdr", "dnsmos"],
                ref=clean.astype(np.float32), sr=sr)
    print(f"[{label:15}]  SI-SDR={r['si_sdr']:7.2f} dB   "
          f"DNSMOS ovrl={r['dnsmos.ovrl']:.2f}  bak={r['dnsmos.bak']:.2f}")

print()
print("Reading the movement:")
print("  - clean vs itself: SI-SDR is effectively perfect (its own reference).")
print("  - noisy: SI-SDR drops sharply and DNSMOS.bak falls (background is intrusive).")
print("  - enhanced: both recover toward clean — that recovery IS the win you report.")
print()
print("Report both: SI-SDR proves signal fidelity, DNSMOS proves it sounds better to a")
print("listener. A denoiser can raise one while hurting the other, so never trust just one.")
