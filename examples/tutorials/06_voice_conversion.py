"""Tutorial 06 — the voice-conversion (VC) eval triangle.

Voice conversion re-speaks source content in a TARGET speaker's voice. It has to satisfy
three INDEPENDENT goals at once, so it needs three metrics — any one alone is gameable:

  1. speaker_similarity -> is it the TARGET speaker? (cosine between speaker embeddings)
  2. utmos              -> does it sound natural?
  3. mcd               -> how far did the acoustics move? (mel-cepstral distortion, dB)

We use the voiceclonnx demo set: source.wav (source voice), reference_aria.wav (the
target speaker 'aria'), and several systems' conversions of the source into aria's voice.

    python examples/tutorials/06_voice_conversion.py

First run downloads UTMOS + a speaker-embedding model (needs the 'speaker' extra),
then caches them. If the demo audio is not present, the script explains and exits 0.
"""
import os

import numpy as np

import speechonnxmetrics as s
from speechonnxmetrics._dsp.audio import load_audio
from speechonnxmetrics.speaker import speaker_similarity

DEMO = "/home/miro/AgentWorkspaces/ml/voiceclonnx/demo"
SOURCE = os.path.join(DEMO, "source.wav")            # source content + source voice
TARGET = os.path.join(DEMO, "reference_aria.wav")    # the target speaker we aim for
SYSTEMS = {                                           # each: source converted to 'aria'
    "knnvc":     os.path.join(DEMO, "outputs", "knnvc__aria.wav"),
    "openvoice": os.path.join(DEMO, "outputs", "openvoice__aria.wav"),
    "cosyvoice": os.path.join(DEMO, "outputs", "cosyvoice__aria.wav"),
}

print(__doc__)

if not (os.path.exists(TARGET) and os.path.exists(SOURCE)):
    print(f"Demo audio not found under {DEMO} — skipping (this is fine).")
    raise SystemExit(0)

target_wav, target_sr = load_audio(TARGET)
source_wav, source_sr = load_audio(SOURCE)

print(f"{'system':11}  spk~TARGET  spk~SOURCE   UTMOS   MCD(vs source content)")
print("-" * 66)
for name, path in SYSTEMS.items():
    if not os.path.exists(path):
        print(f"{name:11}  (missing output — skipped)")
        continue
    conv, conv_sr = load_audio(path)

    # 1) Identity: similarity to the TARGET speaker (want HIGH) and to the SOURCE
    #    speaker (want LOWER — a good conversion moves away from the source voice).
    sim_target = speaker_similarity(np.asarray(conv), conv_sr, ref=np.asarray(target_wav), ref_sr=target_sr)
    sim_source = speaker_similarity(np.asarray(conv), conv_sr, ref=np.asarray(source_wav), ref_sr=source_sr)

    # 2) Naturalness (no reference).
    utmos = s.score(np.asarray(conv, dtype=np.float32), ["utmos"], sr=conv_sr)["utmos"]

    # 3) Acoustic distance to the source (same linguistic content). MCD uses internal
    #    time-warping, so loose alignment is fine. In a full VC benchmark you would
    #    measure MCD against a PARALLEL recording of the target speaker saying the same
    #    words; here, distance-to-source content is an illustrative stand-in.
    mcd = s.score(np.asarray(conv, dtype=np.float32), ["mcd"],
                  ref=np.asarray(source_wav, dtype=np.float32), sr=conv_sr)["mcd"]

    print(f"{name:11}  {sim_target:9.3f}  {sim_source:10.3f}   {utmos:5.2f}   {mcd:6.2f} dB")

print()
print("Reading the triangle:")
print("  - spk~TARGET high AND spk~SOURCE lower  => it moved to the target voice. Good.")
print("  - UTMOS says whether that voice sounds natural or robotic.")
print("  - MCD gives the acoustic-distance leg. No single column decides the winner —")
print("    a system can nail the target speaker while sounding bad, or sound great in the")
print("    wrong voice. You need all three axes at once.")
