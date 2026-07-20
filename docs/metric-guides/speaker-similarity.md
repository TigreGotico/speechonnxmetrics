# Speaker similarity — cosine between speaker embeddings

## History
Speaker similarity is not one named metric with a single paper but a **standard practice**
that grew out of speaker-recognition research: represent a voice as a fixed-length embedding
(i-vectors historically; today neural x-vectors / ECAPA-style embeddings), then compare two
voices by the **cosine similarity** of their embeddings. As voice conversion and voice
cloning matured, this became the default objective measure of *speaker-identity preservation*
— "does the output sound like the target speaker?" — reported as **SECS** (speaker embedding
cosine similarity) or simply "speaker similarity" in TTS/VC papers. The number is only as
good as the embedding model behind it; here that model comes from `speakeronnx`.

## What it measures
How close two clips are in speaker-identity space — cosine similarity between their speaker
embeddings. High = same-sounding speaker; low = different speaker. It says nothing about
naturalness or intelligibility.

## Range & direction
Cosine similarity, roughly **−1 to 1** (in practice speaker embeddings cluster in the upper
part of that range), **higher is better** for identity match. It is *not* in the `score()`
registry — call it from `speechonnxmetrics.speaker`, and it needs the `speaker` extra
(`pip install speechonnxmetrics[speaker]`) for embedding extraction.

## When to use / when not
- **Use** as one leg of the voice-conversion / voice-cloning eval triangle (with UTMOS for
  naturalness and MCD for acoustic distance — see [../choosing.md](../choosing.md)), and to
  check that a TTS voice stays on-identity.
- **Do not** read an absolute threshold as "verified same person" — that is the job of the
  verification metrics ([eer-min_dcf.md](eer-min_dcf.md)) with a calibrated operating point.
  And do not use it as a quality score: a robotic clip can still score high similarity.

## Domain
Speaker recognition → voice conversion and voice cloning (identity preservation).

## In this library
`speaker_similarity` takes numpy waveforms, so load the audio first:
```python
from speechonnxmetrics._dsp.audio import load_audio
from speechonnxmetrics.speaker import speaker_similarity

deg, deg_sr = load_audio("converted.wav")
ref, ref_sr = load_audio("target_speaker.wav")
speaker_similarity(deg, deg_sr, ref=ref, ref_sr=ref_sr)   # e.g. 0.83
```
`SpeakerSimilarity` is the reusable form with a lazily-loaded embedder.

## Further reading
Background lies in speaker-verification embedding literature (i-vector / x-vector / ECAPA-TDNN
lines of work); in TTS/VC it is usually reported as speaker-embedding cosine similarity (SECS).
