# Speaker similarity
<!-- agent-hint
covers: what speaker_similarity measures, its range, when to use it and its call
source: speechonnxmetrics/speaker/similarity.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/speaker-similarity.md"
see: eer-min_dcf.md
-->

Speaker similarity is the cosine similarity between the speaker embeddings of two
clips. It shows if an output keeps the voice of a target speaker.

## History

Speaker similarity is a standard practice from speaker recognition, with no single
source paper. A model maps a voice to an embedding of fixed length: i-vectors in older
work, neural x-vector or ECAPA embeddings in later work. The cosine similarity of two
embeddings compares the voices. Voice conversion and voice cloning papers report it as
SECS (speaker embedding cosine similarity). The number is only as good as the embedding
model. This package takes the model from `speakeronnx`.

## Range and direction

The value is a cosine similarity, from −1 to 1, and higher means a closer voice.
Speaker embeddings usually give values in the upper part of that range. The metric is
not in the `score()` registry. Call it from `speechonnxmetrics.speaker`, with the
`speaker` extra installed.

## When to use it

- Use it as one side of the voice conversion triangle, with UTMOS for naturalness and
  MCD for acoustic distance. See [Choosing metrics](../choosing.md).
- Use it to check that a TTS voice keeps its identity.
- Do not read a fixed threshold as proof of the same person. Use the verification
  metrics in [EER and minDCF](eer-min_dcf.md) with a calibrated threshold.
- Do not use it for quality. A robotic clip can have a high similarity.

## In this library

`speaker_similarity` takes numpy waveforms, so load the audio first. The call downloads
the embedding model on the first run, so this page shows it without output:

```python
from speechonnxmetrics._dsp.audio import load_audio
from speechonnxmetrics.speaker import speaker_similarity

deg, deg_sr = load_audio("converted.wav")
ref, ref_sr = load_audio("target_speaker.wav")
speaker_similarity(deg, deg_sr, ref=ref, ref_sr=ref_sr)
```

`SpeakerSimilarity` is the reusable form. It loads its embedder on the first call.

## Further reading

The speaker verification literature on i-vector, x-vector and ECAPA-TDNN embeddings.
TTS and voice conversion papers report the metric as SECS.

---
[← WER family](wer-family.md) · [Home](../index.md) · [EER and minDCF →](eer-min_dcf.md)
