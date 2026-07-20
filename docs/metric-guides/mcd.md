# MCD — mel-cepstral distortion

## History
MCD was defined by Robert Kubichek in *Mel-Cepstral Distance Measure for Objective Speech
Quality Assessment* (IEEE Pacific Rim Conference, PACRIM, 1993). It was proposed in the
speech-**coding** era as an objective measure of how far a coded/synthesized signal's
spectral envelope had drifted from the original — a cheap stand-in for listening tests when
comparing codecs. Over the following decades it became the *workhorse* objective metric for
**text-to-speech and voice conversion**, because it captures exactly the spectral-envelope
differences those systems get judged on, and it tolerates the loose time-alignment typical
of synthesized speech.

## What it measures
The distance (in dB) between the **mel-cepstral coefficients** of two signals — essentially
how different their spectral envelopes are, frame by frame. Lower means the synthesized or
converted speech is acoustically closer to the target.

## Range & direction
Unbounded **dB**, **lower is better**. Intrusive: pass `ref=`. Unlike SI-SDR, MCD aligns the
two signals with **dynamic time warping** over frames (see `speechonnxmetrics/_dsp/dtw.py`),
so small timing differences do not wreck it — which is precisely why it suits TTS/VC where
the reference and output say the *same words* but not at the same instants.

## When to use / when not
- **Use** for TTS, voice conversion, and vocoder/neural-codec resynthesis, where you have a
  parallel ground-truth utterance of the same content and want a spectral-fidelity number.
- **Do not** compute MCD between clips of *different* content — the comparison is only
  meaningful for the same utterance. It is not a quality/naturalness score (use a MOS
  predictor) and not a speaker-identity score (use speaker_similarity).

## Domain
Speech coding (origin) → the standard objective metric of TTS and voice-conversion research
today.

## In this library
```python
import speechonnxmetrics as s
s.score("test/fixtures/audio/facodec_aria.wav", ["mcd"],
        ref="test/fixtures/audio/source.wav")
# -> {'mcd': 10.45...}   (dB; lower is closer to the reference)
```
Validated against `pymcd` (see [../metrics.md](../metrics.md)).

## Further reading
Kubichek, *Mel-Cepstral Distance Measure for Objective Speech Quality Assessment*, IEEE
PACRIM 1993.
