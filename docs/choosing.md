# Choosing metrics — a decision guide by task

Read [concepts.md](concepts.md) first for the intrusive/non-intrusive and
intelligibility/quality/similarity distinctions this page leans on. Here we go straight
to "I am building X, what do I report?" Ranges and exact call shapes are in
[metrics.md](metrics.md); the *why* of each metric is in its
[per-metric guide](metric-guides/).

## Task → metric cheat sheet

| You are evaluating… | Report | Also useful | Needs a reference? |
|---|---|---|---|
| **Text-to-speech (TTS)** naturalness | `utmos` | `dnsmos`, `nisqa` | no |
| **Speech enhancement / denoising** | `dnsmos` (P.835 sig/bak/ovrl), `sigmos` | `nisqa`, `si_sdr` (if clean ref) | SI-SDR yes, MOS no |
| **Voice conversion (VC)** | `speaker_similarity`, `utmos`, `mcd` | `dnsmos` | similarity+MCD yes |
| **Automatic speech recognition (ASR)** | `wer`, `cer` | `mer`, `wil`/`wip` | text reference |
| **Source separation** | `si_sdr` | `sdr`, `stoi` | yes |
| **Speaker verification** | `eer`, `min_dcf` | — | scores+labels |
| **Vocoder / neural codec resynthesis** | `mcd`, `stoi` | `si_sdr`, `mel_l1`, `lsd` | yes |
| **Hearing / intelligibility research** | `stoi`, `estoi` | — | yes |

## The reasoning, task by task

### TTS — no reference exists, so predict the listener
A synthesizer produces speech that has no "correct" waveform, so intrusive metrics do not
apply. Use **UTMOS** as the single naturalness number (it was built for exactly this — the
VoiceMOS Challenge on synthesized speech), and add **DNSMOS** and **NISQA** for a
multi-dimensional read when UTMOS alone is ambiguous. Do *not* reach for STOI or MCD unless
you have parallel ground-truth recordings of the same utterances — and even then, MCD needs
them time-aligned.

### Enhancement / denoising — separate the voice from the noise
The defining question is "did you remove noise without hurting the voice?" — which is
exactly the SIG/BAK/OVRL split. **DNSMOS** (named after the P.835 protocol) is the standard
answer, with **SIGMOS**/**NISQA** giving finer degradation dimensions (coloration, reverb,
discontinuity). If you have the clean signal — you usually do in a denoising benchmark,
because you added the noise — add **SI-SDR** for the signal-fidelity view. Report both: a
denoiser can raise SI-SDR while introducing artifacts that DNSMOS penalizes.

### Voice conversion — the eval triangle
VC has to satisfy three independent goals, so it needs three metrics (see
[concepts.md](concepts.md) on the three axes):

- **`speaker_similarity`** — is it the *target* speaker? (cosine between speaker embeddings)
- **`utmos`** — does it sound natural?
- **`mcd`** — how far did the acoustic realization move from the target? (MCD tolerates the
  loose alignment typical of VC via internal time-warping)

Any one of these alone is gameable: a system can nail the target speaker while sounding
robotic, or sound gorgeous in the wrong voice.

### ASR — the output is text
Audio metrics do not apply. **WER** (word error rate) is the universal default; **CER**
(character error rate) is better for languages without clear word boundaries, for
morphologically rich languages, and when you want partial-credit on near-miss words. Use
**MER/WIL/WIP** when you want a bounded [0,1] information-theoretic view (WER is unbounded).
Crucial subtlety: **normalization changes the score**, and this package never normalizes on
its own — you pass a normalizer explicitly. Decide and document your normalization before
comparing systems.

### Source separation — gain-invariant fidelity
**SI-SDR** exists precisely because the older SDR/BSS-Eval was "abused" for single-channel
separation (see the SI-SDR guide). It is scale-invariant, so a system is not rewarded or
punished for the arbitrary gain of its output — the right choice when comparing separators.

### Speaker verification — a detection problem
Here each trial is "same speaker or not?" and you have a score per trial plus a genuine/
impostor label. **EER** (equal error rate) is the single easy-to-read summary; **minDCF**
(from the NIST SRE convention) weights misses and false alarms for an operating point and is
what the field reports when the application cares more about one error type.

## When NOT to use a metric

- **MCD needs parallel, aligned utterances.** It compares mel-cepstra frame by frame (with
  time-warping, but still the *same content*). Computing MCD between two clips of *different*
  sentences is meaningless. Great for TTS/VC/vocoder resynthesis where the target content is
  known; useless for open-ended "how good does this sound".
- **STOI is intelligibility, not quality.** A clip can score high STOI and still sound
  unpleasant or unnatural. Never use STOI as a proxy for "quality" — use a MOS predictor.
- **SI-SDR is alignment- and reference-sensitive.** A few milliseconds of latency between
  degraded and reference tank it even when the audio is perceptually identical. Do not use it
  for TTS (no aligned reference) or when your pipeline introduces variable delay.
- **WER has no upper bound.** A hypothesis longer than the reference can score WER > 1.0.
  When you need a bounded, symmetric measure, use MER or WIL.
- **PESQ is deliberately absent.** ITU-T P.862 (PESQ) licensing is incompatible with an
  open, pip-installable package, and the neural MOS predictors here supersede it for most
  uses. If you specifically need PESQ numbers, use a dedicated package under your own licence
  review. See [../README.md](../README.md) ("Not provided (on purpose)").
- **NISQA weights are NonCommercial (CC BY-NC-SA 4.0).** Every other metric is safe for
  commercial use; if yours is commercial, drop `nisqa` and lean on UTMOS/DNSMOS/SIGMOS. See
  [models.md](models.md).

---

Next: the [per-metric guides](metric-guides/) for the history and detail behind each choice.
