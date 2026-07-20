# Concepts — the mental model before the metrics

This is the page to read first. It builds the vocabulary the rest of the docs assume.
The [metrics reference](metrics.md) tells you *what each metric is*; this tells you *how
to think about them* so the reference makes sense. If you only remember one sentence,
make it the last one on this page.

## Why we measure at all

You changed a text-to-speech model, trained a denoiser, or swapped an ASR engine, and
you want to know whether it got better. You could sit a panel of listeners down and ask
them — that is the ground truth — but you cannot do that after every training step. So
we use **objective metrics**: numbers a computer produces from the audio (and sometimes
a reference or a transcript) that *approximate* what humans would have said. Every metric
in this package is a stand-in for a human judgement, and knowing which judgement it
stands in for is the whole game.

## The two big families: intrusive vs non-intrusive

The first fork in the road is whether the metric needs a **clean reference** to compare
against.

- **Intrusive (reference-based).** You have the "right answer" audio and the degraded
  audio, aligned in time, and the metric measures the difference. STOI, SI-SDR, MCD and
  the spectral distances all work this way. In this package they are called with `ref=`
  and are pure numpy — no model download. They are precise and cheap, but they only exist
  when you actually have a reference: a clean recording, a ground-truth studio take, or
  the source signal before you added noise.
- **Non-intrusive (no-reference).** You have only the audio you want to judge — no clean
  version exists. This is the normal situation for synthesized speech (there is no "true"
  waveform a TTS clip should equal) and for real-world recordings. Here we use a **neural
  model trained to predict human ratings**: UTMOS, DNSMOS, NISQA, SIGMOS. They download an
  ONNX model on first use and output a score directly from one signal.

The reference requirement is not a detail — it decides which metrics are even available
to you. See [choosing.md](choosing.md) for the task-by-task version of this decision.

## What "MOS" is, and why we predict it

**MOS** is *Mean Opinion Score*. It comes from **ITU-T Recommendation P.800**, the
telephony-quality standard: you play a clip to many listeners, each rates it 1 (bad) to
5 (excellent), and the MOS is the average. That laboratory average is the human ground
truth for speech quality.

The trouble is that a P.800 listening test is slow and expensive. So two things happened:

- The protocol was adapted. **P.808** runs the same absolute-category-rating test through
  *crowdsourcing* instead of a lab. **P.835** separates the judgement into three scores —
  the **speech signal** alone (SIG), the **background** intrusiveness (BAK), and the
  **overall** quality (OVRL) — which is what you need to tell "the denoiser hurt the
  voice" apart from "the denoiser left noise in". **P.804** defines a finer set of
  degradation dimensions (coloration, discontinuity, loudness, noisiness, reverb).
- Models were trained to *predict* those human scores from audio. That is what a
  "no-reference MOS predictor" is. DNSMOS is literally named after the P.808 and P.835
  protocols it was trained to reproduce; SIGMOS after P.804; UTMOS predicts the naturalness
  MOS collected for synthesized speech. When UTMOS prints `4.1`, it means *"a P.800-style
  listening panel would probably have rated this around 4.1"* — a prediction, not a
  measurement.

So the neural MOS metrics are non-intrusive by construction: they replace the human panel,
and a panel does not need a reference.

## Three axes that are easy to confuse

"Better audio" is not one thing. Keep these three apart:

- **Intelligibility** — *can you make out the words?* This is what STOI/ESTOI estimate.
  A clip can be perfectly intelligible and still sound robotic or noisy. Intelligibility
  is the axis that matters for hearing aids and noise-suppression-for-comprehension.
- **Quality / naturalness** — *does it sound good / human / clean?* This is what the MOS
  predictors estimate. High quality usually implies high intelligibility, but not the
  reverse.
- **Similarity** — *is this the same speaker / the same signal as the reference?* This is
  what speaker similarity (voice-conversion, verification) and the signal-fidelity metrics
  (SI-SDR, MCD) measure. A voice can be natural and intelligible while being the *wrong*
  voice.

A metric that is excellent on one axis tells you almost nothing about the others. Reporting
a single number across all three is the most common evaluation mistake.

## Why each task needs different metrics

- **Text-to-speech (TTS).** There is no reference waveform, and you care about naturalness.
  → no-reference MOS (UTMOS, DNSMOS, NISQA).
- **Speech enhancement / denoising.** You often *do* have the clean signal (you added the
  noise yourself), and you care about how much of the target you kept vs how much noise
  remains. → SI-SDR when a reference exists, plus DNSMOS/SIGMOS/NISQA to judge the result
  the way a listener would.
- **Voice conversion (VC).** You care about three things at once: is it the *target*
  speaker (similarity), does it sound natural (MOS), and how far did the acoustics move
  (MCD). → speaker_similarity + UTMOS + MCD.
- **Automatic speech recognition (ASR).** The output is text, not audio, so audio metrics
  do not apply at all. → WER/CER over the transcript.
- **Source separation.** You have reference sources and want to know how cleanly each was
  recovered, regardless of gain. → SI-SDR.

## Two gotchas that silently corrupt scores

- **Sample rate.** Every metric has a rate it expects. This package resamples for you —
  UTMOS/DNSMOS run at 16 kHz, SIGMOS at 48 kHz, STOI analyses at 10 kHz, NISQA adapts to
  the file's own rate — but if you compare numbers computed by *different* tools, mismatched
  rates are a classic source of disagreement. Feeding an 8 kHz telephone clip to a 48 kHz
  model measures the upsampling as much as the speech.
- **Alignment.** Intrusive metrics compare sample *n* of the degraded signal against sample
  *n* of the reference. If the two are offset — even by a few milliseconds of leading
  silence, or a different codec latency — a sample-fidelity metric like SI-SDR collapses
  even though the audio sounds identical. MCD sidesteps this with dynamic time warping over
  frames, which is exactly why MCD is the workhorse for TTS/VC where alignment is loose,
  and SI-SDR is not. Length-matching and the rules for unscorable inputs live in
  `speechonnxmetrics.intrusive._common`.

## The golden rule

**No objective metric replaces a listening test — it only approximates one.** UTMOS is a
model's *guess* at what listeners would say; STOI correlates with intelligibility but is
not intelligibility; SI-SDR can be high on a clip that sounds bad and low on one that
sounds fine. Use these metrics to *rank* systems quickly, to catch regressions, and to
guide iteration — then confirm the decisions that matter with real ears. Every metric here
is a fast, cheap, repeatable proxy for a human judgement, and it is only ever as trustworthy
as that proxy relationship.

---

Next: [choosing.md](choosing.md) — pick the right metrics for your task · the
[per-metric guides](metric-guides/) — history and use of each one.
