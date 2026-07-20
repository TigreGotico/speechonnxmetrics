# Tutorials — a zero-to-hero path

Seven numbered, runnable scripts that teach speech evaluation by doing it. Each is heavily
commented and prints as it goes, so read the output alongside the code. Run them from the
**repo root** so the fixture paths resolve:

```bash
python examples/tutorials/01_what_is_a_metric.py
```

They pair with the written path in [`../../docs/`](../../docs/): read
[concepts.md](../../docs/concepts.md) and [choosing.md](../../docs/choosing.md) for the
theory, run these for the practice.

| # | Script | Teaches |
|---|---|---|
| 01 | `01_what_is_a_metric.py` | What an objective metric *is* — compute one intrusive metric and read the number |
| 02 | `02_reference_vs_noreference.py` | The intrusive vs no-reference split, via STOI (needs a reference) and UTMOS (does not) |
| 03 | `03_evaluating_a_tts_system.py` | The canonical TTS eval: UTMOS + DNSMOS + NISQA, and how to interpret them |
| 04 | `04_evaluating_enhancement.py` | Enhancement eval: SI-SDR and DNSMOS move as a clip goes clean → noisy → "enhanced" |
| 05 | `05_evaluating_asr.py` | WER/CER, and how the *normalizer* you choose changes the score |
| 06 | `06_voice_conversion.py` | The VC eval triangle: speaker_similarity + MCD + UTMOS |
| 07 | `07_batch_and_scale.py` | `score_batch` as an eval loop: failure isolation and reading a results table |

**First-run downloads.** Scripts 02, 03, 04, 06 and 07 use neural MOS predictors, which
download their ONNX weights (~440 MB across all MOS families) on first use into
`$XDG_DATA_HOME/speechonnxmetrics` and cache them thereafter. Script 06 also downloads a
speaker-embedding model via the `speaker` extra. Scripts 01 and 05 are pure numpy and never
touch the network.

**Licence note.** Script 03 uses NISQA, whose weights are CC BY-NC-SA 4.0 (NonCommercial).
Every other metric used here is safe for commercial use. See
[../../docs/models.md](../../docs/models.md).
