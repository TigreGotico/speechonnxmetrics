# Metric guides
<!-- agent-hint
covers: one teaching page for each metric or family, and the order to read them
source: docs/metric-guides/
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/README.md"
see: utmos.md
-->

Each guide covers one metric or one family of metrics: where it comes from, what it
measures, its range and direction, when to use it, and the call in this library. The
[metrics reference](../metrics.md) is the short lookup table. Read
[Concepts](../concepts.md) and [Choosing metrics](../choosing.md) first.

| Family | Guide | What you get there |
| :--- | :--- | :--- |
| No-reference MOS | [UTMOS](utmos.md) | Naturalness MOS for synthesized speech |
| No-reference MOS | [DNSMOS](dnsmos.md) | DNSMOS P.835 and P.808, for noise suppression |
| No-reference MOS | [SIGMOS](sigmos.md) | The seven P.804 quality dimensions |
| No-reference MOS | [NISQA](nisqa.md) | Overall quality and four dimensions, with NonCommercial weights |
| Intrusive | [STOI and ESTOI](stoi-estoi.md) | Intelligibility |
| Intrusive | [SI-SDR, SDR and SNR](si_sdr-sdr-snr.md) | Signal fidelity in dB |
| Intrusive | [MCD](mcd.md) | Mel-cepstral distortion for TTS and voice conversion |
| Intrusive | [Pitch](pitch.md) | Log-F0 RMSE and voiced/unvoiced error |
| Intrusive | [Spectral distances](spectral.md) | LSD, MSD and mel-L1 |
| Text | [WER family](wer-family.md) | WER, CER, MER, WIL and WIP |
| Speaker | [Speaker similarity](speaker-similarity.md) | Cosine similarity of speaker embeddings |
| Speaker | [EER and minDCF](eer-min_dcf.md) | Speaker verification metrics |

---
[← Models and licences](../models.md) · [Home](../index.md) · [UTMOS →](utmos.md)
