# Per-metric guides

One teaching page per metric (or tight family): where it came from, what it measures,
its range and direction, when to use it and when not, the domain it lives in, and the
exact call in this library. These complement the [metrics reference](../metrics.md),
which is the terse lookup table.

Start with [concepts.md](../concepts.md) and [choosing.md](../choosing.md) if you have
not yet.

## No-reference MOS (predict a listening test)
- [utmos.md](utmos.md) — UTMOS naturalness MOS for synthesized speech
- [dnsmos.md](dnsmos.md) — DNSMOS and DNSMOS P.808, for noise suppression
- [sigmos.md](sigmos.md) — SIGMOS, P.804 degradation dimensions
- [nisqa.md](nisqa.md) — NISQA multidimensional quality (NonCommercial weights)

## Intrusive (reference-based)
- [stoi-estoi.md](stoi-estoi.md) — STOI / ESTOI intelligibility
- [si_sdr-sdr-snr.md](si_sdr-sdr-snr.md) — SI-SDR, SDR, SNR signal fidelity
- [mcd.md](mcd.md) — mel-cepstral distortion, the TTS/VC workhorse
- [pitch.md](pitch.md) — log-F0 RMSE and voiced/unvoiced error
- [spectral.md](spectral.md) — LSD, MSD, mel-L1 spectral distances

## ASR / text
- [wer-family.md](wer-family.md) — WER, CER, MER, WIL, WIP

## Speaker
- [speaker-similarity.md](speaker-similarity.md) — cosine speaker similarity
- [eer-min_dcf.md](eer-min_dcf.md) — EER and minDCF verification metrics
