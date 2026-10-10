# speechonnxmetrics

Unified **speech evaluation metrics**: no-reference MOS, intrusive (reference-based)
signal metrics, ASR-based text metrics, and speaker similarity, behind one small API
that runs on **numpy and onnxruntime only**.

## Why numpy + onnxruntime only

The entire runtime depends on just **numpy, onnxruntime and huggingface_hub**. There is
**no torch in the runtime path**. Neural MOS predictors (UTMOS, DNSMOS, NISQA, SIGMOS)
run as exported ONNX graphs, so the package installs small, imports fast, and runs the
same on CPU everywhere. The `export` extra pulls in torch/onnx, but only for offline
model conversion by maintainers, never at inference time.

Model weights download from public HuggingFace repos on first use and cache locally.
`import speechonnxmetrics` touches neither the network nor onnxruntime.

## Install

Not yet on PyPI. Install from git:

```bash
pip install git+https://github.com/TigreGotico/speechonnxmetrics
```

Once published, `pip install speechonnxmetrics` will work too.

Extras (all optional):

| extra | pulls in | needed for |
|---|---|---|
| `audio` | `soundfile` | non-WAV input (mp3/flac/ogg…). Base install decodes WAV via the stdlib. |
| `speaker` | `speakeronnx` | speaker-embedding extraction for `speaker_similarity` |
| `asr` | `onnx-asr` | running an ASR model to get hypotheses (WER/CER scoring itself needs nothing) |
| `vad` | `vadonnx` | voice-activity gating |
| `lect` | `orthography2ipa[portuguese]`, `scriptconv` | expected phones per lect for `lect_fidelity`, with Portuguese syllabification and notation conversion |
| `lect-tugaphone` | the `lect` extra and `tugaphone` | the `tugaphone` expected-phone provider for `lect_fidelity` |
| `export` | `torch`, `onnx` | maintainer-only offline model conversion |
| `test` | `pytest`, `pytest-cov`, `scipy`, the `lect` dependencies and `tugaphone` | running the test suite (scipy is a test-only oracle) |

```bash
pip install "speechonnxmetrics[audio,speaker]"
```

## Quickstart

```python
import speechonnxmetrics as s

# No-reference neural MOS (downloads UTMOS weights on first call, then caches):
print(s.score("test/fixtures/audio/source.wav", ["utmos"]))
# -> {'utmos': 4.4115...}

# Intrusive metrics need a clean reference (pure numpy, no download):
print(s.score("test/fixtures/audio/facodec_aria.wav",
              ["stoi", "mcd", "si_sdr"],
              ref="test/fixtures/audio/source.wav"))
# -> {'stoi': 0.662..., 'mcd': 10.459..., 'si_sdr': -26.937...}
```

Dict-valued metrics flatten into the result. `dnsmos` becomes `dnsmos.sig`,
`dnsmos.bak`, `dnsmos.ovrl`. `nisqa` becomes `nisqa.mos`, `nisqa.noi`, and more.
See [`examples/`](examples/): every script there runs against the bundled fixture audio.

## Metrics

21 metrics live in one registry. `score()`/`score_batch()` dispatch the **audio**
metrics. The **text** metrics compare strings and are called directly from
`speechonnxmetrics.asr`. Enumerate everything at runtime with `s.list_metrics()` or
`speechonnxmetrics list`.

### No-reference MOS (audio, needs model download)

| metric | range | ↑better | output | meaning |
|---|---|---|---|---|
| `utmos` | 1-5 | yes | float | UTMOS22 naturalness MOS |
| `dnsmos` | 1-5 | yes | `sig`, `bak`, `ovrl` | DNSMOS P.835 speech / background / overall quality |
| `dnsmos_p808` | 1-5 | yes | float | DNSMOS P.808 crowdsourced-listening MOS |
| `sigmos` | 1-5 | yes | 7 dims | SIGMOS P.804 quality (`col`,`disc`,`loud`,`noise`,`reverb`,`sig`,`ovrl`) |
| `nisqa` | 1-5 | yes | `mos`,`noi`,`dis`,`col`,`loud` | NISQA-v2 quality, **NonCommercial weights** |

### Intrusive / reference-based (audio, pure numpy, needs `ref=`)

| metric | range | ↑better | meaning |
|---|---|---|---|
| `stoi` | 0-1 | yes | short-time objective intelligibility |
| `estoi` | 0-1 | yes | extended STOI |
| `si_sdr` | dB | yes | scale-invariant signal-to-distortion ratio |
| `sdr` | dB | yes | signal-to-distortion ratio |
| `snr` | dB | yes | signal-to-noise ratio |
| `mcd` | dB | no | mel-cepstral distortion |
| `log_f0_rmse` | n/a | no | log-F0 RMSE (pitch error) |
| `vuv_error` | 0-1 | no | voiced/unvoiced decision error rate |
| `lsd` | dB | no | log-spectral distance |
| `msd` | dB | no | mel-spectral distortion |
| `mel_l1` | n/a | no | L1 distance on log-mel spectrograms |

### ASR-based text metrics (text, pure numpy, string in/out)

| metric | range | ↑better | meaning |
|---|---|---|---|
| `wer` | ≥0 | no | word error rate |
| `cer` | 0-1 | no | character error rate |
| `mer` | 0-1 | no | match error rate |
| `wil` | 0-1 | no | word information lost |
| `wip` | 0-1 | yes | word information preserved |

### Speaker similarity and verification

Not in the `score()` registry. Call these from `speechonnxmetrics.speaker`:
`speaker_similarity` (cosine between speaker embeddings, needs the `speaker` extra),
plus pure-numpy `eer`, `min_dcf` and `equal_error_threshold` over score/label arrays.

### Lect fidelity

`lect_fidelity` measures how far a clip of known text realises one lect of a language
against another, for example European against Brazilian Portuguese. It is not in the
`score()` registry, because it needs the text and the lects. It needs the `lect` extra.

```python
from speechonnxmetrics.lect_fidelity import LectFidelity

scorer = LectFidelity(("pt-PT", "pt-BR"))          # Allosaurus, orthography2ipa
result = scorer.score("clip.wav", "Os meninos partiram mais cedo.")
if result is not None:
    print(result.shares)                 # percent of sites read as pt-PT, pt-BR, neither
    print(result.shares_by_level)        # the same, pre-lexical and post-lexical apart
    print(result.log_likelihood_ratio, result.decision)
    for row in result.report():
        print(row["class"], row["context"], row["expected.pt-PT"], row["expected.pt-BR"], row["realised"], row["reading"])

pooled = scorer.pool([scorer.score(p, t) for p, t in clips_of_one_voice])
print(pooled.duration, pooled.posterior, pooled.shares)
```

The metric works in three steps.

1. **Expected phones.** A provider gives each lect's readings of each word. The default
   is `orthography2ipa`; `provider="tugaphone"` adds number verbalisation, homograph
   marking and the `tugalex` lexicon, and needs the `lect-tugaphone` extra. A provider is
   a small interface (`PhoneProvider.candidates(text, lect)`), so another phonemiser can
   take its place. A word's candidates are its reading in the sentence, its reading in
   isolation and the spec's free variants, so an optional process is a licensed reading
   rather than a miss where the spec licenses it. European word-final e is licensed to
   drop only before a word that begins with a vowel (`que o` read `k o`); before a
   consonant or at the end of an utterance the specs keep it, so a final e missing there
   is read as neither lect.
2. **Sites.** The two lects' readings are aligned, and a one-segment difference becomes a
   *site* only in a documented class: coda s, unstressed e, unstressed o, t and d before
   i, and coda l. Differences that are fused with a neighbouring difference (final
   `-de` read `dɨ` against `d͡ʒi`), sites whose candidate sets overlap, and differences
   outside the classes are excluded and counted. Unstressed reduction is pre-lexical;
   coda s, coda l and affrication are post-lexical, and the two groups are reported apart.
3. **Readings.** A phone recogniser writes the audio as IPA; `allosaurus` is the default
   backend. Each site is read as the lect whose candidates hold the nearer realisation
   by phonetic feature distance, or as neither. A deleted segment counts only when the
   phones around it were heard. A clip with fewer than 4 realised phones, or fewer than
   0.4 of the expected count, returns `None`, so silence gives no verdict. So does a clip
   with more than twice the expected count, such as a short clip repeated for minutes;
   the most any measured CLUL clip realised, with either backend, was 1.77 times its
   expected count.

The per-clip figures are the shares and a log-likelihood ratio summed over sites, with
the reliability of each site class measured on real speech of both lects. The table
ships in `speechonnxmetrics/data/lect_calibration.json`, one entry per provider and
backend, with the corpora, sizes and package versions it was measured on, and it serves
either lect order: `("pt-BR", "pt-PT")` gives the same verdict as `("pt-PT", "pt-BR")`,
with the log-likelihood ratio in favour of the lect named first. A class whose
readings do not differ between the lects at the 5% level carries no weight; unstressed o
has no sites at all, because the Brazilian spec licenses `u` there too. A posterior over
the lects is given only for pooled clips, with the pooled duration beside it: a clip of a
few seconds holds two to seven sites, too few for one.

#### Measured on real speech

The shipped calibration and the main measurement come from one corpus holding both
lects, so that the corpus cannot be what separates them: the CLUL corpus "Spoken
Portuguese: Geographical and Social Varieties", sentence clips of interviews recorded
in Portugal and Brazil (`Jarbas/SpokenPortugueseGeographicalSocialVarieties_splits`,
MIT), split by recording so that no recording is on both sides. compare-accents-pt, one
paragraph read by each of twenty speakers, is a second test. The 37 training and 13
held-out recordings are named in `evaluation/lect_fidelity/clul-split.json`.

| part | recordings or speakers | clips scored, pt-PT / pt-BR | speech, pt-PT / pt-BR |
|---|---|---|---|
| CLUL training recordings, calibration | 37 recordings, 22 pt-PT and 15 pt-BR | 1,058 / 888 | 102.5 / 74.4 min |
| CLUL held-out recordings, evaluation | 13 recordings, 8 pt-PT and 5 pt-BR | 375 / 333 | 41.2 / 22.5 min |
| compare-accents-pt, evaluation | 20 speakers, 6 pt-PT and 14 pt-BR | 6 / 14 | one paragraph each |

Of the CLUL clips, those without a site and those with too little realised speech give
no result and are not counted: 567 of 2,513 in the training part and 231 of 939 in the
held-out part, the latter mostly short or overlapping turns. The counts above are
Allosaurus's, decoded with the restriction the results below use. The wav2vec2 model
runs several times slower on CPU, so its rows use six training clips and up to 31
held-out clips per recording, 194 and 337 of them scored; the clips are named in
`evaluation/lect_fidelity/wav2vec2-clips.json`.

Results with the default `orthography2ipa` provider, decoding restricted to the union
of the two lects' phones, which is how the shipped calibration was measured:

| backend | CLUL held-out, clip AUC | CLUL clips right, pt-PT / pt-BR | CLUL recordings right | compare-accents-pt, AUC | speakers right, pt-PT / pt-BR |
|---|---|---|---|---|---|
| `allosaurus` | 0.70 (0.66–0.74) | 237 of 375 / 224 of 333 | 12 of 13 | 1.00 | 6 of 6 / 13 of 14 |
| `wav2vec2_espeak` | 0.78 (0.73–0.83) | 157 of 215 / 84 of 122 | 13 of 13 | 0.99 (0.93–1.00) | 5 of 6 / 14 of 14 |

Decoding without the restriction, the CLUL clip AUC is 0.69 (0.66–0.73) for
`allosaurus` and 0.59 (0.54–0.64) for `wav2vec2_espeak`. With `provider="tugaphone"`
the restricted CLUL AUCs are 0.69 (0.65–0.72) and 0.81 (0.76–0.86).

AUC is that of the clip log-likelihood ratio, with a stratified bootstrap 95% interval.
Clip accuracy is at the calibrated decision point. A recording or speaker is decided by
pooling all its clips, 2.3 to 11 minutes per CLUL recording. On the CLUL held-out part
the Allosaurus AUC stays between 0.66 and 0.76 in every duration bin from under 3 to
over 10 seconds, so clip length does not carry it.

A single clip of a few seconds holds two to seven sites, and the recogniser often misses
the vowel or the sibilant there, so one clip is weak evidence; a whole recording of one
speaker is classified reliably.

`wav2vec2_espeak` learned Portuguese from Common Voice clips labelled with espeak-ng's
European voice, so it tends to write European phones for Brazilian speech. Its CLUL rows
rest on a calibration of 194 clips, and the unrestricted 0.59 may come from that small
calibration rather than from the model: unrestricted, only one site class passes the
calibration gate there, and a leave-one-recording-out cross-validation within the
training recordings gives a clip AUC of 0.63 on those 194 clips but 0.77 on another
draw of 332 clips, ten per recording. Restricted, the same cross-validation gives 0.80
and 0.78, in line with the held-out 0.78. On the cross-corpus calibration below no site
class passes the gate even restricted, and it gives no evidence at all. It is not the
default.

A second measurement takes each lect from a different corpus and is confounded by
corpus, channel, text domain and length: pt-PT from Speech-MASSIVE pt-PT (read
assistant commands, CC-BY-NC-SA-4.0, used for evaluation only and not redistributed)
against pt-BR from the FLEURS pt_br test split (read news, CC-BY-4.0), calibrated on
EuroSpeech Portugal validation (parliament, six sessions) and the FLEURS pt_br dev split.
Clip duration alone separates its two evaluation corpora with an AUC of 0.99, and the
duration bins barely overlap, so its figures say little about lect:

| backend | clip AUC | clips right, pt-PT / pt-BR | pools of 30 s right, pt-PT / pt-BR |
|---|---|---|---|
| `allosaurus` | 0.90 (0.88–0.92) | 236 of 343 / 300 of 348 | 40 of 42 / 116 of 121 |
| `wav2vec2_espeak` | 0.50, no usable site class | 0 of 128 / 149 of 149 | 0 of 15 / 54 of 54 |

**To add a language pair**, the site classes must be documented for it, as
`SITE_CLASSES` in `speechonnxmetrics/lect_fidelity/sites.py` documents them for
Portuguese, and a calibration must be measured on real speech of both lects with
`evaluation/lect_fidelity/`. The scorer refuses a pair without classes.

The ONNX exports, their pinned revisions and licences are in
[`docs/models.md`](docs/models.md); the export scripts are in
[`conversion/`](conversion/).

The lect-fidelity tests run without the real models: two one-layer ONNX graphs with
the recognisers' shapes exercise each backend's whole path, and Allosaurus's own feature
model output is stored for the frontend. The Allosaurus parity tests download its 44 MB
export from the pinned revision. The wav2vec2 parity test needs the 1.26 GB export and
runs only when `SPEECHONNXMETRICS_LECT_MODELS` names a directory holding
`wav2vec2_xlsr53_espeak_cv_ft.onnx`. The twelve synthesised test clips come from Piper
voices fine-tuned from a voice trained on research-only data, so whether they may be
redistributed freely is unclear; their manifest records the lineage.

Sample-rate handling is automatic: the base resamples input to each model's native rate
(UTMOS/DNSMOS 16 kHz, SIGMOS 48 kHz, STOI analysis at 10 kHz), and NISQA is
rate-adaptive and never resamples. See [`docs/metrics.md`](docs/metrics.md) for the
per-metric detail and paper citations, and [`docs/models.md`](docs/models.md) for the
ONNX models and their licences.

## GPU inference

The ONNX-backed metrics (the MOS predictors, speaker similarity) run on
`CPUExecutionProvider` by default. To run them on GPU, install `onnxruntime-gpu`
(instead of, or alongside, `onnxruntime`) with the matching CUDA/cuDNN setup, then
pick one of two knobs:

* **`SPEECHONNXMETRICS_PROVIDERS`** — a comma-separated onnxruntime provider list,
  e.g. `SPEECHONNXMETRICS_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider`. This
  is the knob that matters for batch pipelines and the CLI, which have no
  `providers=` argument to thread through — set the env var once and every ONNX
  session in the process picks it up with no code change.
* **`providers=`** — pass it explicitly to `score()`/`score_batch()`, or to an
  `OnnxMetric` subclass's constructor, when a given call needs providers different
  from the process default. An explicit `providers=` always wins over the env var.

Either way, a provider that onnxruntime does not have built in (or that fails to
initialize) is silently dropped rather than raised: the requested list is intersected
with `onnxruntime.get_available_providers()`, and `CPUExecutionProvider` is always
kept as the final fallback so scoring never crashes for lack of a GPU.

## CLI

```
$ speechonnxmetrics --help
usage: speechonnxmetrics [-h] [--version] {score,list} ...

positional arguments:
  {score,list}
    score       score one or more audio files
    list        list available metrics
```

```
$ speechonnxmetrics score --help
usage: speechonnxmetrics score [-h] [--ref REF] --metrics METRICS [--sr SR]
                               [--json]
                               audio [audio ...]

positional arguments:
  audio              degraded audio file(s) to score

options:
  --ref REF          reference audio file (required for intrusive metrics)
  --metrics METRICS  comma-separated metric names
  --sr SR            sample rate hint for raw input
  --json             emit JSON instead of a table
```

Real invocations:

```
$ speechonnxmetrics score test/fixtures/audio/facodec_aria.wav \
      --ref test/fixtures/audio/source.wav --metrics stoi,mcd,si_sdr
audio                                 mcd                 si_sdr               stoi
test/fixtures/audio/facodec_aria.wav  10.459728433678961  -26.937894650414812  0.6620030195244008

$ speechonnxmetrics list
name         kind   intrusive  requires_download
cer          text   True       False
dnsmos       audio  False      True
...
```

Full reference in [`docs/cli.md`](docs/cli.md).

## Licence

The package itself is **Apache-2.0**. Model weights carry their own licences:

| licence | metrics | commercial use |
|---|---|---|
| MIT | `dnsmos`, `dnsmos_p808`, `sigmos`, `utmos` | permitted |
| **CC BY-NC-SA 4.0 (NonCommercial)** | **`nisqa`** | **forbidden** |
| Apache-2.0 | `lect_fidelity` with `wav2vec2_espeak` | permitted |
| GPL-3.0 | `lect_fidelity` with `allosaurus`, the default backend | permitted; distributing the weights or a work that includes them carries the GPL's obligations |

**`nisqa` is the one NonCommercial caveat.** `lect_fidelity` with its default backend
uses GPL-3.0 weights: commercial use is allowed, and distribution brings the GPL's
copyleft terms. Every other metric is safe for commercial use. The package makes no choice for you. It exposes the metric
and states the terms, and selecting it is your call. Full per-model breakdown in
[`docs/models.md`](docs/models.md).

Models are grouped in the HuggingFace collection **speechonnxmetrics models** under the
`TigreGotico` org.

## Not provided (on purpose)

- **PESQ**: ITU-T P.862 licensing is incompatible with an open, pip-installable
  package, and neural MOS predictors supersede it. Use a dedicated PESQ package under
  your own licence review if you need it.
- **UTMOSv2**: its published score is an ensemble over five folds times five random 3 s
  crops, so any single-fold single-crop export would be a *different* estimator, not an
  approximation of the published numbers.

## Docs

- [`docs/index.md`](docs/index.md): orientation
- [`docs/metrics.md`](docs/metrics.md): every metric family in detail
- [`docs/usage.md`](docs/usage.md): `score` vs `score_batch`, custom metrics, caching
- [`docs/cli.md`](docs/cli.md): CLI reference
- [`docs/models.md`](docs/models.md): ONNX models, HF repos and licences
