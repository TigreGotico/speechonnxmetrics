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
| `export` | `torch`, `onnx` | maintainer-only offline model conversion |
| `test` | `pytest`, `pytest-cov`, `scipy` | running the test suite (scipy is a test-only oracle) |

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

Sample-rate handling is automatic: the base resamples input to each model's native rate
(UTMOS/DNSMOS 16 kHz, SIGMOS 48 kHz, STOI analysis at 10 kHz), and NISQA is
rate-adaptive and never resamples. See [`docs/metrics.md`](docs/metrics.md) for the
per-metric detail and paper citations, and [`docs/models.md`](docs/models.md) for the
ONNX models and their licences.

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

**`nisqa` is the one caveat: its weights are NonCommercial.** Every other metric is
safe for commercial use. The package makes no choice for you. It exposes the metric
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
