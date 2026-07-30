# CLI reference

Installing the package exposes a `speechonnxmetrics` console script (entry point
`speechonnxmetrics.cli:main`). Without installing, run `python -m speechonnxmetrics.cli`.
The tool has two subcommands, `score` and `list`, described below with their exact
help text and real example invocations against the bundled fixture audio.

```
$ speechonnxmetrics --help
usage: speechonnxmetrics [-h] [--version] {score,list} ...

positional arguments:
  {score,list}
    score       score one or more audio files
    list        list available metrics

options:
  -h, --help    show this help message and exit
  --version     show program's version number and exit
```

## `score`

```
$ speechonnxmetrics score --help
usage: speechonnxmetrics score [-h] [--ref REF] --metrics METRICS [--sr SR]
                               [--json]
                               audio [audio ...]

positional arguments:
  audio              degraded audio file(s) to score

options:
  -h, --help         show this help message and exit
  --ref REF          reference audio file (required for intrusive metrics)
  --metrics METRICS  comma-separated metric names
  --sr SR            sample rate hint for raw input
  --json             emit JSON instead of a table
```

- `--metrics` takes comma-separated audio metric names. Text metrics are not scored here.
- `--ref` is required when any chosen metric is intrusive, and applies to every positional `audio` argument.
- Output is a table by default, one row per file, one column per flattened metric key.
- `--json` emits a `{path: result}` object instead of a table.
- Exit status is `1` on a resolution error, or when any item recorded a per-file `_errors` entry.
- On a partial error, successful rows still print and errors go to stderr.

### Examples

Intrusive metrics with a reference:

```
$ speechonnxmetrics score test/fixtures/audio/facodec_aria.wav \
      --ref test/fixtures/audio/source.wav --metrics stoi,mcd,si_sdr
audio                                 mcd                 si_sdr               stoi
test/fixtures/audio/facodec_aria.wav  10.459728433678961  -26.937894650414812  0.6620030195244008
```

No-reference MOS, JSON output (downloads models on first run):

```
$ speechonnxmetrics score test/fixtures/audio/source.wav --metrics utmos --json
{
  "test/fixtures/audio/source.wav": {
    "utmos": 4.411556243896484
  }
}
```

## `list`

```
$ speechonnxmetrics list --help
usage: speechonnxmetrics list [-h] [--json]

options:
  -h, --help  show this help message and exit
  --json      emit JSON instead of a table
```

```
$ speechonnxmetrics list
name         kind   intrusive  requires_download
cer          text   True       False
dnsmos       audio  False      True
dnsmos_p808  audio  False      True
estoi        audio  True       False
log_f0_rmse  audio  True       False
lsd          audio  True       False
mcd          audio  True       False
mel_l1       audio  True       False
mer          text   True       False
msd          audio  True       False
nisqa        audio  False      True
sdr          audio  True       False
si_sdr       audio  True       False
sigmos       audio  False      True
snr          audio  True       False
stoi         audio  True       False
utmos        audio  False      True
vuv_error    audio  True       False
wer          text   True       False
wil          text   True       False
wip          text   True       False
```

`--json` emits the same rows as an array of `{name, kind, intrusive, requires_download}`
objects.

---
[← Usage](usage.md) · [Home](index.md) · [Models →](models.md)
