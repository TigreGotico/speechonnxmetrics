# CLI reference
<!-- agent-hint
covers: the speechonnxmetrics command, its score and list subcommands, flags and exit status
source: speechonnxmetrics/cli.py, pyproject.toml [project.scripts]
generated: no
verify: diff <(speechonnxmetrics score --help) <(sed -n '/^\$ speechonnxmetrics score --help/,/^```/p' docs/cli.md | sed '1d;$d')
see: metrics.md
-->

This page shows the `speechonnxmetrics` command. The install adds it as a console
script with the entry point `speechonnxmetrics.cli:main`. You can also run
`python -m speechonnxmetrics.cli`. The command has two subcommands, `score` and
`list`. The examples use the fixture clips in the repository.

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
audio                                 mcd                si_sdr               stoi
test/fixtures/audio/facodec_aria.wav  10.45972843367896  -26.937894650414812  0.6620030195244008
```

JSON output:

```
$ speechonnxmetrics score test/fixtures/audio/facodec_aria.wav \
      --ref test/fixtures/audio/source.wav --metrics stoi --json
{
  "test/fixtures/audio/facodec_aria.wav": {
    "stoi": 0.6620030195244008
  }
}
```

A no-reference MOS metric needs no `--ref`. It downloads its model on the first run:

```
$ speechonnxmetrics score test/fixtures/audio/source.wav --metrics utmos --json
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
[← Usage](usage.md) · [Home](index.md) · [Metrics reference →](metrics.md)
