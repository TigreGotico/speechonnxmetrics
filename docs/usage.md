# Usage
<!-- agent-hint
covers: score and score_batch, the result shape, text metrics, custom metrics, the model cache and GPU providers
source: speechonnxmetrics/api.py, speechonnxmetrics/registry.py, speechonnxmetrics/base.py, speechonnxmetrics/resolver.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "usage.md"
see: cli.md
-->

This page shows the Python API: how to score one clip or a batch, how to read the
result, and how to add your own metric.

## Example

```python
import speechonnxmetrics as s

ref = "test/fixtures/audio/source.wav"
deg = "test/fixtures/audio/facodec_aria.wav"
print(s.score(deg, ["stoi"], ref=ref))  # {'stoi': 0.6620030195244008}
print(s.score_batch([deg, deg], ["stoi"], refs=[ref, ref]))  # [{'stoi': 0.6620030195244008}, {'stoi': 0.6620030195244008}]
```

## `score` and `score_batch`

```python
score(audio, metrics, ref=None, sr=None, providers=None) -> dict[str, Any]
score_batch(audios, metrics, refs=None, sr=None, providers=None) -> list[dict[str, Any]]
```

`score` calls `score_batch` with one item and returns its result.

- `audio`: a path, bytes or a numpy array. `sr` gives the sample rate of a raw array.
- `metrics`: the names of registered audio metrics. The names are not case-sensitive.
  An empty list, a duplicate name or the name of a text metric raises `ValueError`.
- `ref` and `refs`: the reference audio. An intrusive metric needs it. `refs` has the
  same length as `audios`.
- `providers`: the onnxruntime execution providers for this call. See
  [GPU inference](#gpu-inference).

`score_batch` resolves each metric name once. An ONNX metric builds its onnxruntime
session on the first item, and the next items use the same session.

A metric that fails on one item does not stop the batch. The value for that item is
`None`, and the error message goes under `_errors`:

```python
print(s.score_batch([deg, b"not audio"], ["stoi"], refs=[ref, ref])[1]["stoi"])  # None
```

When an item does not load, each metric of that item is `None` and records the load
error. A no-reference metric refuses digital silence with `SilentAudioError`.

## Result shape

The result is a flat dict. A metric that gives one value maps its name to the value. A
metric that gives several values adds a dotted name for each:

| Metric | Keys |
| :--- | :--- |
| `dnsmos` | `dnsmos.sig`, `dnsmos.bak`, `dnsmos.ovrl` |
| `nisqa` | `nisqa.mos`, `nisqa.noi`, `nisqa.dis`, `nisqa.col`, `nisqa.loud` |
| `sigmos` | `sigmos.col`, `sigmos.disc`, `sigmos.loud`, `sigmos.noise`, `sigmos.reverb`, `sigmos.sig`, `sigmos.ovrl` |

## Text metrics

The registry holds the text metrics, so `list_metrics()` and the CLI show them. But
`score()` runs only audio metrics. Call the text metrics from `speechonnxmetrics.asr`:

```python
from speechonnxmetrics import asr

print(asr.wer("the quick brown fox", "the quick brown box"))  # 0.25
print(asr.cer("the quick brown fox", "the quick brown box"))  # 0.05263157894736842
try:
    s.score(ref, ["wer"])
except ValueError as error:
    print(error)  # metric 'wer' is a text metric — call speechonnxmetrics.asr.wer directly
```

The [metrics reference](metrics.md#asr-based-text-metrics) lists the normalizer presets.

## Add a custom metric

A function or an object with the `Metric` call shape can go in the registry. Then
`score`, `score_batch`, `list_metrics` and the CLI use it:

```python
import numpy as np
from speechonnxmetrics import RegistryEntry, register

def rms_dbfs(deg, sr, *, ref=None, ref_sr=None) -> float:
    x = np.asarray(deg, dtype=np.float64)
    return 20.0 * np.log10(float(np.sqrt(np.mean(x ** 2))) or 1e-12)

register(RegistryEntry(
    name="rms_dbfs", kind="audio", intrusive=False, requires_download=False,
    fn=rms_dbfs, range=None, higher_is_better=True,
    description="Signal RMS level in dBFS",
))
print(s.score(ref, ["rms_dbfs"])["rms_dbfs"])  # -20.73763214588542
```

The `Metric` protocol in `speechonnxmetrics/base.py`:

```python
name: str
sample_rate: int | None      # native rate, or None when the metric takes any rate
intrusive: bool              # True when the metric needs a reference
range: tuple[float, float] | None
higher_is_better: bool
def __call__(self, deg, sr, *, ref=None, ref_sr=None) -> float | dict[str, float]: ...
```

For an ONNX metric, subclass `OnnxMetric`. Set the class attributes, give a
`ModelEntry`, and write `_frontend` (audio to model input) and `_postprocess` (model
output to score). The base class makes the session on the first call, resamples to
`ModelEntry.sample_rate` and downloads the model. The classes in `speechonnxmetrics/mos/`
are examples, and [`examples/custom_metric.py`](../examples/custom_metric.py) runs one.

## Model cache and offline use

The import opens no network connection. A MOS metric downloads nothing when you make
it. It makes its onnxruntime session, and downloads the model, on its first call.

- The weights go in the Hugging Face cache layout under
  `$XDG_DATA_HOME/speechonnxmetrics/hf`, or `~/.local/share/speechonnxmetrics/hf` when
  `XDG_DATA_HOME` is not set.
- The download goes through `huggingface_hub`, so `HF_HUB_OFFLINE` applies to it. The
  package sets its own cache directory, so `HF_HOME` does not move the weights.
- Each model is pinned to a commit. [Models](models.md) lists the repos and revisions.
- The metrics with `requires_download=True` download a model: the five MOS
  predictors. The intrusive and text metrics never download.
- Each `OnnxMetric` has `.model_info` with `repo_id`, `filename` and `revision`. Record
  it next to a score to name the weights that made it.

To fill the cache before an offline run, call each model once:

```python
downloadable = [e.name for e in s.list_metrics(requires_download=True)]
s.score("test/fixtures/audio/source.wav", downloadable)
```

## GPU inference

The ONNX metrics run on `CPUExecutionProvider` by default. To use a GPU, install
`onnxruntime-gpu` with a matching CUDA and cuDNN, then set the providers in one of two
ways:

- Set `SPEECHONNXMETRICS_PROVIDERS` to a comma-separated list, for example
  `CUDAExecutionProvider,CPUExecutionProvider`. Each ONNX session in the process reads
  it. Use it for the CLI and for code that you cannot change.
- Pass `providers=` to `score()`, `score_batch()` or an `OnnxMetric` constructor. It
  has priority over the environment variable. The numpy metrics ignore it.

The package keeps only the providers that `onnxruntime.get_available_providers()`
lists. When the list does not hold `CPUExecutionProvider`, the package adds it at
the end. A provider that is not
available does not raise an error.

---
[← Choosing metrics](choosing.md) · [Home](index.md) · [CLI →](cli.md)
