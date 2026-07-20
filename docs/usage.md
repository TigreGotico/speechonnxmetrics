# Usage

## `score` vs `score_batch`

`score` handles one item; `score_batch` handles many. Both return the same per-item
flat dict — `score` is a thin wrapper over `score_batch([audio], ...)[0]`.

```python
import speechonnxmetrics as s

# single item
s.score("clip.wav", ["utmos", "dnsmos"])

# batch
s.score_batch(["a.wav", "b.wav"], metrics=["utmos"])
```

Signatures:

```python
score(audio, metrics, ref=None, sr=None) -> dict[str, Any]
score_batch(audios, metrics, refs=None, sr=None) -> list[dict[str, Any]]
```

- `audio`/`audios` — path, bytes, or a numpy array. `sr` is a sample-rate hint used when
  the input is a raw array or headerless.
- `metrics` — a sequence of registered **audio** metric names (case-insensitive).
  Duplicate names, an empty list, or a text-metric name raise `ValueError`.
- `ref`/`refs` — reference audio, required when any requested metric is intrusive.
  `refs` must match `audios` in length.

### Why `score_batch` is the hot path

Each metric name is resolved from the registry **once** and reused across every item.
For an ONNX-backed MOS metric that means the onnxruntime session is built on the first
item and every later item reuses it — you do not pay session setup per clip. Over an
eval of thousands of clips that is the difference between one model load and thousands.

It also isolates failures. A metric that raises on one item — a corrupt file, audio too
short for NISQA — records `None` for that item's value and the exception message under
`"_errors"`, and the batch keeps running. A single bad file cannot sink an hours-long
run.

```python
results = s.score_batch(["ok.wav", "corrupt.wav"], ["utmos"])
# results[1] == {'utmos': None, '_errors': {'utmos': '...'}}
```

If a whole item fails to load, every requested metric for that item is `None` with the
load error recorded.

## Output shape

Results are a **flat dict of floats**. Scalar metrics map name → value; dict-valued
metrics (multi-head models) are flattened with a dotted prefix:

```python
s.score("clip.wav", ["utmos", "dnsmos", "nisqa"])
# {
#   'utmos': 4.41,
#   'dnsmos.sig': 3.45, 'dnsmos.bak': 3.60, 'dnsmos.ovrl': 2.93,
#   'nisqa.mos': 4.74, 'nisqa.noi': 4.64, 'nisqa.dis': 4.78,
#   'nisqa.col': 4.34, 'nisqa.loud': 4.63,
# }
```

`sigmos` flattens to `sigmos.col`, `sigmos.disc`, `sigmos.loud`, `sigmos.noise`,
`sigmos.reverb`, `sigmos.sig`, `sigmos.ovrl`.

## Audio vs text metrics

The registry holds both, so `list_metrics()` and the CLI enumerate everything in one
place, but `score()`/`score_batch()` dispatch only `kind="audio"`. Passing a text metric
name (`wer`, `cer`, `mer`, `wil`, `wip`) to `score()` raises, telling you to call it
directly:

```python
from speechonnxmetrics import asr

asr.wer("the quick brown fox", "the quick brown box")          # 0.25
asr.cer("the quick brown fox", "the quick brown box")          # per-character
asr.compute("ref", "hyp")                                       # AsrMetrics with all rates
asr.wer(ref, hyp, normalizer=asr.STRICT)                        # opt-in normalization
```

See [metrics.md](metrics.md) for the normalizer presets and the LEGACY caveat.

## Registering a custom metric

Any object matching the `Metric` protocol can be registered and then used by
`score`/`score_batch`, `list_metrics` and the CLI. The protocol is:

```python
name: str
sample_rate: int | None      # native rate, or None for rate-agnostic
intrusive: bool              # True => a reference is required
range: tuple[float, float] | None
higher_is_better: bool
def __call__(self, deg, sr, *, ref=None, ref_sr=None) -> float | dict[str, float]: ...
```

Wrap it (or a plain function of the same call shape) in a `RegistryEntry` and register:

```python
import numpy as np
import speechonnxmetrics as s
from speechonnxmetrics import RegistryEntry, register

def rms_dbfs(deg, sr, *, ref=None, ref_sr=None) -> float:
    x = np.asarray(deg, dtype=np.float64)
    return 20.0 * np.log10(float(np.sqrt(np.mean(x ** 2))) or 1e-12)

register(RegistryEntry(
    name="rms_dbfs", kind="audio", intrusive=False, requires_download=False,
    fn=rms_dbfs, range=None, higher_is_better=True,
    description="Signal RMS level in dBFS",
))

s.score("clip.wav", ["rms_dbfs"])   # {'rms_dbfs': -20.73...}
```

For an ONNX-backed metric, subclass `OnnxMetric` instead: set the class attributes,
provide a `ModelEntry`, and implement `_frontend` (audio → input feed) and
`_postprocess` (outputs → score). The base handles lazy, thread-safe session creation,
resampling to `ModelEntry.sample_rate`, and download resolution — see
`speechonnxmetrics/mos/` for worked subclasses. A full runnable example is in
[`../examples/custom_metric.py`](../examples/custom_metric.py).

## Model cache and offline behaviour

`import speechonnxmetrics` touches neither the network nor onnxruntime. Constructing a
MOS metric also downloads nothing — the onnxruntime session, and therefore the download,
is created lazily on the **first call**.

- **Where weights land:** `$XDG_DATA_HOME/speechonnxmetrics` (i.e.
  `~/.local/share/speechonnxmetrics` by default), fetched via `huggingface_hub` from the
  per-model HF repos listed in [models.md](models.md), pinned by revision.
- **HF cache:** the download goes through `huggingface_hub`, so `HF_HOME` /
  `HF_HUB_OFFLINE` apply as usual for the fetch step.
- **Pre-fetch:** call the metric once (e.g. `s.score(fixture, ["utmos","dnsmos","nisqa","sigmos"])`)
  in a warm-up step to populate the cache before an offline run. After that, scoring runs
  fully offline.
- **Which metrics download:** exactly the ones with `requires_download=True` — the five
  MOS predictors. Every intrusive and text metric is pure numpy and never downloads.

```python
# pre-fetch every downloadable model, then run offline
downloadable = [e.name for e in s.list_metrics(requires_download=True)]
s.score("test/fixtures/audio/source.wav", downloadable)
```
