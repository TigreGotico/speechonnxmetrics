# speechonnxmetrics
<!-- agent-hint
covers: what the library does, how to install it, one quickstart, links into docs/
source: speechonnxmetrics/api.py, speechonnxmetrics/registry.py, pyproject.toml
generated: no
verify: python -m pytest -q test/test_docs.py -k "README.md and not guides"
see: docs/index.md
-->

speechonnxmetrics scores speech with 21 metrics behind one API: no-reference MOS
predictors, reference-based signal metrics, ASR text metrics and speaker similarity.
The runtime needs only numpy, onnxruntime and huggingface_hub. It does not use torch.
The MOS predictors run as ONNX graphs, and their weights download on the first call.

## Install

```bash
uv pip install --prerelease=allow speechonnxmetrics
```

The optional extras (`audio`, `speaker`, `asr`, `vad`, `lect`) are in
[docs/install.md](docs/install.md).

## Quickstart

The reference-based metrics run in numpy and download no model:

```python
from urllib.request import urlretrieve
import speechonnxmetrics as s

clips = "https://raw.githubusercontent.com/TigreGotico/speechonnxmetrics/4210419/test/fixtures/audio/"
ref, _ = urlretrieve(clips + "source.wav")
deg, _ = urlretrieve(clips + "facodec_aria.wav")
print(s.score(deg, ["stoi", "mcd", "si_sdr"], ref=ref))  # {'stoi': 0.6620030195244008, 'mcd': 10.45972843367896, 'si_sdr': -26.937894650414812}
```

A no-reference MOS predictor needs no reference. UTMOS downloads its weights on the
first call:

```python
s.score(ref, ["utmos"])
```

## Documentation

| Page | What you get there |
| :--- | :--- |
| [Docs index](docs/index.md) | The reading order and a one-line summary of each page |
| [Install](docs/install.md) | The extras and what each one adds |
| [Concepts](docs/concepts.md) | Reference against no-reference metrics, MOS, sample rates |
| [Choosing metrics](docs/choosing.md) | Which metrics to report for each task |
| [Usage](docs/usage.md) | `score`, `score_batch`, custom metrics, the model cache, GPU |
| [CLI](docs/cli.md) | The `speechonnxmetrics` command |
| [Metrics reference](docs/metrics.md) | Range, direction and source of each metric |
| [Lect fidelity](docs/lect-fidelity.md) | European against Brazilian Portuguese, measured on real speech |
| [Models and licences](docs/models.md) | Each ONNX model, its pinned revision and its licence |

The package is Apache-2.0. The `nisqa` weights are CC BY-NC-SA 4.0 and forbid
commercial use. The other model licences are in [docs/models.md](docs/models.md).
