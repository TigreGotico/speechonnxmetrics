# speechonnxmetrics documentation
<!-- agent-hint
covers: the reading order of the docs and one line for each page
source: docs/
generated: no
verify: diff <(ls docs/*.md docs/metric-guides/README.md | grep -v index.md | sed 's|^docs/||' | sort) <(grep -o '](\([a-z-]*/\)\?[A-Za-z_-]*\.md)' docs/index.md | sed 's/^](//; s/)$//' | sort -u)
see: ../README.md
-->

speechonnxmetrics gives you 21 metrics to score speech audio and transcripts, on numpy
and onnxruntime only.

**21 metrics, one call.** `score()` runs the audio metrics. `speechonnxmetrics.asr`
runs the text metrics. Neither needs torch.

## Quickstart

```bash
uv pip install --prerelease=allow speechonnxmetrics
```

```python
import speechonnxmetrics as s

ref = "test/fixtures/audio/source.wav"
deg = "test/fixtures/audio/facodec_aria.wav"
print(s.score(deg, ["stoi", "mcd", "si_sdr"], ref=ref))  # {'stoi': 0.6620030195244008, 'mcd': 10.45972843367896, 'si_sdr': -26.937894650414812}
print(sorted(e.name for e in s.list_metrics(requires_download=True)))  # ['dnsmos', 'dnsmos_p808', 'nisqa', 'sigmos', 'utmos']
```

The paths point at the fixture clips in the repository. The second line lists the
metrics that download a model.

## Pages

| Page | What you get there |
| :--- | :--- |
| [Install](install.md) | The base install and each optional extra |
| [Concepts](concepts.md) | Reference against no-reference metrics, MOS, the three axes, sample rates |
| [Choosing metrics](choosing.md) | The metrics to report for TTS, enhancement, voice conversion, ASR, separation and verification |
| [Usage](usage.md) | `score` and `score_batch`, the output shape, text metrics, custom metrics, the model cache, GPU |
| [CLI](cli.md) | The `score` and `list` commands, with real output |
| [Metrics reference](metrics.md) | Range, direction, sample rate and paper of each metric |
| [Lect fidelity](lect-fidelity.md) | How the metric reads European against Brazilian Portuguese, and how well it does on real speech |
| [Models and licences](models.md) | Each ONNX model, its Hugging Face repo, pinned revision and licence |
| [Metric guides](metric-guides/README.md) | One page for each metric: history, use and the call in this library |

The scripts in [`examples/`](../examples/) and [`examples/tutorials/`](../examples/tutorials/)
run on the fixture clips in `test/fixtures/audio/`.
