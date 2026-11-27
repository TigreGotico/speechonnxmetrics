# Install
<!-- agent-hint
covers: how to install the package and what each optional extra adds
source: pyproject.toml, speechonnxmetrics/_dsp/audio.py, speechonnxmetrics/resolver.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "install.md"
see: concepts.md
-->

This page tells you how to install speechonnxmetrics and which extra to add for each
task.

## Example

```bash
uv pip install --prerelease=allow "speechonnxmetrics[audio,speaker]"
```

```python
import speechonnxmetrics as s

print(len(s.list_metrics()))  # 21
```

## Steps

1. Install the base package. It needs Python 3.11 or later, and it pulls in `numpy`,
   `onnxruntime>=1.17` and `huggingface_hub`.
2. Add the extras for your task from the table below.
3. Import the package. The import opens no network connection and loads no
   onnxruntime session. A MOS metric downloads its weights on its first call.

The base install decodes WAV files with the Python `wave` module. Install the `audio`
extra to decode other formats, such as MP3, FLAC or OGG.

## Extras

| Extra | Pulls in | Use it for |
| :--- | :--- | :--- |
| `audio` | `soundfile>=0.11` | Input that is not WAV |
| `speaker` | `speakeronnx` | `speechonnxmetrics.speaker`: speaker similarity, EER and minDCF |
| `asr` | `onnx-asr` | An ASR model to make hypotheses. The package does not import it. WER scoring needs no extra. |
| `vad` | `vadonnx` | Voice activity detection. The package does not import it. |
| `lect` | `orthography2ipa[portuguese]>=7.124.14a2`, `scriptconv>=0.0.3` | Expected phones for `lect_fidelity` |
| `lect-tugaphone` | the `lect` extra and `tugaphone>=1.4.1a2` | The `tugaphone` phone provider for `lect_fidelity` |
| `export` | `torch`, `onnx` | The model export scripts in `conversion/`. Inference never uses it. |
| `test` | `pytest`, `pytest-cov`, `scipy`, the `lect` and `tugaphone` packages | The test suite. `scipy` is a test oracle only. |

`pyproject.toml` holds the exact pins.

---
[Home](index.md) · [Concepts →](concepts.md)
