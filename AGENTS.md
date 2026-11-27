# AGENTS.md

speechonnxmetrics is a library of speech evaluation metrics on numpy and onnxruntime:
no-reference MOS predictors, reference-based signal metrics, ASR text metrics, speaker
metrics and `lect_fidelity`.

## Repository map

| Path | Holds |
| :--- | :--- |
| `speechonnxmetrics/api.py` | `score` and `score_batch` |
| `speechonnxmetrics/registry.py` | The metric registry and `list_metrics` |
| `speechonnxmetrics/base.py` | `Metric`, `OnnxMetric`, `ModelEntry`, execution providers |
| `speechonnxmetrics/mos/` | UTMOS, DNSMOS, SIGMOS, NISQA |
| `speechonnxmetrics/intrusive/` | STOI, SI-SDR, MCD, pitch and spectral metrics |
| `speechonnxmetrics/asr/` | WER family and normalizers |
| `speechonnxmetrics/speaker/` | Speaker similarity, EER, minDCF |
| `speechonnxmetrics/lect_fidelity/` | Lect fidelity, with the calibration in `data/lect_calibration.json` |
| `speechonnxmetrics/_dsp/` | Audio loading, resampling, STFT, mel, pitch, DTW |
| `conversion/` | Model export scripts (`export` extra) |
| `evaluation/lect_fidelity/` | Scripts that measure the lect calibration |
| `docs/` | The documentation. `docs/img/plot_*.py` make the figures |
| `test/` | The test suite. `test/test_docs.py` checks the docs |

## Setup and tests

```bash
uv venv && uv pip install --prerelease=allow -e ".[test,audio,speaker]"
python -m pytest -q test -m "not models"     # offline: no model downloads
python -m pytest -q test/test_docs.py        # docs examples, links, figure data
```

## Rules the code, the tests or the docs enforce

- The runtime imports only numpy, onnxruntime and huggingface_hub. `scipy` is a test
  oracle, and `torch` is for `conversion/` only.
- `import speechonnxmetrics` opens no network connection and makes no onnxruntime
  session.
- Each `ModelEntry` pins a commit SHA as its revision.
- Tests that download models carry the `models` mark.
- Each `print(...)  # value` line in a docs `python` block must print that value.
  `test/test_docs.py` runs it.
- A figure JSON in `docs/img/` must equal a fresh run of its script with
  `DOCS_FIGURES_DATA_ONLY=1`. After a change to the calibration table, run
  `python docs/img/plot_lect_auc.py` and update the table in `docs/lect-fidelity.md`.
