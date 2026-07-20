# speechmetrics

Unified **speech evaluation metrics** — no-reference MOS, intrusive (reference-based)
signal metrics, ASR-based metrics, and speaker similarity — on top of numpy and
onnxruntime.

**Nothing in this repo is implemented yet.** Only the packaging, CI wiring, and shared
numpy DSP primitives (STFT/ISTFT, Kaldi-compatible log-mel filterbank, kaiser-window
resampling) exist so far. Every metric below is *planned*, not shipped — do not assume
any of them work until this note is removed.

```bash
pip install speechmetrics
```

Inference is intended to run entirely on onnxruntime, with no torch at runtime. Weights
will download on first use and cache under `~/.local/share/speechmetrics`, pinned by
revision, via the same `resolver` pattern as `audiosronnx`.

## Planned scope

### No-reference MOS (`speechmetrics.mos`)
- UTMOS
- UTMOSv2
- NISQA
- SIGMOS
- DNSMOS P.808 / P.835

### Intrusive metrics (`speechmetrics.intrusive`)
Reference-based signal metrics, compared against a clean/target signal:
- STOI / ESTOI
- SI-SDR / SDR / SNR
- MCD (mel-cepstral distortion)
- log-F0 RMSE
- V/UV (voiced/unvoiced) error
- LSD (log-spectral distance)
- MSD (mel-spectral distortion)
- Mel-L1

### ASR-based metrics (`speechmetrics.asr`)
- WER, CER, MER, WIL, WIP

### Speaker similarity (`speechmetrics.speaker`)
- Cosine similarity between speaker embeddings (backed by `speakeronnx`).

### Explicitly out of scope
**PESQ is intentionally not provided.** ITU-T P.862 licensing terms are incompatible
with shipping it in an open-source, pip-installable package; use a dedicated PESQ
package under your own license review if you need it.

## Optional dependencies

Runtime dependencies are `numpy`, `onnxruntime`, and `huggingface_hub` only.

| Extra | Adds | For |
|-------|------|-----|
| `test` | `pytest`, `pytest-cov`, `scipy` | running the test suite (`scipy` is a test-only oracle, never a runtime dependency) |
| `export` | `torch`, `onnx` | offline model conversion scripts only, not runtime inference |
| `speaker` | `speakeronnx` | speaker-similarity metrics |
| `asr` | `onnx-asr` | ASR-based metrics |
| `vad` | `vadonnx` | voice-activity gating ahead of metrics that need it |

## Development

```bash
pip install -e '.[test]'
pytest test/ -q
```
