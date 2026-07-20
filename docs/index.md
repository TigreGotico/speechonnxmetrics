# speechonnxmetrics documentation

Unified speech-evaluation metrics on numpy + onnxruntime. No torch at runtime; neural
MOS predictors run as exported ONNX graphs whose weights download from public
HuggingFace repos on first use.

## Where to go

- **[metrics.md](metrics.md)** — reference for every metric family: what each measures,
  its range and direction, reference requirement, sample-rate handling, paper citation,
  and parity number where measured.
- **[usage.md](usage.md)** — the Python API: `score` vs `score_batch`, the flat-dict
  output shape, audio vs text metrics, registering a custom metric, and the model cache
  / offline behaviour.
- **[cli.md](cli.md)** — the `speechonnxmetrics` command-line tool.
- **[models.md](models.md)** — the ONNX models each metric downloads, their HuggingFace
  repos, upstream sources, and licences (including the one NonCommercial caveat).

## Runnable examples

Every script under [`../examples/`](../examples/) runs green against the bundled
fixture audio in `test/fixtures/audio/`:

- `quickstart.py` — one file, one no-reference metric
- `reference_metrics.py` — STOI / MCD / SI-SDR on a degraded/reference pair
- `batch_eval.py` — `score_batch` with per-item failure isolation
- `tts_mos.py` — UTMOS + DNSMOS + NISQA, the canonical TTS-eval case
- `asr_wer.py` — WER/CER with normalizer presets (incl. LEGACY)
- `custom_metric.py` — register your own metric through the protocol

## The 30-second picture

```python
import speechonnxmetrics as s

s.list_metrics()                       # every registered metric + its properties
s.score(audio, ["utmos", "dnsmos"])    # no-reference, dict flattened to utmos, dnsmos.sig...
s.score(deg, ["stoi"], ref=clean)      # intrusive metrics need ref=
s.score_batch(audios, metrics, refs=refs)  # eval-loop hot path, per-item isolation
```

Text metrics (WER/CER/MER/WIL/WIP) compare strings, so they are not dispatched by
`score()` — call them from `speechonnxmetrics.asr` directly.
