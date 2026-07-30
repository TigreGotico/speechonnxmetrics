# speechonnxmetrics documentation

Unified speech-evaluation metrics on numpy + onnxruntime. No torch at runtime. Neural
MOS predictors run as exported ONNX graphs whose weights download from public
HuggingFace repos on first use.

## Learning path (zero to hero)

New here? Follow this order. Each page builds on the last.

1. **[concepts.md](concepts.md)**: the mental model *before* the metric list, covering
   intrusive vs non-intrusive, what MOS is and why we predict it, the
   intelligibility/quality/similarity axes, sample-rate and alignment gotchas, and the
   golden rule.
2. **[choosing.md](choosing.md)**: a task-to-metric decision guide covering what to
   report for TTS, enhancement, voice conversion, ASR, separation and verification, and
   when *not* to use a metric.
3. **[metric-guides/](metric-guides/)**: one teaching page per metric, covering verified
   history, what it measures, range/direction, when to use it, and the exact call in
   this library.
4. **[usage.md](usage.md)** and **[cli.md](cli.md)**: the API and command line once you
   know what you want to compute.

The **[examples/tutorials/](../examples/tutorials/)** series is the hands-on companion:
seven numbered, runnable, heavily-narrated scripts that walk the same path with real audio.

## Where to go (reference)

- **[metrics.md](metrics.md)**: reference for every metric family, covering what each
  measures, its range and direction, reference requirement, sample-rate handling, paper
  citation, and parity number where measured.
- **[usage.md](usage.md)**: the Python API, covering `score` vs `score_batch`, the
  flat-dict output shape, audio vs text metrics, registering a custom metric, and the
  model cache / offline behaviour.
- **[cli.md](cli.md)**: the `speechonnxmetrics` command-line tool.
- **[models.md](models.md)**: the ONNX models each metric downloads, their HuggingFace
  repos, upstream sources, and licences (including the one NonCommercial caveat).

## Runnable examples

Every script under [`../examples/`](../examples/) runs green against the bundled
fixture audio in `test/fixtures/audio/`:

- `quickstart.py`: one file, one no-reference metric
- `reference_metrics.py`: STOI / MCD / SI-SDR on a degraded/reference pair
- `batch_eval.py`: `score_batch` with per-item failure isolation
- `tts_mos.py`: UTMOS + DNSMOS + NISQA, the canonical TTS-eval case
- `asr_wer.py`: WER/CER with normalizer presets (including LEGACY)
- `custom_metric.py`: register your own metric through the protocol

## The 30-second picture

```python
import speechonnxmetrics as s

s.list_metrics()                       # every registered metric + its properties
s.score(audio, ["utmos", "dnsmos"])    # no-reference, dict flattened to utmos, dnsmos.sig...
s.score(deg, ["stoi"], ref=clean)      # intrusive metrics need ref=
s.score_batch(audios, metrics, refs=refs)  # eval-loop hot path, per-item isolation
```

Text metrics (WER/CER/MER/WIL/WIP) compare strings, so they are not dispatched by
`score()`. Call them from `speechonnxmetrics.asr` directly.
