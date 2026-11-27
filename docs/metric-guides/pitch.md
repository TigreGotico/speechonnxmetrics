# Log-F0 RMSE and voiced/unvoiced error
<!-- agent-hint
covers: what log_f0_rmse and vuv_error measure, their range, when to use them and their call
source: speechonnxmetrics/intrusive/, speechonnxmetrics/_dsp/pitch.py
generated: no
verify: python -m pytest -q test/test_docs.py -k "metric-guides/pitch.md"
see: spectral.md
-->

These two metrics compare the pitch of a clip with that of a reference. This page
covers `log_f0_rmse` and `vuv_error`.

## Example

```python
import speechonnxmetrics as s

ref = "test/fixtures/audio/source.wav"
deg = "test/fixtures/audio/facodec_aria.wav"
print(s.score(deg, ["log_f0_rmse", "vuv_error"], ref=ref))  # {'log_f0_rmse': 0.7277250911360245, 'vuv_error': 0.014084507042253521}
```

## History

These metrics have no single source paper. They are conventional prosody measures in
TTS and voice conversion evaluation, and papers report them next to MCD. They measure
the pitch, which MCD does not.

## What they measure

- `log_f0_rmse` is the root-mean-square error between the two F0 (pitch) contours, in
  log-Hz, over the frames where both signals are voiced. The log scale follows how
  listeners hear pitch, and high-pitched frames do not dominate the error.
- `vuv_error` is the voiced/unvoiced error rate: the fraction of frames where one signal
  is voiced and the other is not.

## Range and direction

`log_f0_rmse` has no bound, and lower is better. `vuv_error` is 0 to 1, and lower is
better. Both are intrusive, so pass `ref=`. The pitch tracker is in
`speechonnxmetrics/_dsp/pitch.py`.

## When to use it

- Use them next to MCD for TTS and voice conversion. MCD checks the timbre. These two
  check the intonation. This matters for expressive TTS and for voice conversion that
  changes pitch.
- Do not read them alone. Pitch tracking is noisy, and the estimates are unstable on
  speech that is almost all unvoiced, such as a whisper.

## Further reading

No single paper defines them. They go with [MCD](mcd.md).

---
[← MCD](mcd.md) · [Home](../index.md) · [Spectral distances →](spectral.md)
