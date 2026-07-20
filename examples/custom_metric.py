"""Register a custom metric and score with it through the public API.

Any object matching the ``Metric`` protocol — the class attributes ``name``,
``sample_rate``, ``intrusive``, ``range``, ``higher_is_better`` and a
``__call__(deg, sr, *, ref=None, ref_sr=None)`` returning a float or dict — can be
wrapped in a ``RegistryEntry`` and registered. Once registered it is available to
``score``/``score_batch``, ``list_metrics`` and the CLI like any built-in.

    python examples/custom_metric.py
"""
import numpy as np

import speechonnxmetrics as s
from speechonnxmetrics import RegistryEntry, register


def rms_dbfs(deg, sr, *, ref=None, ref_sr=None) -> float:
    """A trivial no-reference metric: signal RMS level in dBFS."""
    x = np.asarray(deg, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x ** 2))) or 1e-12
    return 20.0 * np.log10(rms)


register(
    RegistryEntry(
        name="rms_dbfs",
        kind="audio",
        intrusive=False,
        requires_download=False,
        fn=rms_dbfs,
        range=None,
        higher_is_better=True,
        description="Signal RMS level in dBFS (demo custom metric)",
    )
)

print("rms_dbfs is now registered:", "rms_dbfs" in [e.name for e in s.list_metrics()])
print(s.score("test/fixtures/audio/source.wav", ["rms_dbfs", "utmos"]))
