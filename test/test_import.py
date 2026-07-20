"""Smoke test: the package imports cleanly and stays torch-free at import time."""
from __future__ import annotations

import sys

import speechonnxmetrics


def test_version_is_nonempty_string():
    assert isinstance(speechonnxmetrics.__version__, str)
    assert speechonnxmetrics.__version__


def test_subpackages_import():
    import speechonnxmetrics.asr  # noqa: F401
    import speechonnxmetrics.intrusive  # noqa: F401
    import speechonnxmetrics.mos  # noqa: F401
    import speechonnxmetrics.speaker  # noqa: F401


def test_torch_not_eagerly_imported():
    assert "torch" not in sys.modules
