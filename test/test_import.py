"""Smoke test: the package imports cleanly and stays torch-free at import time."""
from __future__ import annotations

import sys

import speechmetrics


def test_version_is_nonempty_string():
    assert isinstance(speechmetrics.__version__, str)
    assert speechmetrics.__version__


def test_subpackages_import():
    import speechmetrics.asr  # noqa: F401
    import speechmetrics.intrusive  # noqa: F401
    import speechmetrics.mos  # noqa: F401
    import speechmetrics.speaker  # noqa: F401


def test_torch_not_eagerly_imported():
    assert "torch" not in sys.modules
