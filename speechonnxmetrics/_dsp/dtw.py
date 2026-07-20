"""Dynamic time warping, needed downstream for MCD (mel-cepstral distortion) alignment.

The local-distance matrix is fully vectorized (broadcasted, no python loop). The
accumulated-cost recursion is inherently sequential — ``cost[i, j]`` depends on
``cost[i, j-1]`` within the same row — so the standard symmetric step pattern is
computed with a plain double loop over the (small, frame-rate) cost matrix; there is no
per-sample audio loop anywhere in this module.
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np

StepPattern = Literal["symmetric1", "symmetric2"]


class DTWError(ValueError):
    """Raised for DTW inputs that cannot be aligned (empty sequences, bad shapes)."""


def _euclidean_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def dtw(
    x: np.ndarray, y: np.ndarray,
    distance: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    step_pattern: StepPattern = "symmetric1",
) -> tuple[np.ndarray, float]:
    """Align ``x`` ``[n, d]`` to ``y`` ``[m, d]``; returns ``(path, cost)``.

    ``path`` is an ``[k, 2]`` array of ``(i, j)`` index pairs from ``(0, 0)`` to
    ``(n-1, m-1)``. ``cost`` is the total accumulated cost (``symmetric2`` normalizes by
    path length, matching the step pattern most MCD implementations use).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    if x.size == 0 or y.size == 0:
        raise DTWError("dtw requires non-empty sequences")
    if x.shape[-1] != y.shape[-1]:
        raise DTWError(f"feature dims must match, got {x.shape[-1]} and {y.shape[-1]}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise DTWError("input sequences contain NaN/inf")

    dist = _euclidean_matrix(x, y) if distance is None else distance(x, y)
    n, m = dist.shape

    cost = np.full((n, m), np.inf, dtype=np.float64)
    weight = np.ones((n, m), dtype=np.float64) if step_pattern == "symmetric2" else None
    cost[0, 0] = dist[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0:
                candidates.append(cost[i - 1, j])
            if j > 0:
                candidates.append(cost[i, j - 1])
            if i > 0 and j > 0:
                diag = cost[i - 1, j - 1]
                candidates.append(diag * 2 if step_pattern == "symmetric2" else diag)
            cost[i, j] = dist[i, j] + min(candidates)

    path = []
    i, j = n - 1, m - 1
    while (i, j) != (0, 0):
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            prev = min(
                (cost[i - 1, j], (i - 1, j)),
                (cost[i, j - 1], (i, j - 1)),
                (cost[i - 1, j - 1], (i - 1, j - 1)),
                key=lambda t: t[0],
            )
            i, j = prev[1]
    path.append((0, 0))
    path.reverse()
    return np.array(path, dtype=np.int64), float(cost[n - 1, m - 1])
