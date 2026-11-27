"""Per-site-class reliabilities measured on real speech, and the clip decision point.

A recogniser reads a site as one lect, the other, or neither, and how often it does each
for real speakers of each lect is a property of the recogniser, the expected-phone
provider and the site class together. :func:`fit` measures it on labelled clips:

* per site class, the counts of each reading for speakers of each lect, and from them
  the log-ratio of each reading's probability under the two lects, with add-one
  smoothing;
* whether the class is usable: speakers of each lect must be read as their own lect
  more often than speakers of the other, and at least one of the two differences must
  hold at the 5% level (two-proportion z above 1.96). An unusable class is reported but
  carries no weight;
* a logistic fit of the clip label on the summed log-ratio, balanced between the lects,
  whose slope is the temperature that corrects for sites not being independent and whose
  zero is the clip decision point.

The tables shipped with the package are in ``speechonnxmetrics/data/lect_calibration.json``,
with the corpora, splits and sizes they were measured on.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources

import numpy as np

from speechonnxmetrics.lect_fidelity.scoring import NEITHER, SiteReading
from speechonnxmetrics.lect_fidelity.sites import SITE_CLASSES

#: The z a difference in reading rates must exceed for a site class to be usable.
USABLE_Z = 1.96
_TABLE = "lect_calibration.json"


@dataclass(frozen=True)
class ClassCalibration:
    """Measured readings of one site class: ``counts[speaker lect][reading]``, the
    log-ratio of each reading, and whether the class carries weight."""

    counts: dict[str, dict[str, int]]
    llr: dict[str, float]
    usable: bool


@dataclass(frozen=True)
class Calibration:
    """The measured behaviour of one backend with one provider on one lect pair."""

    lects: tuple[str, str]
    provider: str
    backend: str
    classes: dict[str, ClassCalibration]
    temperature: float
    threshold: float
    restrict_inventory: bool = False
    provenance: dict = field(default_factory=dict)

    def log_likelihood_ratio(self, readings: Sequence[SiteReading]) -> float:
        """The summed log-ratio of ``readings`` in favour of ``lects[0]``."""
        total = 0.0
        for r in readings:
            cls = self.classes.get(r.site.cls)
            if cls is not None and cls.usable:
                total += cls.llr[r.label]
        return total

    def posterior(self, llr: float, n_clips: int = 1) -> float:
        """The calibrated probability of ``lects[0]`` given a summed log-ratio over
        ``n_clips`` clips."""
        z = self.temperature * (llr - n_clips * self.threshold)
        return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z))))

    def decide(self, llr: float, n_clips: int = 1) -> str:
        """The lect at the calibrated decision point."""
        return self.lects[0] if llr > n_clips * self.threshold else self.lects[1]

    def swapped(self) -> Calibration:
        """The same calibration with the lects in the other order: every log-ratio and
        the decision point change sign, so both orders reach the same verdict."""
        return Calibration(
            lects=(self.lects[1], self.lects[0]),
            provider=self.provider,
            backend=self.backend,
            classes={
                name: ClassCalibration(c.counts, {k: -v for k, v in c.llr.items()}, c.usable)
                for name, c in self.classes.items()
            },
            temperature=self.temperature,
            threshold=-self.threshold,
            restrict_inventory=self.restrict_inventory,
            provenance=self.provenance,
        )

    def to_dict(self) -> dict:
        return {
            "lects": list(self.lects),
            "provider": self.provider,
            "backend": self.backend,
            "temperature": self.temperature,
            "threshold": self.threshold,
            "restrict_inventory": self.restrict_inventory,
            "classes": {
                name: {"counts": c.counts, "llr": c.llr, "usable": c.usable} for name, c in self.classes.items()
            },
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(
            lects=tuple(d["lects"]),
            provider=d["provider"],
            backend=d["backend"],
            classes={
                name: ClassCalibration(c["counts"], c["llr"], c["usable"]) for name, c in d["classes"].items()
            },
            temperature=d["temperature"],
            threshold=d["threshold"],
            restrict_inventory=d["restrict_inventory"],
            provenance=d["provenance"],
        )


def _z(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    if not n_a or not n_b:
        return 0.0
    p = (hits_a + hits_b) / (n_a + n_b)
    se = math.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))
    return (hits_a / n_a - hits_b / n_b) / se if se else 0.0


def fit_class(counts: dict[str, dict[str, int]], lects: tuple[str, str]) -> ClassCalibration:
    """Log-ratios and usability of one site class from its reading counts."""
    a, b = lects
    labels = (a, b, NEITHER)
    n = {lect: sum(counts[lect].get(k, 0) for k in labels) for lect in lects}
    prob = {
        lect: {k: (counts[lect].get(k, 0) + 1) / (n[lect] + len(labels)) for k in labels} for lect in lects
    }
    llr = {k: math.log(prob[a][k]) - math.log(prob[b][k]) for k in labels}
    z_a = _z(counts[a].get(a, 0), n[a], counts[b].get(a, 0), n[b])
    z_b = _z(counts[b].get(b, 0), n[b], counts[a].get(b, 0), n[a])
    usable = llr[a] > 0 > llr[b] and max(z_a, z_b) > USABLE_Z
    return ClassCalibration({lect: {k: counts[lect].get(k, 0) for k in labels} for lect in lects}, llr, usable)


def _logistic(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of a class-balanced logistic regression of ``y`` on ``x``."""
    w = np.where(y == 1, 0.5 / max(1, y.sum()), 0.5 / max(1, (1 - y).sum()))
    X = np.stack([x, np.ones_like(x)], axis=1)
    beta = np.zeros(2)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -700, 700)))
        grad = X.T @ (w * (y - p))
        hess = X.T @ (X * (w * p * (1 - p))[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def fit(
    clips: Sequence[tuple[str, Sequence[SiteReading]]],
    lects: Sequence[str],
    provider: str,
    backend: str,
    provenance: dict | None = None,
    restrict_inventory: bool = False,
) -> Calibration:
    """Measure a calibration from ``(speaker lect, site readings)`` per clip, decoded
    with or without the inventory restriction as ``restrict_inventory`` says."""
    lects = tuple(lects)
    counts = {c.name: {lect: {} for lect in lects} for c in SITE_CLASSES}
    for lect, readings in clips:
        for r in readings:
            row = counts[r.site.cls][lect]
            row[r.label] = row.get(r.label, 0) + 1
    classes = {name: fit_class(c, lects) for name, c in counts.items()}
    draft = Calibration(lects, provider, backend, classes, 1.0, 0.0)
    x = np.array([draft.log_likelihood_ratio(r) for _, r in clips], dtype=np.float64)
    y = np.array([lect == lects[0] for lect, _ in clips], dtype=np.float64)
    slope, intercept = _logistic(x, y)
    temperature = slope if slope > 0 else 1.0
    threshold = -intercept / slope if slope > 0 else 0.0
    return Calibration(
        lects, provider, backend, classes, temperature, threshold, restrict_inventory, dict(provenance or {})
    )


def _key(lects: Sequence[str], provider: str, backend: str) -> str:
    return f"{lects[0]}|{lects[1]}|{provider}|{backend}"


def load_table() -> dict[str, Calibration]:
    """Every calibration shipped with the package, by ``lect|lect|provider|backend``."""
    data = json.loads(resources.files("speechonnxmetrics").joinpath("data", _TABLE).read_text())
    return {k: Calibration.from_dict(v) for k, v in data["calibrations"].items()}


def shipped(lects: Sequence[str], provider: str, backend: str) -> Calibration | None:
    """The shipped calibration for this lect pair, provider and backend, if any, with
    its lects in the order of ``lects``."""
    table = load_table()
    if (cal := table.get(_key(lects, provider, backend))) is not None:
        return cal
    if (cal := table.get(_key(tuple(reversed(lects)), provider, backend))) is not None:
        return cal.swapped()
    return None


def table_entry(calibration: Calibration) -> tuple[str, dict]:
    """The key and value of ``calibration`` in the shipped table."""
    return _key(calibration.lects, calibration.provider, calibration.backend), calibration.to_dict()
