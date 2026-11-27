"""Docs check: figure data, measured numbers, code examples and relative links.

A ``python`` block runs when one of its lines is ``print(<expr>)  # <expected>``. The
check compares ``str(<expr>)`` with ``<expected>``, and compares numbers with a
relative tolerance of 1e-6. Blocks of one page run in order in one process, from the
repository root.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = [ROOT / "README.md", *sorted(DOCS.rglob("*.md"))]
FIGURES = sorted(p for p in (DOCS / "img").glob("plot_*.py") if p.name != "plot_style.py")
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
CHECK = re.compile(r"^(\s*)print\((.*)\)\s+# (.+)$")
LINK = re.compile(r"\]\(([^)\s]+)\)|(?:src|srcset)=\"([^\"]+)\"")


def page_id(path: Path) -> str:
    return str(path.relative_to(ROOT))


def same(value, expected) -> bool:
    if isinstance(value, bool) or isinstance(expected, bool):
        return value == expected
    if isinstance(expected, (int, float)) and not isinstance(value, (str, bytes)):
        try:
            return math.isclose(float(value), expected, rel_tol=1e-6, abs_tol=1e-9)
        except TypeError:
            return False
    if isinstance(expected, dict) and isinstance(value, dict):
        return value.keys() == expected.keys() and all(same(value[k], expected[k]) for k in expected)
    if isinstance(expected, (list, tuple)) and isinstance(value, (list, tuple)):
        return len(value) == len(expected) and all(same(v, e) for v, e in zip(value, expected))
    return value == expected


def check(value, expected: str, where: str) -> None:
    if str(value) == expected:
        return
    try:
        literal = ast.literal_eval(expected)
    except (ValueError, SyntaxError):
        literal = None
    assert literal is not None and same(value, literal), f"{where}: printed {value!r}, the page says {expected}"


def runnable_blocks(path: Path) -> list[str]:
    blocks = BLOCK.findall(path.read_text(encoding="utf-8"))
    return [b for b in blocks if any(CHECK.match(line) for line in b.splitlines())]


def program(path: Path) -> str:
    """The page's runnable blocks as one script, each checked print turned into a ``check`` call."""
    lines = [f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r})", "from test_docs import check"]
    for n, block in enumerate(runnable_blocks(path), 1):
        for i, line in enumerate(block.splitlines(), 1):
            m = CHECK.match(line)
            if m:
                indent, expr, expected = m.groups()
                line = f"{indent}check({expr}, {expected!r}, {f'{page_id(path)} block {n} line {i}'!r})"
            lines.append(line)
    return "\n".join(lines)


@pytest.mark.parametrize("path", PAGES, ids=page_id)
def test_examples(path: Path) -> None:
    """Each page runs in its own process, so a metric that a page registers stays on that page."""
    if not runnable_blocks(path):
        return
    run = subprocess.run([sys.executable, "-c", program(path)], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]


@pytest.mark.parametrize("path", PAGES, ids=page_id)
def test_links(path: Path) -> None:
    text = re.sub(r"```.*?```|<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
    for groups in LINK.findall(text):
        target = (groups[0] or groups[1]).split("#")[0]
        if not target or re.match(r"[a-z]+:", target):
            continue
        assert (path.parent / target).exists(), f"{page_id(path)}: broken link {target}"


@pytest.mark.parametrize("script", FIGURES, ids=lambda p: p.name)
def test_figure_data(script: Path) -> None:
    stem = script.stem.removeprefix("plot_").replace("_", "-")
    data = DOCS / "img" / f"{stem}.json"
    committed = data.read_bytes()
    env = {**os.environ, "DOCS_FIGURES_DATA_ONLY": "1"}
    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT, env=env)
        assert data.read_bytes() == committed, f"{data.name} differs from a fresh read: run {script.name}"
    finally:
        data.write_bytes(committed)


def test_lect_auc_table_and_alt_text() -> None:
    page = (DOCS / "lect-fidelity.md").read_text(encoding="utf-8")
    clul, accents = "CLUL spoken varieties held-out recordings", "compare-accents-pt"
    for row in json.loads((DOCS / "img" / "lect-auc.json").read_text(encoding="utf-8")):
        cells = [f"{row[c]['auc']:.2f} ({row[c]['auc_95ci'][0]:.2f} to {row[c]['auc_95ci'][1]:.2f})" for c in (clul, accents)]
        line = f"| `{row['backend']}` | `{row['provider']}` | {cells[0]} | {cells[1]} |"
        assert line in page, f"missing table row: {line}"
        alt = f"{row['backend']} with {row['provider']} {row[clul]['auc']:.2f}"
        assert alt in page, f"missing alt text value: {alt}"


def test_metric_count() -> None:
    import speechonnxmetrics as s

    count = f"{len(s.list_metrics())} metrics"
    for name in ("README.md", "docs/index.md"):
        assert count in (ROOT / name).read_text(encoding="utf-8"), f"{name} does not say {count}"
