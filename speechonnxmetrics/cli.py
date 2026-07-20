"""speechonnxmetrics command-line entry point.

``speechonnxmetrics score <audio>... [--ref <audio>] [--metrics a,b,c] [--json] [--sr N]``
``speechonnxmetrics list`` — enumerate registered metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from speechonnxmetrics.api import score_batch
from speechonnxmetrics.registry import list_metrics
from speechonnxmetrics.version import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speechonnxmetrics")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    score_p = sub.add_parser("score", help="score one or more audio files")
    score_p.add_argument("audio", nargs="+", help="degraded audio file(s) to score")
    score_p.add_argument("--ref", help="reference audio file (required for intrusive metrics)")
    score_p.add_argument("--metrics", required=True, help="comma-separated metric names")
    score_p.add_argument("--sr", type=int, default=None, help="sample rate hint for raw input")
    score_p.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    list_p = sub.add_parser("list", help="list available metrics")
    list_p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    return parser


def _print_table(rows: Sequence[tuple[str, ...]], headers: tuple[str, ...]) -> None:
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    for row in rows:
        print(fmt.format(*row))


def _cmd_list(args: argparse.Namespace) -> int:
    entries = list_metrics()
    if args.json:
        print(json.dumps(
            [
                {
                    "name": e.name, "kind": e.kind, "intrusive": e.intrusive,
                    "requires_download": e.requires_download,
                }
                for e in entries
            ],
            indent=2,
        ))
        return 0
    rows = [(e.name, e.kind, str(e.intrusive), str(e.requires_download)) for e in entries]
    _print_table(rows, ("name", "kind", "intrusive", "requires_download"))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    refs = [args.ref] * len(args.audio) if args.ref else None
    try:
        results = score_batch(args.audio, metrics, refs=refs, sr=args.sr)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(dict(zip(args.audio, results)), indent=2))
        return 0

    keys = sorted({k for row in results for k in row if k != "_errors"})
    rows = [(path, *(str(row.get(k, "")) for k in keys)) for path, row in zip(args.audio, results)]
    _print_table(rows, ("audio", *keys))
    if any("_errors" in row for row in results):
        for path, row in zip(args.audio, results):
            for metric, err in row.get("_errors", {}).items():
                print(f"error: {path}: {metric}: {err}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return _cmd_list(args)
    if args.command == "score":
        return _cmd_score(args)
    parser.print_help(file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
