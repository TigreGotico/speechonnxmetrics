"""speechonnxmetrics command-line entry point (stub)."""
from __future__ import annotations

import sys

from speechonnxmetrics.version import __version__


def main() -> int:
    print(f"speechonnxmetrics {__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
