"""speechmetrics command-line entry point (stub)."""
from __future__ import annotations

import sys

from speechmetrics.version import __version__


def main() -> int:
    print(f"speechmetrics {__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
