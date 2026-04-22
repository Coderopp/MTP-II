"""Top-level CLI entry point.

Usage (via the `savrptw` console script, registered in pyproject.toml):

    savrptw experiment=default solver=ga city=bengaluru instance.N=50

Dispatches to the thin runners under `scripts/` / `savrptw.runner`.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point — currently a placeholder until Task #13 lands."""
    raise NotImplementedError(
        "savrptw CLI — scheduled for Task #13 (experiment runner).  "
        "For now invoke scripts/ directly once they are populated."
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
