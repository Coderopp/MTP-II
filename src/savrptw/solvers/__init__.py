"""Solvers for SA-VRPTW.

Public API:

    from savrptw.solvers import build, Solver

Importing this package auto-registers the three concrete implementations
(MILP, GA, ALNS) so `build(name, cfg)` works out of the box.
"""

from savrptw.solvers.base import Solver, build

# Auto-register the concrete implementations.
from savrptw.solvers import alns as _alns  # noqa: F401
from savrptw.solvers import ga as _ga  # noqa: F401
from savrptw.solvers import milp as _milp  # noqa: F401

__all__ = ["Solver", "build"]
