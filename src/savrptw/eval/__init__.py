"""Evaluator — the single source of truth for objective and feasibility.

Every solver MUST call `objective()` and `validate()` from this package; no
solver is permitted to compute F₁ or check feasibility internally.
"""

from savrptw.eval.objective import objective
from savrptw.eval.feasibility import validate

__all__ = ["objective", "validate"]
