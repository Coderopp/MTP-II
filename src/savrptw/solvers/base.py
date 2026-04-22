"""Solver abstract base and factory.

Every solver (MILP / GA / ALNS) subclasses `Solver`, implements `solve()`, and
MUST:

* consume a `savrptw.types.Instance`,
* return a `savrptw.types.Solution`,
* call `savrptw.eval.validate()` on the final incumbent before returning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from omegaconf import DictConfig

from savrptw.types import Instance, Solution


class Solver(ABC):
    """Abstract base for every SA-VRPTW solver."""

    name: ClassVar[str] = "base"

    def __init__(self, cfg: DictConfig | dict[str, Any]):
        self.cfg = cfg

    @abstractmethod
    def solve(self, instance: Instance) -> Solution:
        """Return a validated Solution or raise `InfeasibleError`."""


class InfeasibleError(RuntimeError):
    """Raised when a solver cannot produce a solution that passes validation."""


_REGISTRY: dict[str, type[Solver]] = {}


def register(cls: type[Solver]) -> type[Solver]:
    """Decorator: register a solver class under its `name` attribute."""
    if cls.name in _REGISTRY:
        raise RuntimeError(f"solver name collision: {cls.name!r}")
    _REGISTRY[cls.name] = cls
    return cls


def build(name: str, cfg: DictConfig | dict[str, Any]) -> Solver:
    """Factory: return a solver instance by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown solver {name!r}; available: {sorted(_REGISTRY)} "
            "(import the module to trigger registration)"
        )
    return _REGISTRY[name](cfg)
