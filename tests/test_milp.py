"""MILP solver — tiny-instance correctness checks.

Uses CBC (bundled with PuLP).  Tests are scoped to small N so they finish in
a few seconds each.
"""

from __future__ import annotations

import pytest

from savrptw.eval.feasibility import validate
from savrptw.solvers import build
from savrptw.solvers.base import InfeasibleError
from tests.fixtures import build_mini_instance


def _cfg(**overrides):
    cfg = dict(backend="cbc", time_limit_s=30, mip_gap=0.0, threads=2, log_to_console=False)
    cfg.update(overrides)
    return cfg


@pytest.mark.slow
def test_milp_returns_feasible_solution_tiny():
    inst = build_mini_instance(N=4)
    sol = build("milp", _cfg()).solve(inst)
    rep = validate(inst, sol)
    assert rep.feasible, rep.violations
    assert {c.customer_id for c in inst.customers} == {
        cid for r in sol.routes for cid in r.customers_visited()
    }


@pytest.mark.slow
def test_milp_objective_not_worse_than_ga_on_tiny_instance():
    inst = build_mini_instance(N=4)
    ga_sol = build("ga", {"pop_size": 30, "generations": 60, "seed": 17}).solve(inst)
    milp_sol = build("milp", _cfg()).solve(inst)
    # MILP (proven optimal on tiny N) should be ≤ GA heuristic result.
    assert milp_sol.objective <= ga_sol.objective + 1e-6


@pytest.mark.slow
def test_milp_raises_on_impossible_risk_budget():
    inst = build_mini_instance(N=3, R_bar=0.001)
    with pytest.raises(InfeasibleError):
        build("milp", _cfg()).solve(inst)
