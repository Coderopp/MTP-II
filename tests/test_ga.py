"""GA solver — end-to-end on synthetic Instances."""

from __future__ import annotations

import pytest

from savrptw.eval.feasibility import validate
from savrptw.solvers import build
from savrptw.solvers.base import InfeasibleError
from tests.fixtures import build_mini_instance, make_spiky_risk


def _fast_cfg(**overrides):
    """Cheap GA config to keep tests under a second."""
    cfg = dict(
        pop_size=24,
        generations=40,
        cx_prob=0.85,
        mut_prob=0.25,
        elite_size=2,
        tournament_k=3,
        seed=42,
    )
    cfg.update(overrides)
    return cfg


def test_ga_returns_validated_solution():
    inst = build_mini_instance(N=6)
    solver = build("ga", _fast_cfg())
    sol = solver.solve(inst)
    assert sol.solver == "ga"
    assert len(sol.routes) >= 1
    assert set(c.customer_id for c in inst.customers) == {
        cid for r in sol.routes for cid in r.customers_visited()
    }
    rep = validate(inst, sol)
    assert rep.feasible, rep.violations


def test_ga_is_deterministic_with_same_seed():
    inst = build_mini_instance(N=6)
    s1 = build("ga", _fast_cfg(seed=123)).solve(inst).objective
    s2 = build("ga", _fast_cfg(seed=123)).solve(inst).objective
    assert s1 == s2


def test_ga_finds_better_solution_with_more_search():
    inst = build_mini_instance(N=8)
    cheap = build("ga", _fast_cfg(pop_size=10, generations=5, seed=42)).solve(inst).objective
    rich = build("ga", _fast_cfg(pop_size=30, generations=80, seed=42)).solve(inst).objective
    assert rich <= cheap + 1e-9  # richer search cannot do worse on this instance


def test_ga_handles_single_customer():
    inst = build_mini_instance(N=1)
    sol = build("ga", _fast_cfg()).solve(inst)
    assert len(sol.routes) == 1
    assert sol.routes[0].customers_visited() == [0]


def test_ga_respects_capacity():
    # Q=2, demands all 1 → each route has ≤ 2 customers.
    inst = build_mini_instance(N=6, Q=2)
    sol = build("ga", _fast_cfg()).solve(inst)
    for r in sol.routes:
        assert len(r.customers_visited()) <= inst.Q


def test_ga_refuses_when_risk_budget_infeasible():
    """Set R̄ tighter than any single-arc can satisfy → should raise."""
    inst = build_mini_instance(N=4, R_bar=0.001)
    # Every super-arc has R_uv=0.02 by construction > 0.001 → unreachable.
    with pytest.raises(InfeasibleError):
        build("ga", _fast_cfg()).solve(inst)


def test_ga_reshapes_routes_under_spiky_risk():
    """Raising risk on specific arcs should change the optimal F₁."""
    baseline = build_mini_instance(N=8, R_bar=0.5)
    sp = build_mini_instance(N=8, R_bar=0.5)
    make_spiky_risk(sp, [(0, 1), (2, 3), (4, 5), (6, 7)], high_R=0.4)

    cfg = _fast_cfg(seed=7, pop_size=30, generations=60)
    f_base = build("ga", cfg).solve(baseline).objective
    f_sp = build("ga", cfg).solve(sp).objective
    # Spiky risk forces detours or route re-splits ⇒ F₁ should not improve.
    assert f_sp >= f_base - 1e-6
