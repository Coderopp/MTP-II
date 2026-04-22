"""ALNS solver tests."""

from __future__ import annotations

from savrptw.eval.feasibility import validate
from savrptw.solvers import build
from tests.fixtures import build_mini_instance


def _cfg(**overrides):
    cfg = dict(iterations=500, segment_length=50, reaction_factor=0.2, seed=11)
    cfg.update(overrides)
    return cfg


def test_alns_returns_validated_solution():
    inst = build_mini_instance(N=6)
    sol = build("alns", _cfg()).solve(inst)
    rep = validate(inst, sol)
    assert rep.feasible, rep.violations
    assert {c.customer_id for c in inst.customers} == {
        cid for r in sol.routes for cid in r.customers_visited()
    }


def test_alns_respects_capacity():
    inst = build_mini_instance(N=6, Q=2)
    sol = build("alns", _cfg()).solve(inst)
    for r in sol.routes:
        assert len(r.customers_visited()) <= inst.Q


def test_alns_is_deterministic_with_same_seed():
    inst = build_mini_instance(N=6)
    a = build("alns", _cfg(seed=77)).solve(inst).objective
    b = build("alns", _cfg(seed=77)).solve(inst).objective
    assert a == b


def test_alns_matches_or_improves_initial_heuristic():
    """A longer ALNS run should not do worse than a short one on the same seed."""
    inst = build_mini_instance(N=8, R_bar=0.5)
    cheap = build("alns", _cfg(iterations=50)).solve(inst).objective
    rich = build("alns", _cfg(iterations=1500)).solve(inst).objective
    assert rich <= cheap + 1e-6
