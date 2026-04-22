"""Crash-survival MC tests."""

from __future__ import annotations

import math

from savrptw.sim.crash_mc import analytic, simulate
from savrptw.solvers import build
from tests.fixtures import build_mini_instance, make_spiky_risk


def _solve(inst):
    return build("ga", {"pop_size": 20, "generations": 40, "seed": 5}).solve(inst)


def test_analytic_matches_mc_for_low_risk():
    inst = build_mini_instance(N=6, R_bar=0.5)
    sol = _solve(inst)
    a = analytic(inst, sol, n_trips=5000)
    m = simulate(inst, sol, n_trips=5000, seed=123)
    # With R_uv=0.02 across all arcs, P(crash) ≈ 1 − exp(−route_R).
    # MC should agree within ~2× stderr.
    assert abs(a.fleet_p_crash - m.fleet_p_crash) < 0.02


def test_higher_risk_means_higher_expected_crashes():
    base = build_mini_instance(N=6, R_bar=5.0)
    spiky = build_mini_instance(N=6, R_bar=5.0)
    make_spiky_risk(spiky, [(0, 2), (1, 3), (-1, 0), (-2, 1)], high_R=0.9)
    sol_b = _solve(base)
    sol_s = _solve(spiky)
    a_b = analytic(base, sol_b)
    a_s = analytic(spiky, sol_s)
    assert a_s.fleet_p_crash > a_b.fleet_p_crash


def test_wilson_ci_encloses_point_estimate():
    inst = build_mini_instance(N=4, R_bar=1.0)
    sol = _solve(inst)
    a = analytic(inst, sol, n_trips=10_000)
    lo, hi = a.fleet_ci95
    assert lo <= a.fleet_expected_crashes <= hi


def test_per_route_stats_populated():
    inst = build_mini_instance(N=6, R_bar=1.0)
    sol = _solve(inst)
    a = analytic(inst, sol)
    assert len(a.per_route) == len(sol.routes)
    for s in a.per_route:
        assert 0.0 <= s.p_crash <= 1.0
        assert math.isclose(1.0 - math.exp(-s.R_route), s.p_crash, rel_tol=1e-9)
