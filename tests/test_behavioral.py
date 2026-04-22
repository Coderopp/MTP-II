"""Behavioural-compliance MC tests."""

from __future__ import annotations

from savrptw.sim.behavioral import sensitivity_sweep, simulate
from savrptw.solvers import build
from tests.fixtures import build_mini_instance


def _solve(inst):
    return build("ga", {"pop_size": 20, "generations": 40, "seed": 9}).solve(inst)


def test_non_compliance_increases_expected_crashes():
    inst = build_mini_instance(N=6, R_bar=1.0)
    sol = _solve(inst)
    compliant = simulate(
        inst, sol, alpha=10.0, beta=1.0, n_riders_per_route=30, n_trips=6_000, seed=42
    )
    noncompliant = simulate(
        inst, sol, alpha=1.0, beta=10.0, n_riders_per_route=30, n_trips=6_000, seed=42
    )
    # Low-compliance cohort should see more expected crashes.
    assert noncompliant.fleet_expected_crashes > compliant.fleet_expected_crashes


def test_sweep_emits_nine_points_for_3x3_grid():
    inst = build_mini_instance(N=6)
    sol = _solve(inst)
    report = sensitivity_sweep(
        inst, sol, alphas=(3.0, 5.0, 8.0), betas=(1.0, 2.0, 3.0),
        n_riders_per_route=20, n_trips=2_000,
    )
    assert len(report.points) == 9
    # Base R reported equals sum of route Rs (small, positive).
    assert report.base_fleet_R > 0.0


def test_mean_compliance_tracks_alpha_over_beta():
    inst = build_mini_instance(N=4)
    sol = _solve(inst)
    low = simulate(inst, sol, alpha=1.0, beta=9.0, n_riders_per_route=50, n_trips=3_000)
    high = simulate(inst, sol, alpha=9.0, beta=1.0, n_riders_per_route=50, n_trips=3_000)
    assert low.mean_compliance < high.mean_compliance
