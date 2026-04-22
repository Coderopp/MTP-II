"""Behavioural-compliance evaluator.

FORMULATION.md §10.7 and §12.2.

Rider compliance `c ∈ [0, 1]` is drawn from Beta(α, β) per simulated rider.
Non-compliance inflates crash risk per edge:

    r_effective = 1 − (1 − r)^(1 + k_c·(1 − c))

Because R_uv of a super-arc is `Σ -ln(1 − r_edge)`, this scales cleanly:

    R_effective_uv = R_uv · (1 + k_c·(1 − c))

So we can work at the super-arc level without unpacking the underlying street
graph.  The evaluator:

1. Samples compliance per rider per trip.
2. Scales R_uv for each super-arc on that rider's route.
3. Computes per-trip crash probability analytically (1 − exp(−R_scaled_route)).
4. Aggregates across riders and trips with a Wilson CI.
5. Sweeps `(α, β)` to produce sensitivity bands.

This module is a **post-hoc evaluator** — nothing in the solver optimisation
loop depends on it (FORMULATION.md §10.7).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from savrptw.sim.crash_mc import _route_R, _wilson_ci
from savrptw.types import Instance, Solution


@dataclass
class BehaviorPoint:
    alpha: float
    beta: float
    n_riders_per_route: int
    n_trips: int
    mean_compliance: float
    fleet_expected_crashes: float
    fleet_ci95: tuple[float, float]
    seed: int


@dataclass
class BehaviorReport:
    points: list[BehaviorPoint]
    base_fleet_R: float
    k_c: float

    def as_dict(self) -> dict:
        return {
            "base_fleet_R": self.base_fleet_R,
            "k_c": self.k_c,
            "points": [asdict(p) for p in self.points],
        }


def _route_Rs(instance: Instance, solution: Solution) -> list[float]:
    return [_route_R(instance, r) for r in solution.routes if len(r.nodes) >= 2]


def simulate(
    instance: Instance,
    solution: Solution,
    *,
    alpha: float = 5.0,
    beta: float = 2.0,
    k_c: float = 2.0,
    n_riders_per_route: int = 100,
    n_trips: int = 10_000,
    seed: int = 42,
) -> BehaviorPoint:
    """Single (α, β) behavioural simulation.

    For each route we spawn `n_riders_per_route` rider profiles (draws from
    Beta(α, β)); each rider runs `n_trips // n_riders_per_route` dispatches
    with its own scaled risk.  The output is a population-level estimate of
    expected crashes on that configuration.
    """
    if n_riders_per_route < 1:
        raise ValueError("n_riders_per_route must be >= 1")
    rng = np.random.default_rng(seed)

    per_route_Rs = _route_Rs(instance, solution)
    if not per_route_Rs:
        return BehaviorPoint(
            alpha=alpha,
            beta=beta,
            n_riders_per_route=n_riders_per_route,
            n_trips=n_trips,
            mean_compliance=float("nan"),
            fleet_expected_crashes=0.0,
            fleet_ci95=(0.0, 0.0),
            seed=seed,
        )

    # Sample compliance per rider; shape (n_routes, n_riders_per_route).
    c = rng.beta(alpha, beta, size=(len(per_route_Rs), n_riders_per_route))
    # Scale factor (1 + k_c (1 − c)).
    scale = 1.0 + k_c * (1.0 - c)
    # Scaled route R per rider per route.
    R_matrix = np.asarray(per_route_Rs, dtype=float)[:, None] * scale  # (routes, riders)
    # Per-rider per-route P(crash on one dispatch).
    p_rider_route = 1.0 - np.exp(-R_matrix)
    # Trips per rider.
    trips_per_rider = max(1, n_trips // n_riders_per_route)
    # Expected crashes per (route, rider) across its trips = p × trips.
    crashes_matrix = p_rider_route * trips_per_rider
    total_crashes = float(crashes_matrix.sum())
    total_trips = trips_per_rider * n_riders_per_route * len(per_route_Rs)
    ci = _wilson_ci(total_crashes, total_trips)
    ci_count = (ci[0] * total_trips, ci[1] * total_trips)

    return BehaviorPoint(
        alpha=alpha,
        beta=beta,
        n_riders_per_route=n_riders_per_route,
        n_trips=total_trips,
        mean_compliance=float(c.mean()),
        fleet_expected_crashes=total_crashes,
        fleet_ci95=ci_count,
        seed=seed,
    )


def sensitivity_sweep(
    instance: Instance,
    solution: Solution,
    *,
    alphas: Iterable[float] = (3.0, 5.0, 8.0),
    betas: Iterable[float] = (1.0, 2.0, 3.0),
    k_c: float = 2.0,
    n_riders_per_route: int = 100,
    n_trips: int = 10_000,
    seed: int = 42,
) -> BehaviorReport:
    """Sweep (α, β) and produce IQR-friendly bands for the paper figure."""
    points: list[BehaviorPoint] = []
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            pt = simulate(
                instance,
                solution,
                alpha=a,
                beta=b,
                k_c=k_c,
                n_riders_per_route=n_riders_per_route,
                n_trips=n_trips,
                seed=seed + 1000 * i + j,
            )
            points.append(pt)
    base_R = sum(_route_Rs(instance, solution))
    return BehaviorReport(points=points, base_fleet_R=base_R, k_c=k_c)
