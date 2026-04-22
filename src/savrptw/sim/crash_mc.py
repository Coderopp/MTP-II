"""Crash-survival evaluator — replaces the SUMO stub.

FORMULATION.md §12.1.

For each route's super-arc sequence we know the exact expected crash
probability:

    P(no crash on route) = ∏ exp(−R_uv)  =  exp(−R_route)
    P(at least one crash) = 1 − exp(−R_route)

The fleet-level quantity is the probability of ≥1 crash over all routes; if
routes are independent:

    P(fleet clean)   = ∏_routes exp(−R_route)
    P(fleet ≥ 1 crash) = 1 − exp(−Σ R_route)

For paper reporting we also want the *count* of crashes expected over n
dispatches of the same solution.  That count is Binomial(n, p_fleet).  We
report point estimate and a Wilson 95 % CI — accurate even at p near 0.

Optional Monte Carlo mode samples Bernoulli crash events per super-arc for
each of n_trips dispatches; used for consistency-checking against the closed
form and as the substrate for behavioural-compliance experiments (see
`sim/behavioral.py`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from savrptw.types import Instance, Route, Solution


@dataclass
class RouteCrashStat:
    rider_id: int
    depot_id: int
    R_route: float
    p_crash: float  # 1 − exp(−R_route)


@dataclass
class CrashMCResult:
    n_trips: int
    fleet_R: float
    fleet_p_crash: float
    fleet_expected_crashes: float
    fleet_ci95: tuple[float, float]
    per_route: list[RouteCrashStat]
    seed: int | None
    mode: str  # "analytic" | "monte_carlo"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["per_route"] = [asdict(r) for r in self.per_route]
        return d


def _route_R(instance: Instance, route: Route) -> float:
    R = 0.0
    for i in range(len(route.nodes) - 1):
        arc = instance.super_arcs.get((route.nodes[i], route.nodes[i + 1]))
        if arc is None:
            raise ValueError(
                f"super-arc {(route.nodes[i], route.nodes[i + 1])} missing"
            )
        R += float(arc.R_uv)
    return R


def _wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Works well even when p is near 0 or 1.  `successes` may be a fractional
    expectation; we use it as the centre of the proportion.
    """
    if n <= 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def analytic(instance: Instance, solution: Solution, n_trips: int = 10_000) -> CrashMCResult:
    """Closed-form expected crashes + Wilson CI."""
    per_route: list[RouteCrashStat] = []
    fleet_R = 0.0
    for route in solution.routes:
        R = _route_R(instance, route)
        per_route.append(
            RouteCrashStat(
                rider_id=route.rider_id,
                depot_id=route.depot_id,
                R_route=R,
                p_crash=1.0 - math.exp(-R),
            )
        )
        fleet_R += R
    p_fleet = 1.0 - math.exp(-fleet_R)
    n_expected = p_fleet * n_trips
    ci = _wilson_ci(n_expected, n_trips)
    ci_count = (ci[0] * n_trips, ci[1] * n_trips)
    return CrashMCResult(
        n_trips=n_trips,
        fleet_R=fleet_R,
        fleet_p_crash=p_fleet,
        fleet_expected_crashes=n_expected,
        fleet_ci95=ci_count,
        per_route=per_route,
        seed=None,
        mode="analytic",
    )


def simulate(
    instance: Instance,
    solution: Solution,
    n_trips: int = 10_000,
    seed: int = 42,
) -> CrashMCResult:
    """Monte Carlo crash simulation.

    Draws Bernoulli events per super-arc per trip.  Used to cross-check the
    analytic form and to seed the behavioural-compliance sweep.
    """
    rng = np.random.default_rng(seed)
    per_route: list[RouteCrashStat] = []
    fleet_R = 0.0
    fleet_clean = np.ones(n_trips, dtype=bool)

    for route in solution.routes:
        if len(route.nodes) < 2:
            continue
        arc_p = np.array(
            [
                1.0 - math.exp(-float(instance.super_arcs[(route.nodes[i], route.nodes[i + 1])].R_uv))
                for i in range(len(route.nodes) - 1)
            ],
            dtype=float,
        )
        draws = rng.random(size=(n_trips, arc_p.shape[0]))
        # Route survives a trip iff every arc survives.
        arc_survive = draws > arc_p
        route_survive = np.all(arc_survive, axis=1)
        p_crash = float(1.0 - route_survive.mean())
        R = _route_R(instance, route)
        fleet_R += R
        fleet_clean &= route_survive
        per_route.append(
            RouteCrashStat(
                rider_id=route.rider_id,
                depot_id=route.depot_id,
                R_route=R,
                p_crash=p_crash,
            )
        )

    fleet_p = float(1.0 - fleet_clean.mean())
    successes = int(np.sum(~fleet_clean))
    ci = _wilson_ci(successes, n_trips)
    ci_count = (ci[0] * n_trips, ci[1] * n_trips)
    return CrashMCResult(
        n_trips=n_trips,
        fleet_R=fleet_R,
        fleet_p_crash=fleet_p,
        fleet_expected_crashes=float(successes),
        fleet_ci95=ci_count,
        per_route=per_route,
        seed=seed,
        mode="monte_carlo",
    )
