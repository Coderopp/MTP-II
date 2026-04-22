"""Objective function — F₁ (ETA deviation + STW penalty).

Matches §5 of docs/FORMULATION.md exactly.  No other component may compute F₁.
"""

from __future__ import annotations

import math

from savrptw.types import Instance, Route, Solution


def objective(instance: Instance, solution: Solution) -> float:
    """Compute F₁ from scratch for `solution` under `instance`.

    F₁ = Σ_{k, i}  [ w_early · w_i^k  +  ( exp(β_stw · τ_i^k) − 1 ) ]

    `w_i^k` is idle wait (`max(0, e_i − a_i^k)`), `τ_i^k` is lateness
    (`max(0, a_i^k − ETA_i)`).  Arrivals are taken verbatim from the solution;
    this function does not re-simulate timing.
    """
    cust_by_id = {c.customer_id: c for c in instance.customers}
    total = 0.0

    for route in solution.routes:
        for idx, node in enumerate(route.nodes):
            if node == route.depot_id:
                continue
            cust = cust_by_id.get(node)
            if cust is None:
                raise ValueError(
                    f"route {route.rider_id}: node {node} is neither depot nor customer"
                )
            a = route.arrivals[idx]
            w = max(0.0, cust.e_i - a)
            tau = max(0.0, a - cust.eta_i)
            total += instance.w_early * w + (math.exp(instance.beta_stw * tau) - 1.0)

    return total


def breakdown(instance: Instance, solution: Solution) -> dict[str, float]:
    """Per-route and total diagnostic decomposition of F₁."""
    cust_by_id = {c.customer_id: c for c in instance.customers}
    early = 0.0
    late_exp = 0.0
    for route in solution.routes:
        for idx, node in enumerate(route.nodes):
            if node == route.depot_id:
                continue
            cust = cust_by_id[node]
            a = route.arrivals[idx]
            early += max(0.0, cust.e_i - a) * instance.w_early
            tau = max(0.0, a - cust.eta_i)
            late_exp += math.exp(instance.beta_stw * tau) - 1.0
    return {"early_wait": early, "late_stw_exp": late_exp, "F1": early + late_exp}
