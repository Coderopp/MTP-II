"""Routines shared by every solver — arrival-time walk, route cost, split.

None of the solver algorithms live here; only the geometric/temporal
arithmetic that MUST be identical across MILP, GA and ALNS.  This guarantees
solvers report the same F₁ on the same route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from savrptw.types import Customer, Depot, Instance, Route, Solution, SuperArc


@dataclass
class RouteBuild:
    """Result of walking a proposed node sequence through arrival arithmetic.

    If `feasible` is False, `violation_mass` encodes the *magnitude* of each
    breach so the GA can use it as a penalty.  A strict validator still has
    the final word (`savrptw.eval.feasibility.validate`).
    """

    nodes: list[int]
    arrivals: list[float]
    load: int
    duration: float
    total_R: float
    total_H: int
    feasible: bool
    violation_mass: float
    f1_contribution: float


def _arc(instance: Instance, u: int, v: int) -> SuperArc | None:
    return instance.super_arcs.get((u, v))


def build_route(
    instance: Instance,
    depot: Depot,
    customers_in_order: list[Customer],
    *,
    penalty_weights: dict[str, float] | None = None,
) -> RouteBuild:
    """Walk through `depot → customers... → depot`, return a RouteBuild.

    Arrival-time recurrence (matches FORMULATION.md §7.6):

        a_{k+1} = max(a_k, e_{c_k}) + s_{c_k} + T(c_k, c_{k+1})

    where `s` is 0 at the depot.  `e_i` models the "can't serve before order
    placed" rule; if a rider arrives earlier, they idle (counted as w in F₁).
    """
    pen = {
        "tw_late": 0.0,
        "tw_early": 0.0,
        "capacity": 0.0,
        "duration": 0.0,
        "risk_budget": 0.0,
        "residential_route_cap": 0.0,
        "missing_arc": 0.0,
        **(penalty_weights or {}),
    }

    nodes = [depot.depot_id] + [c.customer_id for c in customers_in_order] + [depot.depot_id]
    arrivals = [0.0]  # dispatch time

    load = 0
    total_R = 0.0
    total_H = 0
    f1 = 0.0
    violation_mass = 0.0
    feasible = True
    missing = False

    # Walk forward.
    for i in range(len(nodes) - 1):
        u = nodes[i]
        v = nodes[i + 1]
        arc = _arc(instance, u, v)
        if arc is None:
            missing = True
            violation_mass += pen["missing_arc"] + 1.0
            feasible = False
            arrivals.append(arrivals[-1] + 1e6)  # poison mark
            continue

        # Effective departure from u.
        if i == 0:
            # Leaving depot — no service time, no e_i.
            depart = arrivals[-1]
        else:
            cust = customers_in_order[i - 1]
            e = cust.e_i
            s = cust.service_time
            depart = max(arrivals[-1], e) + s
        a_v = depart + float(arc.T_uv)
        arrivals.append(a_v)

        total_R += float(arc.R_uv)
        total_H += int(arc.H_uv)

        # Load / F1 contribution is only meaningful at customer endpoints.
        if 0 < i + 1 < len(nodes) - 1:
            cust_v = customers_in_order[i]  # arriving at customer i+1 (1-based into customers)
            load += cust_v.demand
            w = max(0.0, cust_v.e_i - a_v)
            tau = max(0.0, a_v - cust_v.eta_i)
            f1 += instance.w_early * w + (math.exp(instance.beta_stw * tau) - 1.0)
            # Soft-TW penalty mass (used as search guidance):
            violation_mass += w * pen["tw_early"] + tau * pen["tw_late"]

    duration = arrivals[-1] - arrivals[0] if arrivals else 0.0

    # Hard-constraint violations.
    if load > instance.Q:
        violation_mass += (load - instance.Q) * pen["capacity"]
        feasible = False
    if duration > instance.T_max + 1e-6:
        violation_mass += (duration - instance.T_max) * pen["duration"]
        feasible = False
    if total_R > instance.R_bar + 1e-6:
        violation_mass += (total_R - instance.R_bar) * pen["risk_budget"]
        feasible = False
    if total_H > instance.H_cap_route:
        violation_mass += (total_H - instance.H_cap_route) * pen["residential_route_cap"]
        feasible = False
    if missing:
        feasible = False

    return RouteBuild(
        nodes=nodes,
        arrivals=arrivals,
        load=load,
        duration=duration,
        total_R=total_R,
        total_H=total_H,
        feasible=feasible,
        violation_mass=violation_mass,
        f1_contribution=f1,
    )


def bellman_ford_split(
    instance: Instance,
    depot: Depot,
    customers_in_order: list[Customer],
    *,
    penalty_weights: dict[str, float] | None = None,
) -> tuple[list[list[Customer]], float, float]:
    """Optimal split of a giant customer sequence for one depot.

    Returns
    -------
    routes : list[list[Customer]]
        Partition of `customers_in_order` into contiguous sub-tours, each
        starting and ending at `depot`.
    f1_sum : float
        Sum of F₁ contributions across the produced routes (feasible parts).
    pen_sum : float
        Sum of violation_mass across all candidate arcs used — guides the GA.

    Complexity O(n²).  Dominated arcs (capacity-infeasible tails) are pruned
    early; the whole routine is numpy-free for low constant overhead.
    """
    n = len(customers_in_order)
    if n == 0:
        return [], 0.0, 0.0

    INF = float("inf")
    cost = [INF] * (n + 1)         # cost[j] = best cost to cover customers 0..j-1
    pred = [-1] * (n + 1)          # pred[j] = split point i  ⇒  route = i..j-1
    pen = [0.0] * (n + 1)
    cost[0] = 0.0

    for i in range(n):
        if cost[i] == INF:
            continue
        cumulative_load = 0
        for j in range(i + 1, n + 1):
            cumulative_load += customers_in_order[j - 1].demand
            if cumulative_load > instance.Q:
                break  # any longer route also violates capacity
            rb = build_route(
                instance,
                depot,
                customers_in_order[i:j],
                penalty_weights=penalty_weights,
            )
            # Even infeasible routes receive a finite cost proportional to
            # violation mass — this makes the GA land on near-feasible splits.
            route_cost = rb.f1_contribution + rb.violation_mass
            total = cost[i] + route_cost
            if total < cost[j]:
                cost[j] = total
                pred[j] = i
                pen[j] = pen[i] + rb.violation_mass

    # Reconstruct partition.
    routes: list[list[Customer]] = []
    j = n
    while j > 0:
        i = pred[j]
        if i < 0:
            raise RuntimeError(
                "bellman_ford_split: no feasible splitting — super-graph likely incomplete"
            )
        routes.append(list(customers_in_order[i:j]))
        j = i
    routes.reverse()
    return routes, cost[n], pen[n]


def group_customers_by_depot(
    instance: Instance, chromosome: list[int]
) -> dict[int, list[Customer]]:
    """Partition a customer-id permutation by home_depot, preserving order."""
    cust_by_id: dict[int, Customer] = {c.customer_id: c for c in instance.customers}
    out: dict[int, list[Customer]] = {d.depot_id: [] for d in instance.depots}
    for cid in chromosome:
        cust = cust_by_id[cid]
        out[cust.home_depot].append(cust)
    return out


def assemble_solution(
    instance: Instance,
    chromosome: list[int],
    *,
    solver_name: str,
    penalty_weights: dict[str, float] | None = None,
) -> tuple[Solution, float, bool]:
    """Decode a giant-tour chromosome into a `Solution`.

    Returns
    -------
    solution : Solution
        The decoded solution (may be infeasible — caller must validate).
    surrogate_cost : float
        F₁ + penalty mass (what the GA should be minimising during search).
    feasible_hint : bool
        Whether every RouteBuild returned `feasible=True`.
    """
    depots_by_id = {d.depot_id: d for d in instance.depots}
    grouped = group_customers_by_depot(instance, chromosome)
    routes_out: list[Route] = []
    total_cost = 0.0
    total_pen = 0.0
    f1_total = 0.0
    all_feasible = True
    rider_id = 0

    for depot_id, cust_seq in grouped.items():
        if not cust_seq:
            continue
        depot = depots_by_id[depot_id]
        sub_routes, sub_cost, sub_pen = bellman_ford_split(
            instance, depot, cust_seq, penalty_weights=penalty_weights
        )
        total_cost += sub_cost
        total_pen += sub_pen
        for r_custs in sub_routes:
            rb = build_route(instance, depot, r_custs, penalty_weights=penalty_weights)
            if not rb.feasible:
                all_feasible = False
            f1_total += rb.f1_contribution
            routes_out.append(
                Route(
                    rider_id=rider_id,
                    depot_id=depot_id,
                    nodes=rb.nodes,
                    arrivals=rb.arrivals,
                )
            )
            rider_id += 1

    sol = Solution(
        routes=routes_out,
        objective=f1_total,
        constraint_summary={"search_cost": total_cost, "violation_mass": total_pen},
        solver=solver_name,
    )
    surrogate = f1_total + total_pen
    return sol, surrogate, all_feasible
