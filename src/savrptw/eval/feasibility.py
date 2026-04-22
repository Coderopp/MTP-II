"""Feasibility validator — gatekeeper for any solver output.

Implements FORMULATION.md §7 constraints (1)-(12) plus the active ε-bounds
from §6.  Every solver MUST call `validate()` on its final incumbent; only
reports with `feasible=True` may be emitted as a Solution.

Design
------
The validator operates directly on a `Solution` object, which carries per-node
arrival times.  It does NOT re-simulate timing — solvers are expected to
deliver consistent `arrivals`.  The arrival-time arithmetic (constraint 9) is
still checked: given `arrivals[i]`, the validator verifies each transition is
arrival-consistent with `T_ij + s_i` on the super-graph.

Numerical tolerance is 1e-6 minutes (sub-millisecond) — tighter than any
legitimate arrival difference.
"""

from __future__ import annotations

from dataclasses import dataclass

from savrptw.types import Instance, Route, Solution

_TOL = 1e-6


@dataclass
class Violation:
    """A single constraint violation attached to a solution."""

    code: str
    route_id: int | None
    message: str

    def __str__(self) -> str:
        r = f"[route {self.route_id}] " if self.route_id is not None else ""
        return f"{self.code}: {r}{self.message}"


@dataclass
class ValidationReport:
    feasible: bool
    violations: list[Violation]

    def as_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "violations": [
                {"code": v.code, "route_id": v.route_id, "message": v.message}
                for v in self.violations
            ],
        }


def _depot_ids(instance: Instance) -> set[int]:
    return {d.depot_id for d in instance.depots}


def _customer_index(instance: Instance) -> dict[int, object]:
    return {c.customer_id: c for c in instance.customers}


def _check_coverage(
    instance: Instance, solution: Solution, violations: list[Violation]
) -> None:
    """Constraint (1): every customer served exactly once across the fleet."""
    visits: dict[int, int] = {c.customer_id: 0 for c in instance.customers}
    for route in solution.routes:
        for cid in route.customers_visited():
            if cid not in visits:
                violations.append(
                    Violation(
                        "UNKNOWN_CUSTOMER",
                        route.rider_id,
                        f"node {cid} is not a customer in this instance",
                    )
                )
                continue
            visits[cid] += 1
    for cid, n in visits.items():
        if n == 0:
            violations.append(Violation("CUSTOMER_UNSERVED", None, f"customer {cid} not visited"))
        elif n > 1:
            violations.append(
                Violation(
                    "CUSTOMER_DOUBLE_VISIT",
                    None,
                    f"customer {cid} visited {n} times (must be exactly once)",
                )
            )


def _check_route_shape(
    instance: Instance, route: Route, violations: list[Violation]
) -> None:
    """Each route: starts and ends at the same depot; no depot in the middle.

    Covers constraints (4)-(6) at the sequence level.
    """
    depots = _depot_ids(instance)
    if len(route.nodes) < 2:
        violations.append(
            Violation("EMPTY_ROUTE", route.rider_id, "route must have at least one customer")
        )
        return
    if route.nodes[0] not in depots:
        violations.append(
            Violation("NO_DEPOT_START", route.rider_id, f"first node {route.nodes[0]} not a depot")
        )
    if route.nodes[-1] not in depots:
        violations.append(
            Violation("NO_DEPOT_END", route.rider_id, f"last node {route.nodes[-1]} not a depot")
        )
    if route.nodes[0] != route.depot_id:
        violations.append(
            Violation(
                "DEPOT_MISMATCH",
                route.rider_id,
                f"route.depot_id={route.depot_id} but starts at {route.nodes[0]}",
            )
        )
    if route.nodes[0] != route.nodes[-1]:
        violations.append(
            Violation(
                "NO_CLOSED_TOUR",
                route.rider_id,
                f"starts at {route.nodes[0]}, ends at {route.nodes[-1]}",
            )
        )
    # No intermediate depot visits.
    for mid in route.nodes[1:-1]:
        if mid in depots:
            violations.append(
                Violation(
                    "DEPOT_IN_MIDDLE",
                    route.rider_id,
                    f"node {mid} is a depot but appears as a customer",
                )
            )


def _check_home_depot(
    instance: Instance,
    route: Route,
    cust_by_id: dict[int, object],
    violations: list[Violation],
) -> None:
    """Each customer on a route must have its `home_depot` == route depot."""
    for cid in route.customers_visited():
        cust = cust_by_id.get(cid)
        if cust is None:
            continue  # already reported by _check_coverage
        if getattr(cust, "home_depot", route.depot_id) != route.depot_id:
            violations.append(
                Violation(
                    "CROSS_DEPOT_CUSTOMER",
                    route.rider_id,
                    f"customer {cid} assigned to depot {cust.home_depot} "  # type: ignore[attr-defined]
                    f"but on a route from depot {route.depot_id}",
                )
            )


def _check_capacity(
    instance: Instance,
    route: Route,
    cust_by_id: dict[int, object],
    violations: list[Violation],
) -> None:
    """Constraint (3): sum of demands on a route ≤ Q."""
    total = 0
    for cid in route.customers_visited():
        cust = cust_by_id.get(cid)
        if cust is None:
            continue
        total += int(getattr(cust, "demand", 0))
    if total > instance.Q:
        violations.append(
            Violation(
                "CAPACITY",
                route.rider_id,
                f"sum(q_i)={total} > Q={instance.Q}",
            )
        )


def _check_arrival_linkage(
    instance: Instance,
    route: Route,
    cust_by_id: dict[int, object],
    violations: list[Violation],
) -> None:
    """Constraint (9): a_j ≥ a_i + s_i + T_ij along the route."""
    if len(route.nodes) != len(route.arrivals):
        violations.append(
            Violation(
                "ARRIVAL_SHAPE",
                route.rider_id,
                f"len(nodes)={len(route.nodes)} != len(arrivals)={len(route.arrivals)}",
            )
        )
        return
    if route.arrivals and abs(route.arrivals[0]) > _TOL:
        violations.append(
            Violation(
                "DEPOT_DEPART_NONZERO",
                route.rider_id,
                f"dispatch time must be 0, got {route.arrivals[0]}",
            )
        )
    for i in range(len(route.nodes) - 1):
        u = route.nodes[i]
        v = route.nodes[i + 1]
        arc = instance.super_arcs.get((u, v))
        if arc is None:
            violations.append(
                Violation(
                    "MISSING_SUPER_ARC",
                    route.rider_id,
                    f"super-arc ({u},{v}) absent from instance — cannot verify timing",
                )
            )
            continue
        # Service time at u (0 if u is a depot).
        s_u = 0.0
        if u in cust_by_id:
            s_u = float(getattr(cust_by_id[u], "service_time", 0.0))
        # Effective departure from u = max(arrival_at_u, e_i) + s_i.
        a_u = route.arrivals[i]
        e_u = 0.0
        if u in cust_by_id:
            e_u = float(getattr(cust_by_id[u], "e_i", 0.0))
        depart = max(a_u, e_u) + s_u
        expected = depart + float(arc.T_uv)
        got = route.arrivals[i + 1]
        if got + _TOL < expected:
            violations.append(
                Violation(
                    "ARRIVAL_LINK",
                    route.rider_id,
                    f"a[{v}]={got:.6f} < expected {expected:.6f} "
                    f"(depart={depart:.6f} + T={arc.T_uv:.6f})",
                )
            )


def _check_route_duration(
    instance: Instance, route: Route, violations: list[Violation]
) -> None:
    """Constraint (10): total route duration ≤ T_max."""
    if not route.arrivals:
        return
    duration = route.arrivals[-1] - route.arrivals[0]
    if duration > instance.T_max + _TOL:
        violations.append(
            Violation(
                "ROUTE_DURATION",
                route.rider_id,
                f"duration {duration:.4f} > T_max={instance.T_max}",
            )
        )


def _check_eps_budgets(
    instance: Instance, solution: Solution, violations: list[Violation]
) -> None:
    """ε-constraints (§6): per-route R̄, per-route H̄_k, fleet H̄."""
    fleet_h = 0
    for route in solution.routes:
        r_sum = 0.0
        h_sum = 0
        for i in range(len(route.nodes) - 1):
            arc = instance.super_arcs.get((route.nodes[i], route.nodes[i + 1]))
            if arc is None:
                continue
            r_sum += float(arc.R_uv)
            h_sum += int(arc.H_uv)
        if r_sum > instance.R_bar + _TOL:
            violations.append(
                Violation(
                    "RISK_BUDGET_ROUTE",
                    route.rider_id,
                    f"Σ R_uv={r_sum:.6f} > R̄={instance.R_bar}",
                )
            )
        if h_sum > instance.H_cap_route:
            violations.append(
                Violation(
                    "RESIDENTIAL_CAP_ROUTE",
                    route.rider_id,
                    f"Σ H_uv={h_sum} > H̄_k={instance.H_cap_route}",
                )
            )
        fleet_h += h_sum
    if fleet_h > instance.H_bar:
        violations.append(
            Violation(
                "RESIDENTIAL_BUDGET_FLEET",
                None,
                f"fleet Σ H_uv={fleet_h} > H̄={instance.H_bar}",
            )
        )


def validate(instance: Instance, solution: Solution) -> ValidationReport:
    """Run every constraint check; return a `ValidationReport`.

    Does not raise — callers (typically a Solver's post-solve hook) decide
    whether to reject the incumbent or emit a warning.
    """
    violations: list[Violation] = []
    cust_by_id = _customer_index(instance)

    _check_coverage(instance, solution, violations)
    for route in solution.routes:
        _check_route_shape(instance, route, violations)
        _check_home_depot(instance, route, cust_by_id, violations)
        _check_capacity(instance, route, cust_by_id, violations)
        _check_arrival_linkage(instance, route, cust_by_id, violations)
        _check_route_duration(instance, route, violations)
    _check_eps_budgets(instance, solution, violations)

    return ValidationReport(feasible=not violations, violations=violations)
