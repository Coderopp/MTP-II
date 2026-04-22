"""Feasibility validator tests — synthetic instance, exhaustive constraint coverage."""

from __future__ import annotations

import networkx as nx

from savrptw.eval.feasibility import validate
from savrptw.types import Customer, Depot, Instance, Route, Solution, SuperArc


def _mini_instance(
    R_bar: float = 1.0, H_bar: int = 1000, H_cap_route: int = 8, T_max: float = 35.0
) -> Instance:
    """Two customers, one depot, two super-arcs out and two back."""
    d = Depot(depot_id=0, osm_node=0, lat=0, lon=0, brand="Blinkit")
    c1 = Customer(
        customer_id=1, osm_node=1, lat=0, lon=0, demand=1,
        e_i=0.0, eta_i=10.0, service_time=2.0, home_depot=0,
    )
    c2 = Customer(
        customer_id=2, osm_node=2, lat=0, lon=0, demand=1,
        e_i=0.0, eta_i=10.0, service_time=2.0, home_depot=0,
    )
    arcs = {
        (0, 1): SuperArc(u=0, v=1, T_uv=3.0, R_uv=0.05, H_uv=1),
        (1, 2): SuperArc(u=1, v=2, T_uv=2.0, R_uv=0.05, H_uv=0),
        (2, 0): SuperArc(u=2, v=0, T_uv=4.0, R_uv=0.05, H_uv=1),
        (0, 2): SuperArc(u=0, v=2, T_uv=3.0, R_uv=0.05, H_uv=1),
        (2, 1): SuperArc(u=2, v=1, T_uv=2.0, R_uv=0.05, H_uv=0),
        (1, 0): SuperArc(u=1, v=0, T_uv=3.0, R_uv=0.05, H_uv=1),
    }
    return Instance(
        city="test",
        depots=[d],
        customers=[c1, c2],
        street_graph=nx.MultiDiGraph(),
        super_arcs=arcs,
        Q=2,
        T_max=T_max,
        H_cap_route=H_cap_route,
        R_bar=R_bar,
        H_bar=H_bar,
    )


def _feasible_solution(inst: Instance) -> Solution:
    # 0→1→2→0; arrivals consistent with s_i=2 service time.
    arrivals = [0.0, 3.0, 3.0 + 2.0 + 2.0, 7.0 + 2.0 + 4.0]  # 0, 3, 7, 13
    r = Route(rider_id=0, depot_id=0, nodes=[0, 1, 2, 0], arrivals=arrivals)
    return Solution(routes=[r], objective=0.0, constraint_summary={}, solver="test")


def test_feasible_solution_passes():
    inst = _mini_instance()
    sol = _feasible_solution(inst)
    rep = validate(inst, sol)
    assert rep.feasible, rep.violations


def test_detects_unserved_customer():
    inst = _mini_instance()
    r = Route(rider_id=0, depot_id=0, nodes=[0, 1, 0], arrivals=[0.0, 3.0, 3.0 + 2.0 + 3.0])
    sol = Solution(routes=[r], objective=0.0, constraint_summary={}, solver="test")
    rep = validate(inst, sol)
    assert not rep.feasible
    assert any(v.code == "CUSTOMER_UNSERVED" for v in rep.violations)


def test_detects_capacity_violation():
    inst = _mini_instance()
    # Force q_i over Q=2.
    inst.customers[0] = Customer(
        customer_id=1, osm_node=1, lat=0, lon=0, demand=2,
        e_i=0.0, eta_i=10.0, service_time=2.0, home_depot=0,
    )
    inst.customers[1] = Customer(
        customer_id=2, osm_node=2, lat=0, lon=0, demand=2,
        e_i=0.0, eta_i=10.0, service_time=2.0, home_depot=0,
    )
    sol = _feasible_solution(inst)
    rep = validate(inst, sol)
    assert any(v.code == "CAPACITY" for v in rep.violations)


def test_detects_risk_budget_violation():
    inst = _mini_instance(R_bar=0.05)  # tight
    sol = _feasible_solution(inst)
    rep = validate(inst, sol)
    assert any(v.code == "RISK_BUDGET_ROUTE" for v in rep.violations)


def test_detects_residential_cap_violation():
    inst = _mini_instance(H_cap_route=1)  # tight; the route uses 2 residential edges
    sol = _feasible_solution(inst)
    rep = validate(inst, sol)
    assert any(v.code == "RESIDENTIAL_CAP_ROUTE" for v in rep.violations)


def test_detects_no_closed_tour():
    inst = _mini_instance()
    r = Route(rider_id=0, depot_id=0, nodes=[0, 1, 2], arrivals=[0.0, 3.0, 7.0])
    sol = Solution(routes=[r], objective=0.0, constraint_summary={}, solver="test")
    rep = validate(inst, sol)
    codes = {v.code for v in rep.violations}
    assert "NO_CLOSED_TOUR" in codes or "NO_DEPOT_END" in codes


def test_detects_arrival_link_violation():
    inst = _mini_instance()
    # arrivals shortchange the 1→2 transition (should be 7, claim 4).
    r = Route(rider_id=0, depot_id=0, nodes=[0, 1, 2, 0], arrivals=[0.0, 3.0, 4.0, 10.0])
    sol = Solution(routes=[r], objective=0.0, constraint_summary={}, solver="test")
    rep = validate(inst, sol)
    assert any(v.code == "ARRIVAL_LINK" for v in rep.violations)


def test_detects_route_duration_violation():
    inst = _mini_instance(T_max=10.0)
    sol = _feasible_solution(inst)  # duration is 13 > 10
    rep = validate(inst, sol)
    assert any(v.code == "ROUTE_DURATION" for v in rep.violations)
