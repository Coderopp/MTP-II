"""Test fixtures — synthetic Instances for solver unit tests.

These Instances are hand-built so unit tests don't depend on OSM downloads,
BASM calibration, or KML parsing.  They are *test inputs*, not production
mocks — the production code paths for data ingestion are exercised elsewhere
in `test_darkstores.py`, `test_bpr.py`, etc.
"""

from __future__ import annotations

import networkx as nx

from savrptw.types import Customer, Depot, Instance, SuperArc


def build_mini_instance(
    N: int = 6,
    *,
    Q: int = 2,
    T_max: float = 35.0,
    R_bar: float = 1.0,
    H_bar: int = 100,
    H_cap_route: int = 4,
    seed: int = 42,
    cross_depot_noise: bool = False,
) -> Instance:
    """Two depots, `N` customers evenly assigned, complete super-graph.

    Travel times scale with absolute id-difference for reproducibility;
    risk/hierarchy are small, uniform values.  Call `make_spiky_risk(inst, ...)`
    to introduce a realistic tension between F₁ and R̄.
    """
    depots = [
        Depot(depot_id=-1, osm_node=1001, lat=0.0, lon=0.0, brand="Blinkit"),
        Depot(depot_id=-2, osm_node=1002, lat=0.01, lon=0.01, brand="Blinkit"),
    ]
    customers: list[Customer] = []
    for i in range(N):
        # Stagger order placement to give the GA some room to respect ETA.
        e_i = float(i) * 1.0
        customers.append(
            Customer(
                customer_id=i,
                osm_node=2000 + i,
                lat=0.001 * i,
                lon=0.001 * i,
                demand=1,
                e_i=e_i,
                eta_i=e_i + 10.0,
                service_time=2.0,
                home_depot=depots[i % 2].depot_id if not cross_depot_noise else depots[0].depot_id,
            )
        )

    # Build a dense super-arc matrix.  Customer-customer arcs use |Δid| as
    # the base travel time; depot→customer arcs use 2 + 0.5·id.
    arcs: dict[tuple[int, int], SuperArc] = {}
    all_nodes: list[int] = [d.depot_id for d in depots] + [c.customer_id for c in customers]
    for u in all_nodes:
        for v in all_nodes:
            if u == v:
                continue
            if u < 0 and v >= 0:
                T = 2.0 + 0.5 * v
            elif u >= 0 and v < 0:
                T = 2.0 + 0.5 * u
            elif u < 0 and v < 0:
                T = 3.0  # depot-to-depot (unused but present for contract)
            else:
                T = abs(u - v) * 1.0 + 1.0
            arcs[(u, v)] = SuperArc(u=u, v=v, T_uv=T, R_uv=0.02, H_uv=0)

    return Instance(
        city="synthetic",
        depots=depots,
        customers=customers,
        street_graph=nx.MultiDiGraph(),
        super_arcs=arcs,
        Q=Q,
        T_max=T_max,
        H_cap_route=H_cap_route,
        R_bar=R_bar,
        H_bar=H_bar,
        seed=seed,
    )


def make_spiky_risk(inst: Instance, high_arcs: list[tuple[int, int]], high_R: float = 0.8) -> None:
    """In-place: raise R_uv on specified arcs to stress the risk budget."""
    for u, v in high_arcs:
        if (u, v) in inst.super_arcs:
            old = inst.super_arcs[(u, v)]
            inst.super_arcs[(u, v)] = SuperArc(u=u, v=v, T_uv=old.T_uv, R_uv=high_R, H_uv=old.H_uv)
