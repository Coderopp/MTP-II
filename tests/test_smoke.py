"""Smoke tests — confirm the scaffolding imports cleanly and enforces contracts.

These tests run without any external data or OSM download.  They MUST keep
passing for the duration of the refactor.
"""

from __future__ import annotations

import pytest


def test_package_imports():
    import savrptw
    import savrptw.types
    import savrptw.eval
    import savrptw.solvers
    import savrptw.graph
    import savrptw.risk
    import savrptw.congestion
    import savrptw.data
    import savrptw.instance
    import savrptw.sim
    import savrptw.viz

    assert savrptw.__version__ == "0.1.0"


def test_solver_registry_populated():
    # Importing the concrete solver modules triggers @register.
    import savrptw.solvers.milp  # noqa: F401
    import savrptw.solvers.ga  # noqa: F401
    import savrptw.solvers.alns  # noqa: F401

    from savrptw.solvers.base import _REGISTRY, build

    assert set(_REGISTRY) == {"milp", "ga", "alns"}
    # build() returns a Solver instance of the requested type.
    for name in ("milp", "ga", "alns"):
        s = build(name, {})
        assert s.name == name


def test_stubbed_solvers_refuse_to_run():
    """Unimplemented solvers MUST raise NotImplementedError, never return a
    fake Solution.  Once a solver is wired (e.g. GA), drop it from this list.
    """
    from savrptw.solvers import build

    stubbed: tuple[str, ...] = ()
    for name in stubbed:
        s = build(name, {})
        with pytest.raises(NotImplementedError):
            s.solve(instance=None)  # type: ignore[arg-type]


def test_uncalibrated_risk_module_refuses():
    from savrptw.risk import attach_risk

    with pytest.raises(RuntimeError):
        attach_risk(None, None)  # type: ignore[arg-type]


def test_feasibility_validator_returns_report():
    """The validator is wired — it returns a ValidationReport, never NotImplemented."""
    import networkx as nx

    from savrptw.eval.feasibility import ValidationReport, validate
    from savrptw.types import Depot, Instance

    empty = Instance(
        city="test",
        depots=[Depot(depot_id=0, osm_node=0, lat=0, lon=0, brand="Blinkit")],
        customers=[],
        street_graph=nx.MultiDiGraph(),
        super_arcs={},
        Q=2,
        T_max=35.0,
        H_cap_route=8,
        R_bar=1.0,
        H_bar=1000,
    )
    from savrptw.types import Solution

    empty_sol = Solution(routes=[], objective=0.0, constraint_summary={}, solver="test")
    rep = validate(empty, empty_sol)
    assert isinstance(rep, ValidationReport)
    # No customers, no routes → trivially feasible.
    assert rep.feasible


def test_types_dataclasses_construct():
    """Cheap contract check on the Instance/Solution schema."""
    from savrptw.types import Customer, Depot, Route, Solution

    d = Depot(depot_id=0, osm_node=1, lat=12.97, lon=77.59, brand="Blinkit")
    c = Customer(
        customer_id=1,
        osm_node=2,
        lat=12.98,
        lon=77.60,
        demand=1,
        e_i=0.0,
        eta_i=10.0,
        service_time=2.0,
        home_depot=0,
    )
    r = Route(rider_id=0, depot_id=0, nodes=[0, 1, 0], arrivals=[0.0, 3.2, 6.4])
    s = Solution(routes=[r], objective=0.0, constraint_summary={}, solver="ga")

    assert d.brand == "Blinkit"
    assert c.eta_i - c.e_i == 10.0
    assert s.routes[0].customers_visited() == [1]


def test_hydra_config_composes():
    """Top-level config should compose with every city × every solver."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    conf_dir = str(Path(__file__).resolve().parents[1] / "conf")
    cities = ["bengaluru", "delhi", "gurugram", "mumbai", "pune"]
    solvers = ["milp", "ga", "alns"]
    with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
        for city in cities:
            for solver in solvers:
                cfg = compose(
                    config_name="config",
                    overrides=[f"city={city}", f"solver={solver}"],
                )
                assert cfg.city.slug == city
                assert cfg.solver.name == solver
                assert cfg.problem.Q == 2
                assert cfg.problem.T_max == 35.0
