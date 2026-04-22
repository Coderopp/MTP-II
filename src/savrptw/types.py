"""Core data contracts — Instance and Solution.

These dataclasses are the single exchange format between every component of the
pipeline (graph loader → instance generator → solvers → evaluator → viz).
Every solver MUST consume `Instance` and return `Solution`; no other shape is
accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Depot:
    """A dark-store depot snapped to an OSM node."""

    depot_id: int
    osm_node: int
    lat: float
    lon: float
    brand: str  # "Blinkit" for the primary experiments; retained for provenance.


@dataclass(frozen=True)
class Customer:
    """A single delivery order in the dispatch batch."""

    customer_id: int
    osm_node: int
    lat: float
    lon: float
    demand: int          # q_i  (items)
    e_i: float           # order placement time (min from batch start)
    eta_i: float         # promised arrival time (= e_i + 10)
    service_time: float  # s_i  (min)  — stratified 2 or 4 per §10.5
    home_depot: int      # depot_id assigned at instance-generation time


@dataclass
class SuperArc:
    """Precomputed super-graph arc between two customer/depot nodes.

    An arc carries the aggregate quantities of the shortest-time street-graph
    path between its endpoints (see §3.2 of FORMULATION.md).
    """

    u: int  # origin (customer_id or depot_id; semantic tag handled upstream)
    v: int
    T_uv: float    # minutes
    R_uv: float    # dimensionless log-survival
    H_uv: int      # residential-edge count


@dataclass
class Instance:
    """Complete SA-VRPTW instance ready for any solver.

    Attributes
    ----------
    city : str
        City slug ("bengaluru", "delhi", ...).
    depots : list[Depot]
    customers : list[Customer]
    street_graph : nx.MultiDiGraph
        The underlying enriched OSM graph with per-edge `t_ij`, `r_ij`, `h_ij`.
    super_arcs : dict[tuple[int, int], SuperArc]
        Dense lookup keyed by `(node_a, node_b)` where node keys draw from a
        single namespace covering both depots and customers.
    Q : int
        Vehicle capacity.
    T_max : float
        Max route duration (min).
    H_cap_route : int
        Per-route residential-edge cap  H̄_k.
    R_bar : float
        ε-constraint: per-route risk budget.  Swept across experiments.
    H_bar : int
        ε-constraint: fleet residential-edge budget.  Swept across experiments.
    beta_stw : float
        Lateness-penalty exponent (default 0.12).
    w_early : float
        Idle-wait weight in F₁.
    seed : int
        Master RNG seed used to build this instance.
    meta : dict[str, Any]
        Provenance — config hash, git SHA, data-file hashes.
    """

    city: str
    depots: list[Depot]
    customers: list[Customer]
    street_graph: nx.MultiDiGraph
    super_arcs: dict[tuple[int, int], SuperArc]
    Q: int
    T_max: float
    H_cap_route: int
    R_bar: float
    H_bar: int
    beta_stw: float = 0.12
    w_early: float = 1.0
    seed: int = 42
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def N(self) -> int:
        return len(self.customers)

    @property
    def n_depots(self) -> int:
        return len(self.depots)


@dataclass
class Route:
    """A single rider's tour.

    `nodes` is a sequence starting and ending at the same depot id, with
    customer ids strictly in between.  `arrivals[i]` is the wall-clock arrival
    at `nodes[i]` (minutes from batch start).
    """

    rider_id: int
    depot_id: int
    nodes: list[int]
    arrivals: list[float]

    def customers_visited(self) -> list[int]:
        return self.nodes[1:-1]


@dataclass
class Solution:
    """Solver output.

    A Solution is NOT accepted as feasible until it passes
    `savrptw.eval.feasibility.validate(instance, solution)`.
    """

    routes: list[Route]
    objective: float           # F₁
    constraint_summary: dict[str, float]  # populated by evaluator
    solver: str                # "milp" | "ga" | "alns"
    run_meta: dict[str, Any] = field(default_factory=dict)

    def unserved_customers(self, instance: Instance) -> list[int]:
        served = {c for r in self.routes for c in r.customers_visited()}
        return [c.customer_id for c in instance.customers if c.customer_id not in served]


@dataclass
class Status:
    """Solver termination status."""

    feasible: bool
    optimal: bool
    reason: str = ""
    mip_gap: float | None = None
    wall_clock_s: float = 0.0


def _placeholder_unused() -> None:  # pragma: no cover
    # Force numpy import to stay — used downstream by super-arc builders.
    _ = np.zeros(0)
