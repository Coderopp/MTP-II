"""Instance generator — orchestrates graph + risk + congestion + depots + customers.

End-to-end pipeline (FORMULATION.md §11):

    1. Load OSM drive network for the city (bbox-bounded).
    2. Attach BPR congestion -> `t_ij` on every edge.
    3. Attach calibrated crash probability -> `r_ij`   ← BLOCKED on Task #16.
    4. Parse Blinkit KML for the city, k-means cluster to `n_depots`.
    5. Sample `N` customer nodes weighted by residential-land-use density.
    6. Assign each customer to its nearest depot by free-flow time.
    7. Stratify service times (2 min default / 4 min high-rise).
    8. Compute super-arcs among (depots ∪ customers).
    9. Package everything into `savrptw.types.Instance`.

Until Task #16 selects a calibrated risk source, step 3 raises
`NotImplementedError` — no synthetic r_ij is ever used in production.
"""

from __future__ import annotations

import random
from pathlib import Path

import networkx as nx
import osmnx as ox
from omegaconf import DictConfig

from savrptw.congestion.bpr import attach_congestion
from savrptw.data.darkstores import load_blinkit_stores
from savrptw.graph.osm import load_city_graph, snap_to_node
from savrptw.instance.super_arc import build_super_arcs
from savrptw.risk.basm import attach_risk
from savrptw.types import Customer, Depot, Instance


# External id namespaces — negative ids for depots, non-negative for customers.
def _depot_id(idx: int) -> int:
    return -(idx + 1)


def _customer_id(idx: int) -> int:
    return idx


def _sample_customers_weighted(
    G: nx.MultiDiGraph,
    n: int,
    rng: random.Random,
) -> list[int]:
    """Sample `n` OSM nodes with residential-land-use weighting.

    Proxy (Task #13 v1): a node gets weight 1 if *any* adjacent edge is
    residential, else 0.25 (a small background weight so isolated customer
    nodes can still be sampled on non-residential outliers).  A richer
    WorldPop-based weighting can plug in here later without changing the
    Instance contract (FORMULATION.md §10.6).
    """
    nodes = list(G.nodes())
    weights: list[float] = []
    for n_id in nodes:
        adj_residential = any(
            G[n_id][v][k].get("h_ij", 0) == 1 for v in G.successors(n_id) for k in G[n_id][v]
        )
        weights.append(1.0 if adj_residential else 0.25)
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("no residential-adjacent nodes found in graph")
    # Reservoir sampling without replacement weighted by `weights`.
    chosen: list[int] = []
    pool = list(zip(nodes, weights, strict=True))
    for _ in range(n):
        if not pool:
            break
        running = 0.0
        r = rng.random() * sum(w for _, w in pool)
        for i, (node, w) in enumerate(pool):
            running += w
            if r <= running:
                chosen.append(node)
                pool.pop(i)
                break
    if len(chosen) < n:
        raise ValueError(f"could not sample {n} customers — graph too small")
    return chosen


def build_instance(cfg: DictConfig) -> Instance:
    """Materialise an Instance from a composed Hydra config.

    Required config keys:

        cfg.city.*                — slug, bbox, osm_place, kml_folders
        cfg.risk                  — BASM config (currently raises until Task #16)
        cfg.congestion            — BPR config
        cfg.problem.Q, T_max, …   — structural constants
        cfg.instance.N            — customers per instance
        cfg.instance.n_depots     — k-means cluster count
        cfg.instance.seed         — RNG seed for sampling
        cfg.paths.raw_data        — location of the committed KML
        cfg.paths.osmnx_cache     — OSMnx cache dir
    """
    seed = int(cfg.instance.seed)
    rng = random.Random(seed)

    cache = Path(cfg.paths.osmnx_cache) if "osmnx_cache" in cfg.paths else None

    # 1) OSM drive graph.
    G = load_city_graph(cfg.city, cache_dir=cache)

    # 2) BPR congestion.
    attach_congestion(G, cfg.congestion)

    # 3) Calibrated r_ij.  RAISES until Task #16 lands.
    attach_risk(G, cfg.risk)

    # 4) Dark stores → k-means → depots.
    kml_path = Path(cfg.paths.raw_data) / "darkstoremap_in_2026-04-09.kml"
    stores = load_blinkit_stores(kml_path, cfg.city, n_clusters=cfg.instance.n_depots, seed=seed)
    depots: list[Depot] = []
    for idx, c in enumerate(stores["depot_centroids"]):
        osm = snap_to_node(G, lat=c["lat"], lon=c["lon"])
        depots.append(
            Depot(
                depot_id=_depot_id(idx),
                osm_node=int(osm),
                lat=float(c["lat"]),
                lon=float(c["lon"]),
                brand="Blinkit",
            )
        )

    # 5) Customers.
    sampled = _sample_customers_weighted(G, int(cfg.instance.N), rng)
    customers: list[Customer] = []
    for i, osm_id in enumerate(sampled):
        node_data = G.nodes[osm_id]
        # Tight stratified service time — FORMULATION.md §10.5 richer version
        # (OSM building-polygon query) will slot in here later.
        service = 2.0
        e_i = rng.uniform(0.0, 60.0)  # order arrives in the first hour
        eta = e_i + float(cfg.problem.eta_promise_min)
        demand = rng.randint(1, 2)
        # Home depot = nearest by free-flow time.
        home_idx = min(
            range(len(depots)),
            key=lambda di: nx.shortest_path_length(
                G, depots[di].osm_node, osm_id, weight="t_ij_free"
            )
            if nx.has_path(G, depots[di].osm_node, osm_id)
            else float("inf"),
        )
        customers.append(
            Customer(
                customer_id=_customer_id(i),
                osm_node=int(osm_id),
                lat=float(node_data["y"]),
                lon=float(node_data["x"]),
                demand=demand,
                e_i=e_i,
                eta_i=eta,
                service_time=service,
                home_depot=depots[home_idx].depot_id,
            )
        )

    # 6) Super-arcs.
    super_nodes = [(d.depot_id, d.osm_node) for d in depots] + [
        (c.customer_id, c.osm_node) for c in customers
    ]
    super_arcs = build_super_arcs(G, super_nodes)

    # 7) Package.
    return Instance(
        city=cfg.city.slug,
        depots=depots,
        customers=customers,
        street_graph=G,
        super_arcs=super_arcs,
        Q=int(cfg.problem.Q),
        T_max=float(cfg.problem.T_max),
        H_cap_route=int(cfg.problem.H_cap_route),
        R_bar=float(cfg.instance.R_bar),
        H_bar=int(cfg.instance.H_bar),
        beta_stw=float(cfg.problem.beta_stw),
        w_early=float(cfg.problem.w_early),
        seed=seed,
        meta={
            "n_stores_raw": int(stores["n_stores_raw"]),
            "n_depots": int(stores["n_clusters"]),
            "city_slug": cfg.city.slug,
        },
    )
