"""Build the customer/depot super-graph from an enriched street graph.

For each pair of (customer, depot) super-nodes we compute:

    T_uv = Σ t_ij         over the shortest-time path on the street graph
    R_uv = Σ -ln(1 - r_ij) "       " (log-survival)
    H_uv = Σ h_ij         "       " (residential-edge count)

Shortest paths are computed once per super-node via Dijkstra on the street
graph with `t_ij` weights.  The same path is used to aggregate R and H, so
all three quantities correspond to the *same* physical route (a requirement
for consistency with FORMULATION.md §3.2).
"""

from __future__ import annotations

import math

import networkx as nx

from savrptw.types import SuperArc


def build_super_arcs(
    G: nx.MultiDiGraph,
    super_nodes: list[tuple[int, int]],
) -> dict[tuple[int, int], SuperArc]:
    """
    Parameters
    ----------
    G
        Street graph with per-edge `t_ij`, `r_ij`, `h_ij` attached.
    super_nodes
        List of (external_id, osm_node) pairs — the depots and customers in
        a single integer namespace.  `external_id` is what appears in
        `savrptw.types.Depot.depot_id` / `Customer.customer_id`.

    Returns
    -------
    dict keyed by (external_id_u, external_id_v) → SuperArc.
    """
    # Sanity-check edge attributes up front — refuse silently-degraded graphs.
    sample = next(iter(G.edges(keys=True, data=True)), None)
    if sample is None:
        raise ValueError("street graph is empty")
    _, _, _, sdata = sample
    for attr in ("t_ij", "r_ij", "h_ij"):
        if attr not in sdata:
            raise ValueError(
                f"edge attribute {attr!r} missing — run graph loader + congestion "
                f"+ risk attachment before building super-arcs"
            )

    # Precompute a single-weight DiGraph view for Dijkstra.  MultiDiGraph
    # edges are aggregated to the min-`t_ij` parallel edge (FORMULATION.md
    # §15).  Risk/h are attached from the same chosen parallel edge so
    # aggregated quantities correspond to one coherent physical path.
    simple = nx.DiGraph()
    for u, v, _k, data in G.edges(keys=True, data=True):
        t = float(data["t_ij"])
        if simple.has_edge(u, v):
            if simple[u][v]["t_ij"] <= t:
                continue
        simple.add_edge(
            u,
            v,
            t_ij=t,
            r_ij=float(data["r_ij"]),
            h_ij=int(data["h_ij"]),
        )

    arcs: dict[tuple[int, int], SuperArc] = {}
    for ext_u, osm_u in super_nodes:
        # Single-source Dijkstra on t_ij gives shortest paths to everything.
        try:
            _dist, paths = nx.single_source_dijkstra(
                simple, osm_u, weight="t_ij"
            )
        except nx.NodeNotFound as e:  # pragma: no cover
            raise ValueError(f"super-node OSM id {osm_u} not in graph") from e
        for ext_v, osm_v in super_nodes:
            if ext_u == ext_v:
                continue
            if osm_v not in paths:
                continue  # disconnected — arc omitted; solvers must handle it
            path = paths[osm_v]
            if len(path) < 2:
                continue
            T = 0.0
            R = 0.0
            H = 0
            for i in range(len(path) - 1):
                e = simple[path[i]][path[i + 1]]
                T += float(e["t_ij"])
                # r_ij ∈ [0, 0.99] — clamp defensively.
                r = max(0.0, min(0.99, float(e["r_ij"])))
                R += -math.log(1.0 - r)
                H += int(e["h_ij"])
            arcs[(ext_u, ext_v)] = SuperArc(u=ext_u, v=ext_v, T_uv=T, R_uv=R, H_uv=H)
    return arcs
