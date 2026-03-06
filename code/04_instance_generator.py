"""
04_instance_generator.py — Step 4: Build the VRPTW problem instance.

Reads the fully-enriched graph (Step 3) and synthesises:
  - A depot node (dark store)
  - N_CUSTOMERS customer nodes (randomly sampled from the OSM graph)
  - Per-customer time windows [e_i, l_i], demand q_i
  - Fleet parameters: K riders, capacity Q
  - Lambda weights for the objective function

Outputs
-------
data/04_vrptw_instance.json   — full problem instance
data/04_vrptw_instance.pkl    — same, as a Python dict (faster to load in solver)
"""

import json
import pickle
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, load_graph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_GRAPH  = DATA_DIR / "03_graph_final.graphml"
OUT_JSON     = DATA_DIR / "04_vrptw_instance.json"
OUT_PKL      = DATA_DIR / "04_vrptw_instance.pkl"

# Problem parameters
N_CUSTOMERS       = 20      # number of delivery locations
K_RIDERS          = 5       # fleet size
VEHICLE_CAPACITY  = 3       # Q: units per rider (QC reality, usually carried in single bag)
DEMAND_MIN        = 1       # minimum order size
DEMAND_MAX        = 2       # maximum order size

# Time-window generation (minutes from start of dispatch window)
TW_EARLIEST_OPEN  = 0       # earliest any order opens
TW_LATEST_OPEN    = 20      # latest an order can open
TW_WINDOW_LENGTH  = 30      # window width (10–40 min delivery promise)

# Objective weights (lambda_1, lambda_2, lambda_3) — must sum to 1.0
# Default: balanced between time and safety
LAMBDA = [0.40, 0.40, 0.20]    # [time, risk, congestion]

# IIT Kharagpur Technology Market as dark store (OSM approx)
DEPOT_LAT = 22.3190
DEPOT_LON = 87.3095

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> str:
    """Return the OSM node ID (as string) closest to (lat, lon)."""
    import osmnx as ox
    node_id = ox.distance.nearest_nodes(G, lon, lat)
    # nearest_nodes returns int; GraphML loads IDs as strings — cast to match
    return str(node_id)


def get_edge_attr(G: nx.MultiDiGraph, u: int, v: int, attr: str,
                  default: float = 0.0) -> float:
    """Return attr from the first key of a multigraph edge."""
    # When nx.shortest_path is used with a weight, it implicitly selects the
    # edge with the minimum weight for each (u,v) pair.
    # So, when summing up the path, we should retrieve the attribute from
    # the edge that was actually chosen by shortest_path.
    # However, networkx's shortest_path for MultiDiGraph returns a path of nodes,
    # not edges. To get the specific edge data, we need to iterate through
    # the parallel edges and find the one that matches the shortest path criteria.
    # For 't_ij', this means finding the minimum 't_ij' edge.
    
    # Fix: explicitly select the minimal parallel edge weight rather than arbitrary index 0
    parallel_edges = G[u][v]
    if not parallel_edges:
        return default
    
    # Find the edge with the minimum 't_ij' among parallel edges
    # This assumes 't_ij' is the relevant attribute for pathfinding.
    best_edge_data = min(parallel_edges.values(), key=lambda e: e.get('t_ij', float('inf')))
    return float(best_edge_data.get(attr, default))


def compute_travel_time(G: nx.MultiDiGraph, u: int, v: int) -> float:
    """
    Approximate travel time on the SHORTEST PATH (by t_ij) between u and v.
    Returns minutes; returns None if no path exists.
    """
    try:
        path = nx.shortest_path(G, u, v, weight="t_ij")
        total = sum(
            get_edge_attr(G, path[i], path[i + 1], "t_ij")
            for i in range(len(path) - 1)
        )
        return round(total, 4)
    except nx.NetworkXNoPath:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)

    print("[04] Loading enriched graph …")
    G = load_graph(INPUT_GRAPH)
    all_nodes = list(G.nodes())
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # --- Depot node ---------------------------------------------------------
    depot_node = nearest_node(G, DEPOT_LAT, DEPOT_LON)
    print(f"  Depot node (dark store): {depot_node}  "
          f"→  lat={G.nodes[depot_node]['y']:.5f}, "
          f"lon={G.nodes[depot_node]['x']:.5f}")

    # --- Sample customer nodes (exclude depot) -------------------------------
    candidate_nodes = [n for n in all_nodes if n != depot_node]
    sampled_nodes = rng.sample(candidate_nodes, N_CUSTOMERS)

    # --- Build customer records ---------------------------------------------
    customers = []
    skipped = 0
    for node_id in sampled_nodes:
        # Check reachability from depot (quick check via t_ij shortest path)
        tt = compute_travel_time(G, depot_node, node_id)
        if tt is None:
            skipped += 1
            continue

        # FIX: Time window must be dynamically feasible based on travel time from depot
        tt_mins = int(np.ceil(tt))
        min_arrival = max(TW_EARLIEST_OPEN, tt_mins)
        
        e_i = rng.randint(min_arrival, min_arrival + 15)
        l_i = e_i + TW_WINDOW_LENGTH
        q_i = rng.randint(DEMAND_MIN, DEMAND_MAX)

        customers.append({
            "node_id":          str(node_id),   # str to match GraphML keys
            "lat":              round(G.nodes[str(node_id)]["y"], 6),
            "lon":              round(G.nodes[str(node_id)]["x"], 6),
            "e_i":              e_i,
            "l_i":              l_i,
            "q_i":              q_i,
            "tt_from_depot":    tt,
        })

    print(f"  Customers generated: {len(customers)} "
          f"(skipped unreachable: {skipped})")

    # --- Assemble instance dict ---------------------------------------------
    instance = {
        "metadata": {
            "description": "SA-VRPTW instance for Kharagpur q-commerce",
            "n_customers":      len(customers),
            "k_riders":         K_RIDERS,
            "vehicle_capacity": VEHICLE_CAPACITY,
            "lambda":           LAMBDA,
            "seed":             RANDOM_SEED,
        },
        "graph_path": str(INPUT_GRAPH.relative_to(DATA_DIR.parent)),
        "depot": {
            "node_id":  str(depot_node),    # str to match GraphML keys
            "lat":      round(G.nodes[str(depot_node)]["y"], 6),
            "lon":      round(G.nodes[str(depot_node)]["x"], 6),
            "e_0":      0,
            "l_0":      120,
        },
        "customers": customers,
        "parameters": {
            "Q":       VEHICLE_CAPACITY,
            "K":       K_RIDERS,
            "lambda1": LAMBDA[0],
            "lambda2": LAMBDA[1],
            "lambda3": LAMBDA[2],
        },
    }

    # --- Save ---------------------------------------------------------------
    with open(OUT_JSON, "w") as f:
        json.dump(instance, f, indent=2)
    print(f"  JSON → {OUT_JSON}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(instance, f)
    print(f"  Pickle → {OUT_PKL}")

    print(f"[04] Done. Instance: {len(customers)} customers, "
          f"{K_RIDERS} riders, Q={VEHICLE_CAPACITY}\n")


if __name__ == "__main__":
    main()
