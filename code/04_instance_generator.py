"""
04_instance_generator.py — Step 4: Build the MD-VRPTW problem instance.

Reads the fully-enriched graph (Step 3) and synthesises:
  - 3 depot nodes (dark stores)
  - N_CUSTOMERS customer nodes (randomly sampled from the OSM graph)
  - Per-customer time windows [e_i, l_i] with a 15-min SLA, demand q_i
  - Fleet parameters: K riders scaled as max(ceil(N / 1.5), 1),
    Q = 2 units / rider, distributed across the 3 dark stores
  - Lambda weights for the objective function: [0.4, 0.4, 0.2]

Outputs
-------
data/04_vrptw_instance.json   — full problem instance
data/04_vrptw_instance.pkl    — same, as a Python dict (faster to load in solver)
"""

import json
import math
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
INPUT_GRAPH = DATA_DIR / "03_graph_final.graphml"
OUT_JSON = DATA_DIR / "04_vrptw_instance.json"
OUT_PKL = DATA_DIR / "04_vrptw_instance.pkl"

# Problem parameters (paper-aligned)
N_CUSTOMERS = 30                # rolling-horizon total customers (override via CLI)
VEHICLE_CAPACITY = 2            # Q = 2 units / rider (quick-commerce thermal bag)
DEMAND_MIN = 1
DEMAND_MAX = 2                  # q_i in {1, 2}; q_i <= Q is always feasible

TW_WINDOW_LENGTH = 15           # hyper-strict 15-min SLA: l_i - e_i = 15

LAMBDA = [0.40, 0.40, 0.20]     # [time, risk, congestion/STW]

DEPOTS = [
    {"name": "DarkStore_TechMarket",  "lat": 22.3190, "lon": 87.3095},
    {"name": "DarkStore_Prembazar",   "lat": 22.3350, "lon": 87.3150},
    {"name": "DarkStore_GoleBazaar",  "lat": 22.3420, "lon": 87.3220},
]

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Fleet sizing
# ---------------------------------------------------------------------------
def fleet_size(n_customers: int) -> int:
    """
    K = max(ceil(N / 1.5), 1) — scaled so a rider with Q=2 can serve roughly
    1.5 orders before returning to depot, accounting for 15-min SLAs.
    """
    return max(math.ceil(n_customers / 1.5), 1)


def distribute_riders_across_depots(k_total: int, depots: list) -> dict:
    """
    Split K riders across the D available dark stores as evenly as possible.
    Returns {depot_node_id: [rider_0, rider_1, ...]}.
    """
    d = len(depots)
    assert d > 0, "Need at least one depot"
    base, remainder = divmod(k_total, d)
    counts = [base + (1 if i < remainder else 0) for i in range(d)]

    mapping = {}
    rider = 0
    for depot, n_riders in zip(depots, counts):
        ids = list(range(rider, rider + n_riders))
        mapping[depot["node_id"]] = ids
        rider += n_riders
    return mapping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> str:
    import osmnx as ox
    node_id = ox.distance.nearest_nodes(G, lon, lat)
    return str(node_id)


def get_edge_attr(G: nx.MultiDiGraph, u, v, attr: str, default: float = 0.0) -> float:
    parallel = G[u][v]
    if not parallel:
        return default
    best = min(parallel.values(), key=lambda e: e.get("t_ij", float("inf")))
    return float(best.get(attr, default))


def compute_travel_time(G: nx.MultiDiGraph, u, v) -> float:
    try:
        path = nx.shortest_path(G, u, v, weight="t_ij")
        return round(
            sum(get_edge_attr(G, path[i], path[i + 1], "t_ij") for i in range(len(path) - 1)),
            4,
        )
    except nx.NetworkXNoPath:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_customers: int = N_CUSTOMERS) -> None:
    rng = random.Random(RANDOM_SEED)

    print("[04] Loading enriched graph ...")
    G = load_graph(INPUT_GRAPH)
    all_nodes = list(G.nodes())
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # --- Depot nodes --------------------------------------------------------
    depots = []
    for d in DEPOTS:
        try:
            d_node = nearest_node(G, d["lat"], d["lon"])
            depots.append({
                "name": d["name"],
                "node_id": str(d_node),
                "lat": round(G.nodes[str(d_node)]["y"], 6),
                "lon": round(G.nodes[str(d_node)]["x"], 6),
                "e_0": 0,
                "l_0": 120,
            })
        except Exception:
            continue

    depot_ids = [d["node_id"] for d in depots]
    assert depots, "No reachable depots within graph."

    # --- Sample customer nodes (exclude depots) -----------------------------
    candidate_nodes = [n for n in all_nodes if n not in depot_ids]
    sampled_nodes = rng.sample(candidate_nodes, n_customers)

    # --- Build customer records (rolling horizon) ---------------------------
    customers = []
    skipped = 0
    current_time_tick = 0

    for node_id in sampled_nodes:
        best_depot, min_tt = None, float("inf")
        for d in depots:
            tt = compute_travel_time(G, d["node_id"], node_id)
            if tt is not None and tt < min_tt:
                min_tt, best_depot = tt, d["node_id"]
        if best_depot is None:
            skipped += 1
            continue

        e_i = current_time_tick
        l_i = e_i + TW_WINDOW_LENGTH         # strict 15-min SLA
        q_i = rng.randint(DEMAND_MIN, DEMAND_MAX)   # q_i in {1, 2}; q_i <= Q

        customers.append({
            "node_id":        str(node_id),
            "assigned_depot": str(best_depot),
            "lat":            round(G.nodes[str(node_id)]["y"], 6),
            "lon":            round(G.nodes[str(node_id)]["x"], 6),
            "e_i":            e_i,
            "l_i":            l_i,
            "q_i":            q_i,
            "tt_from_depot":  min_tt,
        })
        current_time_tick += rng.randint(1, 3)

    print(f"  Customers generated: {len(customers)} (skipped unreachable: {skipped})")

    # --- Fleet sizing (scaled to N) & depot distribution --------------------
    k_total = fleet_size(len(customers))
    depot_riders = distribute_riders_across_depots(k_total, depots)

    for d in depots:
        d["riders"] = depot_riders[d["node_id"]]
        d["num_riders"] = len(d["riders"])
    print(f"  Fleet: K={k_total} riders distributed across {len(depots)} depots "
          f"({[d['num_riders'] for d in depots]})")

    # --- Assemble instance --------------------------------------------------
    instance = {
        "metadata": {
            "description":      "SA-VRPTW Multi-Depot Rolling-Horizon Instance",
            "n_customers":      len(customers),
            "k_riders":         k_total,
            "vehicle_capacity": VEHICLE_CAPACITY,
            "lambda":           LAMBDA,
            "seed":             RANDOM_SEED,
            "type":             "MD-VRPTW",
        },
        "graph_path":    str(INPUT_GRAPH.relative_to(DATA_DIR.parent)),
        "depots":        depots,
        "customers":     customers,
        "depot_riders":  depot_riders,       # explicit depot -> rider-id list mapping
        "parameters": {
            "Q":       VEHICLE_CAPACITY,
            "K":       k_total,
            "N":       len(customers),
            "lambda1": LAMBDA[0],
            "lambda2": LAMBDA[1],
            "lambda3": LAMBDA[2],
            "tw_window": TW_WINDOW_LENGTH,
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(instance, f, indent=2)
    print(f"  JSON  -> {OUT_JSON}")
    with open(OUT_PKL, "wb") as f:
        pickle.dump(instance, f)
    print(f"  Pickle -> {OUT_PKL}")

    print(f"[04] Done. {len(customers)} customers, {k_total} riders, {len(depots)} depots.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_CUSTOMERS
    main(n)
