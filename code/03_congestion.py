"""
03_congestion.py — Step 3: Assign congestion index c_ij to every edge.

TWO MODES (controlled by CONGESTION_MODE env var or config below):

  A. SPEED_PROXY  (default, no API key needed)
     c_ij derived from OSM free-flow speed data:
       raw = 1 + (1 / speed_kph_normalised)   — slower roads → higher congestion
       scaled to [0, 1]

  B. GOOGLE_MAPS  (requires GOOGLE_MAPS_API_KEY in .env)
     Samples the midpoint of each edge via the Google Maps Distance Matrix API
     during simulated AM peak hours and computes:
       c_ij = live_travel_time / free_flow_travel_time − 1   (capped at [0,1])
     To avoid exhausting quota, edges are sampled (max MAX_API_EDGES) and the
     rest are interpolated from nearest-neighbour edge attributes.

Set CONGESTION_MODE = "GOOGLE_MAPS" in this file or in your .env to activate.

Outputs
-------
data/03_graph_final.graphml
    Graph from step 02 with c_ij (normalised [0,1]) on every edge.
data/03_congestion_metadata.json
    Mode used, edge statistics, API call count (if applicable).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, load_env, load_graph, normalize, save_graph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_env()  # pulls .env into os.environ

INPUT_GRAPH  = DATA_DIR / "02_graph_with_risk.graphml"
OUTPUT_GRAPH = DATA_DIR / "03_graph_final.graphml"
META_PATH    = DATA_DIR / "03_congestion_metadata.json"

# Override with env var: export CONGESTION_MODE=GOOGLE_MAPS
CONGESTION_MODE = os.getenv("CONGESTION_MODE", "SPEED_PROXY").upper()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

# Google Maps API: limit edge sampling to avoid quota exhaustion
MAX_API_EDGES = 200
API_SLEEP_SEC = 0.05   # rate-limit between requests


# ---------------------------------------------------------------------------
# Mode A — Speed-Proxy (offline, no API)
# ---------------------------------------------------------------------------

def congestion_speed_proxy(G) -> dict:
    """
    Derive c_ij from free-flow speed:
      - Slower roads (narrow lanes, intersections) are inherently more congested.
      - raw_c = 1 / normalised_speed  →  then min-max scaled to [0,1].
    """
    def _speed(data):
        val = data.get("speed_kph", 30.0)
        try:
            return max(float(val), 5.0)
        except (TypeError, ValueError):
            return 30.0

    speeds = {
        (u, v, k): _speed(data)
        for u, v, k, data in G.edges(keys=True, data=True)
    }
    speed_series = pd.Series(list(speeds.values()))
    speed_norm   = normalize(speed_series)  # 0 = slowest, 1 = fastest

    # FIX: Ensure we don't structurally penalize residential streets. 
    # Highway=residential means inherently slow limits, but usually uncongested.
    congestion_list = []
    keys_list = list(speeds.keys())
    for i, (u, v, k) in enumerate(keys_list):
        data = G[u][v][k]
        hw = data.get("highway", "")
        if isinstance(hw, list):
            hw = hw[0] if hw else ""
            
        is_residential = hw in ["residential", "living_street", "service", "pedestrian"]
        
        if is_residential:
            # inherently slow but usually empty
            congestion_list.append(np.clip(0.1 + 0.1 * (speed_norm.iloc[i]), 0, 1))
        else:
            # Invert: slowest major roads -> highest congestion
            congestion_list.append(np.clip(1.0 - speed_norm.iloc[i], 0, 1))
            
    congestion_series = pd.Series(congestion_list)

    for i, (u, v, k) in enumerate(speeds.keys()):
        G[u][v][k]["c_ij"] = round(float(congestion_series.iloc[i]), 6)

    stats = {
        "mode":      "SPEED_PROXY",
        "c_ij_mean": round(float(congestion_series.mean()), 4),
        "c_ij_std":  round(float(congestion_series.std()),  4),
        "c_ij_min":  round(float(congestion_series.min()),  4),
        "c_ij_max":  round(float(congestion_series.max()),  4),
    }
    print(f"  [Speed Proxy] c_ij stats: mean={stats['c_ij_mean']:.3f}  "
          f"std={stats['c_ij_std']:.3f}  "
          f"min={stats['c_ij_min']:.3f}  max={stats['c_ij_max']:.3f}")
    return stats


# ---------------------------------------------------------------------------
# Mode B — Google Maps Distance Matrix API
# ---------------------------------------------------------------------------

def _edge_midpoint(G, u, v, k):
    """Return (lat, lon) midpoint of an OSM edge."""
    geom = G[u][v][k].get("geometry")
    if geom:
        # Shapely geometry coordinates are (lon, lat)
        coords = list(geom.coords)
        mid = coords[len(coords) // 2]
        return mid[1], mid[0]   # (lat, lon)
    # Fallback to midpoint of endpoint nodes
    lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
    lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
    return lat, lon


def _query_distance_matrix(origins: list[str], destinations: list[str],
                            api_key: str, departure_time: int) -> list[float | None]:
    """
    Call Google Maps Distance Matrix API for a batch of origin→destination pairs.
    Returns a list of durations in seconds (None on error/no route).
    """
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        "origins":         "|".join(origins),
        "destinations":    "|".join(destinations),
        "mode":            "driving",
        "departure_time":  departure_time,
        "traffic_model":   "best_guess",
        "key":             api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    durations = []
    for row in data.get("rows", []):
        for elem in row.get("elements", []):
            if elem.get("status") == "OK":
                # duration_in_traffic is live; duration is free-flow
                live = elem.get("duration_in_traffic", {}).get("value")
                ff   = elem.get("duration",             {}).get("value")
                durations.append((live, ff))
            else:
                durations.append((None, None))
    return durations


def congestion_google_maps(G, api_key: str) -> dict:
    """
    Sample MAX_API_EDGES edges, fetch live vs free-flow travel time via
    Google Maps Distance Matrix, then interpolate remaining edges via
    nearest-neighbour in (lat,lon) space.
    """
    import random
    from scipy.spatial import KDTree  # built-in with scipy

    all_edges = list(G.edges(keys=True, data=True))
    sample_size = min(MAX_API_EDGES, len(all_edges))
    sampled = random.sample(all_edges, sample_size)

    # AM peak timestamp (~8:00 AM IST next weekday, as Unix UTC)
    import datetime
    now = datetime.datetime.utcnow()
    # Move to next Monday 02:30 UTC (= 08:00 IST)
    days_ahead = (0 - now.weekday()) % 7 or 1
    peak_dt = (now + datetime.timedelta(days=days_ahead)).replace(
        hour=2, minute=30, second=0, microsecond=0)
    departure_unix = int(peak_dt.timestamp())

    edge_c = {}
    api_calls = 0
    errors = 0
    batch_size = 10  # max per API call

    print(f"  [Google Maps] Sampling {sample_size} edges (batch={batch_size}) …")
    for i in range(0, sample_size, batch_size):
        batch = sampled[i: i + batch_size]
        origins, dests = [], []
        for u, v, k, data in batch:
            lat_u, lon_u = G.nodes[u]["y"], G.nodes[u]["x"]
            lat_v, lon_v = G.nodes[v]["y"], G.nodes[v]["x"]
            origins.append(f"{lat_u},{lon_u}")
            dests.append(f"{lat_v},{lon_v}")

        try:
            results = _query_distance_matrix(origins, dests, api_key, departure_unix)
            api_calls += 1
            for (u, v, k, _), (live, ff) in zip(batch, results):
                if live and ff and ff > 0:
                    raw_c = max(0.0, (live - ff) / ff)   # congestion ratio
                    edge_c[(u, v, k)] = min(raw_c, 2.0)  # cap at 200% delay
                else:
                    errors += 1
        except Exception as e:
            print(f"    API error on batch {i//batch_size}: {e}")
            errors += batch_size

        time.sleep(API_SLEEP_SEC)

    if not edge_c:
        print("  [Google Maps] No API results — falling back to speed proxy")
        return congestion_speed_proxy(G)

    # Normalise sampled c values to [0,1]
    known_keys = list(edge_c.keys())
    known_vals = pd.Series([edge_c[k] for k in known_keys])
    known_norm = normalize(known_vals)
    sampled_c_norm = dict(zip(known_keys, known_norm.values))

    # FIX: Instead of Euclidean KDTree which ignores civic topology (rivers, railways),
    # we use Shortest Path Graph distance from sampled nodes to propagate congestion.
    import networkx as nx
    
    # Map each node that is part of a sampled edge to the congestion value
    node_to_c = {}
    for u, v, k in known_keys:
        node_to_c[u] = sampled_c_norm[(u, v, k)]
        node_to_c[v] = sampled_c_norm[(u, v, k)]
        
    sampled_nodes = list(node_to_c.keys())
    nearest_sampled_mapping = {}
    
    if sampled_nodes:
        # Add a dummy super-source connected to all sampled_nodes to run multi-source shortest path
        G_dummy = G.copy()
        G_dummy.add_node("SUPER_SOURCE")
        for sn in sampled_nodes:
            G_dummy.add_edge("SUPER_SOURCE", sn, key=0, length=0)
        
        try:
            lengths, paths = nx.single_source_dijkstra(G_dummy, "SUPER_SOURCE", weight="length")
            for node in G.nodes():
                if node in paths and len(paths[node]) > 1:
                    nearest_sn = paths[node][1]
                    nearest_sampled_mapping[node] = node_to_c[nearest_sn]
                else:
                    nearest_sampled_mapping[node] = 0.5 # fallback
        except Exception:
            for node in G.nodes(): nearest_sampled_mapping[node] = 0.5

    # Assign c_ij to ALL edges
    for u, v, k, data in G.edges(keys=True, data=True):
        if (u, v, k) in sampled_c_norm:
            c = sampled_c_norm[(u, v, k)]
        else:
            # Average the inherited congestion of the two endpoints
            c_u = nearest_sampled_mapping.get(u, 0.5)
            c_v = nearest_sampled_mapping.get(v, 0.5)
            c = (c_u + c_v) / 2.0
        G[u][v][k]["c_ij"] = round(float(c), 6)

    all_c = [d.get("c_ij", 0.5) for _, _, d in G.edges(data=True)]
    stats = {
        "mode":           "GOOGLE_MAPS",
        "api_calls":      api_calls,
        "edges_sampled":  sample_size,
        "api_errors":     errors,
        "c_ij_mean":      round(float(np.mean(all_c)), 4),
        "c_ij_std":       round(float(np.std(all_c)),  4),
        "c_ij_min":       round(float(np.min(all_c)),  4),
        "c_ij_max":       round(float(np.max(all_c)),  4),
        "departure_time": peak_dt.isoformat(),
    }
    print(f"  [Google Maps] Done. API calls: {api_calls}, "
          f"errors: {errors}, c_ij mean: {stats['c_ij_mean']:.3f}")
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[03] Loading graph from Step 2 …")
    G = load_graph(INPUT_GRAPH)
    print(f"  CONGESTION_MODE = {CONGESTION_MODE}")

    if CONGESTION_MODE == "GOOGLE_MAPS":
        if not GOOGLE_MAPS_API_KEY:
            print("  WARNING: GOOGLE_MAPS_API_KEY not set in .env — "
                  "falling back to SPEED_PROXY")
            stats = congestion_speed_proxy(G)
        else:
            stats = congestion_google_maps(G, GOOGLE_MAPS_API_KEY)
    else:
        stats = congestion_speed_proxy(G)

    save_graph(G, OUTPUT_GRAPH)

    with open(META_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Metadata → {META_PATH}")
    print(f"[03] Done. c_ij added to all edges.\n")


if __name__ == "__main__":
    main()
