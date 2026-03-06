"""
01_osm_graph.py — Step 1: Download OSM road network for Kharagpur.

Strategy (in order of preference):
  1. graph_from_point — 6 km radius from city centre (fast, reliable)
  2. graph_from_place — Nominatim geocoder fallback

The bbox approach is intentionally NOT used as primary — the Kharagpur
bounding box is ~13,000× larger than the Overpass API default query area,
causing very long download times with automatic subdivision.

Outputs
-------
data/01_osm_graph.graphml
    Directed multigraph with edge attributes:
      - length    : segment length (metres)
      - t_ij      : free-flow travel time (minutes)
      - speed_kph : free-flow speed (km/h)
      - r_ij      : placeholder 0.0  (filled in step 02)
      - c_ij      : placeholder 1.0  (filled in step 03)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, save_graph

import osmnx as ox
import networkx as nx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NETWORK_TYPE = "drive"
OUTPUT_PATH  = DATA_DIR / "01_osm_graph.graphml"

# IIT Kharagpur campus / technology market area — city centre
CENTRE_LAT = 22.3200
CENTRE_LON = 87.3190
RADIUS_M   = 6_000     # 6 km covers urban Kharagpur comfortably


# ---------------------------------------------------------------------------
# Acquisition strategies
# ---------------------------------------------------------------------------

def _try_point():
    print("  Strategy 1: graph_from_point (centre + 6 km radius) …")
    try:
        G = ox.graph_from_point(
            (CENTRE_LAT, CENTRE_LON),
            dist=RADIUS_M,
            network_type=NETWORK_TYPE,
        )
        return G
    except Exception as e:
        print(f"  ⚠ graph_from_point failed: {e}")
        return None


def _try_place():
    print("  Strategy 2: graph_from_place (Nominatim) …")
    try:
        G = ox.graph_from_place(
            "Kharagpur, West Bengal, India",
            network_type=NETWORK_TYPE,
        )
        return G
    except Exception as e:
        print(f"  ⚠ graph_from_place failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_graph():
    print("[01] Downloading OSM drive network for Kharagpur …")
    G = _try_point() or _try_place()

    if G is None:
        raise RuntimeError(
            "All OSM download strategies failed. "
            "Check your internet connection and try again."
        )

    print(f"  ✅ Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")

    # Add free-flow speed + travel time from OSM maxspeed tags
    G = ox.add_edge_speeds(G)        # adds 'speed_kph'
    G = ox.add_edge_travel_times(G)  # adds 'travel_time' (seconds)

    # Convert to t_ij in minutes
    for _, _, data in G.edges(data=True):
        data["t_ij"] = round(data.get("travel_time", 0.0) / 60.0, 4)

    # Add placeholder attributes for steps 02 & 03
    for _, _, data in G.edges(data=True):
        data["r_ij"] = 0.0
        data["c_ij"] = 1.0

    return G


def main():
    G = build_graph()
    save_graph(G, OUTPUT_PATH)
    print(f"[01] Done. Graph saved to {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
