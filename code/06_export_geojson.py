"""
06_export_geojson.py — Export pipeline data to web-ready formats.

Outputs (written to web/static/data/)
--------------------------------------
network.geojson   — road edges with t_ij, r_ij, c_ij as properties
instance.json     — depot + customers (copy of 04_vrptw_instance.json)
graph_vis.json    — Vis.js nodes/edges for the abstract graph explorer
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, load_graph

REPO_ROOT   = DATA_DIR.parent
WEB_DATA    = REPO_ROOT / "web" / "static" / "data"
WEB_DATA.mkdir(parents=True, exist_ok=True)

GRAPH_PATH    = DATA_DIR / "03_graph_final.graphml"
INSTANCE_PATH = DATA_DIR / "04_vrptw_instance.json"

# limits to keep the GeoJSON download reasonable
MAX_EDGES = 8000   # sample of road segments


def export_network_geojson(G) -> None:
    """Convert OSM edges → GeoJSON FeatureCollection."""
    features = []
    edges = list(G.edges(keys=True, data=True))

    # Sort by risk descending so risky edges render on top
    edges.sort(key=lambda e: float(e[3].get("r_ij", 0)), reverse=True)
    edges = edges[:MAX_EDGES]

    for u, v, k, data in edges:
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        coords = [
            [float(u_data["x"]), float(u_data["y"])],
            [float(v_data["x"]), float(v_data["y"])],
        ]
        props = {
            "u": str(u), "v": str(v),
            "t_ij": round(float(data.get("t_ij", 0)), 4),
            "r_ij": round(float(data.get("r_ij", 0)), 4),
            "c_ij": round(float(data.get("c_ij", 1)), 4),
            "length_m": round(float(data.get("length", 0)), 1),
            "name": str(data.get("name", "")),
            "highway": str(data.get("highway", "")),
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        })

    geojson = {"type": "FeatureCollection", "features": features}
    out = WEB_DATA / "network.geojson"
    with open(out, "w") as f:
        json.dump(geojson, f, separators=(",", ":"))
    print(f"  network.geojson → {len(features)} edges  ({out.stat().st_size // 1024} KB)")


def export_instance(instance: dict) -> None:
    """Copy instance JSON to web static folder."""
    out = WEB_DATA / "instance.json"
    with open(out, "w") as f:
        json.dump(instance, f, indent=2)
    print(f"  instance.json → depot + {len(instance['customers'])} customers")


def export_vis_graph(G, instance: dict) -> None:
    """Build Vis.js Network nodes/edges from the abstract delivery graph."""
    depot = instance["depot"]
    customers = instance["customers"]
    params = instance["parameters"]

    nodes = []
    edges_vis = []

    # Depot node
    nodes.append({
        "id": "depot",
        "label": "🏪 Depot",
        "title": f"Dark Store<br>Lat: {depot['lat']}<br>Lon: {depot['lon']}",
        "color": {"background": "#00d4ff", "border": "#0099bb"},
        "shape": "star",
        "size": 28,
        "font": {"color": "#ffffff", "size": 14, "bold": True},
        "lat": depot["lat"], "lon": depot["lon"],
    })

    # Customer nodes
    max_tt = max(c["tt_from_depot"] for c in customers) or 1
    for i, c in enumerate(customers):
        urgency = c["tt_from_depot"] / max_tt   # 0=close, 1=far
        # Color: green (close) → orange → red (far, tight TW)
        r = int(urgency * 220)
        g = int((1 - urgency) * 200)
        color = f"rgb({r},{g},60)"
        slack = c["l_i"] - c["tt_from_depot"]
        nodes.append({
            "id": f"c{i}",
            "label": f"C{i+1}",
            "title": (
                f"Customer {i+1}<br>"
                f"TW: [{c['e_i']}–{c['l_i']} min]<br>"
                f"Demand: {c['q_i']} units<br>"
                f"Travel from depot: {c['tt_from_depot']:.1f} min<br>"
                f"TW slack: {slack:.1f} min"
            ),
            "color": {"background": color, "border": "#555"},
            "shape": "dot",
            "size": 10 + c["q_i"] * 4,   # size ∝ demand
            "font": {"color": "#ffffff", "size": 11},
            "lat": c["lat"], "lon": c["lon"],
            "demand": c["q_i"],
            "e_i": c["e_i"], "l_i": c["l_i"],
            "tt": c["tt_from_depot"],
        })
        # Edge from depot → customer (dashed, showing earliest feasible link)
        edges_vis.append({
            "from": "depot",
            "to": f"c{i}",
            "label": f"{c['tt_from_depot']:.1f}m",
            "dashes": True,
            "color": {"color": "#444466"},
            "font": {"color": "#888", "size": 9},
            "arrows": "to",
        })

    vis_data = {
        "nodes": nodes,
        "edges": edges_vis,
        "options": {
            "physics": {
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {"gravitationalConstant": -60,
                                     "springLength": 120, "damping": 0.4},
            },
            "interaction": {"tooltipDelay": 100},
        },
    }
    out = WEB_DATA / "graph_vis.json"
    with open(out, "w") as f:
        json.dump(vis_data, f, indent=2)
    print(f"  graph_vis.json → {len(nodes)} nodes, {len(edges_vis)} edges")


def main() -> None:
    print("[06] Exporting data for web visualization …")

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Run pipeline steps 1–3 first: {GRAPH_PATH} not found")
    if not INSTANCE_PATH.exists():
        raise FileNotFoundError(
            f"Run pipeline step 4 first: {INSTANCE_PATH} not found")

    G = load_graph(GRAPH_PATH)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    with open(INSTANCE_PATH) as f:
        instance = json.load(f)

    export_network_geojson(G)
    export_instance(instance)
    export_vis_graph(G, instance)

    print(f"[06] Done. Web data written to {WEB_DATA}\n")


if __name__ == "__main__":
    main()
