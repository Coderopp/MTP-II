"""OSMnx-backed city graph loader.

Loads the drive network for a city bounded by its config bbox, fills in
OSM-derived per-edge attributes, and normalises types (OSM often returns
lists for `highway`, `maxspeed`, etc., which breaks clean numerical work).

Returned graph has, on every edge:

* `length` (m, float)
* `highway` (str, single canonical value)
* `speed_kph` (float, OSMnx imputed)
* `t_ij_free` (min, length / speed — travel time at free flow)
* `h_ij` (int, 0/1 — residential-street indicator per FORMULATION.md §3.1)
* `lanes` (int, OSM tag or class default)

Congestion inflation → `savrptw.congestion.bpr.attach_congestion`.
Risk attachment → `savrptw.risk.basm.attach_risk`.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import osmnx as ox
from omegaconf import DictConfig

# Edges with these OSM classes are considered "residential" for the H̄ budget.
_RESIDENTIAL_HIGHWAYS = frozenset({"residential", "living_street"})

# Fallback lane counts when OSM tag missing (mirrors conf/congestion/bpr.yaml).
_DEFAULT_LANES: dict[str, int] = {
    "motorway": 3,
    "trunk": 2,
    "primary": 2,
    "secondary": 1,
    "tertiary": 1,
    "residential": 1,
    "living_street": 1,
    "unclassified": 1,
    "service": 1,
}


def _first(val):
    """OSM sometimes returns a list; take the first element for categorical tags."""
    if isinstance(val, list):
        return val[0] if val else None
    return val


def _coerce_int(val, default: int) -> int:
    v = _first(val)
    if v is None:
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def load_city_graph(city_cfg: DictConfig, cache_dir: Path | None = None) -> nx.MultiDiGraph:
    """Download (or load from OSMnx cache) and enrich a city's drive network.

    Parameters
    ----------
    city_cfg
        Composed Hydra city config with `.osm_place`, `.bbox`.
    cache_dir
        Directory used for OSMnx's HTTP cache.  If None, OSMnx's default is used.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ox.settings.cache_folder = str(cache_dir)
    ox.settings.use_cache = True
    ox.settings.log_console = False

    bbox = city_cfg["bbox"]
    # OSMnx 1.x expects (north, south, east, west).
    G = ox.graph_from_bbox(
        north=float(bbox["lat_max"]),
        south=float(bbox["lat_min"]),
        east=float(bbox["lon_max"]),
        west=float(bbox["lon_min"]),
        network_type="drive",
        simplify=True,
        retain_all=False,
    )

    # Impute speeds; compute travel times.
    G = ox.add_edge_speeds(G)            # adds `speed_kph` to every edge
    G = ox.add_edge_travel_times(G)      # adds `travel_time` (seconds)

    for _u, _v, _k, data in G.edges(keys=True, data=True):
        # Normalise `highway` to a single canonical class.
        hw = _first(data.get("highway"))
        if hw is None:
            hw = "unclassified"
        data["highway"] = str(hw)

        # Canonical edge quantities.
        length_m = float(data.get("length", 0.0))
        speed_kph = float(data.get("speed_kph", 20.0) or 20.0)
        data["length"] = length_m
        data["speed_kph"] = speed_kph
        data["t_ij_free"] = (length_m / 1000.0) / speed_kph * 60.0  # minutes
        data["h_ij"] = 1 if data["highway"] in _RESIDENTIAL_HIGHWAYS else 0
        data["lanes"] = _coerce_int(
            data.get("lanes"), default=_DEFAULT_LANES.get(data["highway"], 1)
        )

    return G


def snap_to_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Return the OSM node id nearest to `(lat, lon)` on `G`."""
    return int(ox.distance.nearest_nodes(G, X=lon, Y=lat))
