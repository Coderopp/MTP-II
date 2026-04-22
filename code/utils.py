"""
utils.py — Shared utility helpers for the SA-VRPTW pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

ENV_PATH = REPO_ROOT / ".env"


def load_env() -> None:
    """Load .env from repo root into os.environ (no-op if not present)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH)
    except ImportError:
        pass  # python-dotenv optional


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def load_graph(path: str | Path) -> nx.MultiDiGraph:
    """Load a GraphML file and restore numeric edge/node attributes."""
    G = nx.read_graphml(str(path))
    # OSMnx saves floats as strings in GraphML — cast known attributes back
    numeric_attrs = ["length", "t_ij", "r_ij", "c_ij", "speed_kph", "travel_time", "maxspeed"]
    for u, v, data in G.edges(data=True):
        for attr in numeric_attrs:
            if attr in data:
                try:
                    data[attr] = float(data[attr])
                except (ValueError, TypeError):
                    data[attr] = 0.0
    return G


def save_graph(G: nx.MultiDiGraph, path: str | Path) -> None:
    """
    Save graph to GraphML, stripping attributes that GraphML cannot encode:
      - Shapely geometry objects (LineString, etc.)
      - Python lists → first element or joined string
      - Any other non-primitive types → str()
    """
    import copy
    G2 = copy.deepcopy(G)

    _PRIMITIVES = (bool, int, float, str)

    def _clean(data: dict) -> None:
        for key, val in list(data.items()):
            if isinstance(val, _PRIMITIVES):
                continue                      # already serialisable
            elif isinstance(val, list):
                # GraphML needs scalars — take first or join
                data[key] = val[0] if len(val) == 1 else "|".join(map(str, val))
            else:
                # Covers Shapely geometries, dicts, etc. — drop or stringify
                # We drop 'geometry' since it's not needed for the VRP solver
                if key == "geometry":
                    del data[key]
                else:
                    try:
                        data[key] = str(val)
                    except Exception:
                        del data[key]

    for u, v, k, data in G2.edges(keys=True, data=True):
        _clean(data)

    for node, data in G2.nodes(data=True):
        _clean(data)

    nx.write_graphml(G2, str(path))
    print(f"  Saved → {path}")



# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(series: pd.Series, clip_zero: bool = True) -> pd.Series:
    """Min-max normalize a pandas Series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    normalized = (series - mn) / (mx - mn)
    if clip_zero:
        normalized = normalized.clip(lower=0.0, upper=1.0)
    return normalized


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

# Approximate bounding box for Bengaluru urban area
BENGALURU_BBOX = {
    "lat_min": 12.8340,
    "lat_max": 13.1436,
    "lon_min": 77.4601,
    "lon_max": 77.7840,
}


def in_bbox(lat: float, lon: float, bbox: dict = BENGALURU_BBOX) -> bool:
    """Return True if (lat, lon) falls inside the given bounding box."""
    return (
        bbox["lat_min"] <= lat <= bbox["lat_max"]
        and bbox["lon_min"] <= lon <= bbox["lon_max"]
    )


def filter_bbox(df: pd.DataFrame, lat_col: str = "Latitude",
                lon_col: str = "Longitude",
                bbox: dict = BENGALURU_BBOX) -> pd.DataFrame:
    """Filter a DataFrame to rows within the bounding box."""
    mask = (
        df[lat_col].between(bbox["lat_min"], bbox["lat_max"])
        & df[lon_col].between(bbox["lon_min"], bbox["lon_max"])
    )
    print(f"  BBox filter: {mask.sum()} / {len(df)} records kept")
    return df[mask].copy()
