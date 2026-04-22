"""
02_irad_risk.py — Step 2: Compute collision risk score r_ij from accident data.

TWO MODES (auto-detected):
  A. REAL DATA  : Set IRAD_CSV_PATH to a real iRAD export → uses actual records.
  B. SYNTHETIC  : If IRAD_CSV_PATH is absent/empty → generates 150 synthetic
                  accident records within the Kharagpur bounding box.

Synthetic data is clearly flagged in the output JSON for transparency.

Outputs
-------
data/02_graph_with_risk.graphml
    Graph from step 01 with r_ij (normalised [0,1]) added to every edge.
data/02_irad_metadata.json
    Summary: data source, accident count, date range, severity breakdown.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, BENGALURU_BBOX, filter_bbox, load_graph, normalize, save_graph

import osmnx as ox

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_GRAPH = DATA_DIR / "01_osm_graph.graphml"
OUTPUT_GRAPH = DATA_DIR / "02_graph_with_risk.graphml"
METADATA_PATH = DATA_DIR / "02_irad_metadata.json"

# If you have a real iRAD CSV, set this path:
IRAD_CSV_PATH: Path | None = DATA_DIR / "irad_accidents.csv"

# Severity encoding: Fatal > Grievous > Minor > Damage Only
SEVERITY_WEIGHTS = {
    "fatal":         4,
    "grievous":      3,
    "minor":         2,
    "damage":        1,
    "damage only":   1,
}

SYNTHETIC_N = 150       # number of synthetic accidents when real data unavailable
YEARS_SPAN  = 5         # denominator for normalisation (accidents per year per km)
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_synthetic_accidents(n: int = SYNTHETIC_N, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic iRAD-like accident records within Bengaluru bounding box.
    Records are clustered around known high-traffic zones via a Gaussian mixture.
    """
    rng = np.random.default_rng(seed)
    bbox = BENGALURU_BBOX

    # Cluster centres for Bengaluru
    cluster_centres = [
        (12.9716, 77.5946),   # Majestic / City Center
        (12.9250, 77.6227),   # Koramangala
        (12.9784, 77.6408),   # Indiranagar
        (13.0285, 77.5895),   # Hebbal
    ]
    weights = [0.35, 0.25, 0.25, 0.15]
    n_per_cluster = rng.multinomial(n, weights)

    rows = []
    severities = ["fatal", "grievous", "minor", "damage only"]
    sev_probs   = [0.10,    0.20,       0.45,    0.25]

    for (lat_c, lon_c), k in zip(cluster_centres, n_per_cluster):
        lats = rng.normal(lat_c, 0.008, k)
        lons = rng.normal(lon_c, 0.008, k)
        sevs = rng.choice(severities, size=k, p=sev_probs)
        years = rng.integers(2018, 2024, k)
        months = rng.integers(1, 13, k)
        for lat, lon, sev, yr, mo in zip(lats, lons, sevs, years, months):
            rows.append({
                "Latitude":  round(float(lat), 6),
                "Longitude": round(float(lon), 6),
                "Severity":  sev,
                "Year":      int(yr),
                "Month":     int(mo),
            })

    df = pd.DataFrame(rows)
    # Clip to bounding box
    df = filter_bbox(df, bbox=bbox)
    print(f"  [Synthetic] Generated {len(df)} accident records")
    return df


# ---------------------------------------------------------------------------
# Real iRAD loader
# ---------------------------------------------------------------------------

def load_irad_real(csv_path: Path) -> pd.DataFrame:
    """
    Load and standardise a real iRAD CSV export.

    Expected columns (MoRTH standard schema):
        Latitude, Longitude, Severity (or Accident_Severity), Year, Month

    The function is tolerant of alternate column names.
    """
    df = pd.read_csv(csv_path)
    print(f"  [Real iRAD] Loaded {len(df):,} raw records from {csv_path.name}")

    # Normalise column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    rename_map = {
        "Accident_Severity": "Severity",
        "accident_severity": "Severity",
        "severity":          "Severity",
        "latitude":          "Latitude",
        "longitude":         "Longitude",
        "year":              "Year",
        "month":             "Month",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"Latitude", "Longitude", "Severity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"iRAD CSV missing required columns: {missing}")

    df = df.dropna(subset=["Latitude", "Longitude", "Severity"])
    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = filter_bbox(df, bbox=BENGALURU_BBOX)
    print(f"  [Real iRAD] {len(df):,} records after bbox filter")
    return df


# ---------------------------------------------------------------------------
# Risk computation
# ---------------------------------------------------------------------------

def encode_severity(series: pd.Series) -> pd.Series:
    """Map severity string → numeric weight."""
    return series.str.lower().str.strip().map(SEVERITY_WEIGHTS).fillna(1)


def compute_risk(G, accidents: pd.DataFrame, data_source: str) -> dict:
    """
    Snap each accident to its nearest OSM edge and accumulate weighted counts.
    Returns metadata dict.
    """
    accidents["sev_weight"] = encode_severity(accidents["Severity"])

    lats = accidents["Latitude"].values
    lons = accidents["Longitude"].values

    print(f"  Snapping {len(accidents):,} accidents to nearest OSM edges …")
    ne_result = ox.distance.nearest_edges(G, lons, lats, return_dist=True)

    # OSMnx 1.9.x returns a 2-tuple: (edges_array, dists_array)
    # where edges_array shape is (N,) with each element being (u, v, key)
    # Earlier versions return a 4-tuple: (u_arr, v_arr, key_arr, dist_arr)
    if len(ne_result) == 2:
        edges_arr, dists = ne_result
        edges_arr = np.asarray(edges_arr)
        dists     = np.asarray(dists)
        if edges_arr.ndim == 2:          # shape (N, 3)
            us = edges_arr[:, 0]
            vs = edges_arr[:, 1]
            ks = edges_arr[:, 2]
        else:                             # shape (N,) of tuples
            us = np.array([str(e[0]) for e in edges_arr])
            vs = np.array([str(e[1]) for e in edges_arr])
            ks = np.array([e[2] for e in edges_arr])
    elif len(ne_result) == 4:
        us, vs, ks, dists = ne_result
        us    = np.asarray(us)
        vs    = np.asarray(vs)
        ks    = np.asarray(ks)
        dists = np.asarray(dists)
    else:
        raise ValueError(f"Unexpected nearest_edges return length: {len(ne_result)}")


    accidents = accidents.copy()
    # Cast to string to match GraphML node ID format (GraphML always stores as str)
    accidents["edge_u"]    = [str(x) for x in us]
    accidents["edge_v"]    = [str(x) for x in vs]
    accidents["edge_key"]  = [int(x) for x in ks]    # edge key stays int (0,1,...)
    accidents["snap_dist"] = dists

    # Aggregate per edge
    edge_risk = {}
    for _, row in accidents.iterrows():
        key = (str(row["edge_u"]), str(row["edge_v"]), int(row["edge_key"]))
        edge_risk[key] = edge_risk.get(key, 0.0) + float(row["sev_weight"])

    # Normalise by road length (km) and years span
    edge_risk_norm = {}
    for (u, v, k), raw in edge_risk.items():
        try:
            length_km = max(G[u][v][k].get("length", 1.0) / 1000.0, 0.001)
        except KeyError:
            length_km = 0.1   # fallback: 100m if edge not found (type mismatch guard)
        edge_risk_norm[(u, v, k)] = raw / (YEARS_SPAN * length_km)

    # Build Series for global min-max normalisation
    all_keys   = list(edge_risk_norm.keys())
    print(f"  [DEBUG] Snapped to {len(all_keys)} unique edges.")
    all_values = pd.Series([edge_risk_norm[k] for k in all_keys])
    print(f"  [DEBUG] Min raw risk: {all_values.min()}, Max raw risk: {all_values.max()}")
    all_values = pd.Series([edge_risk_norm[k] for k in all_keys])
    
    # FIX: Use log-scaling before min-max normalization to prevent extreme outliers 
    # (like a single massive pileup) from squashing the variance of all other roads.
    log_values = np.log1p(all_values)
    all_normalised = normalize(log_values)

    # Write r_ij back to graph
    for i, (u, v, k_) in enumerate(all_keys):
        G[u][v][k_]["r_ij"] = round(float(all_normalised.iloc[i]), 6)

    n_at_risk = sum(1 for _, _, d in G.edges(data=True) if d.get("r_ij", 0.0) > 0)
    print(f"  r_ij updated on {n_at_risk:,} edges "
          f"(out of {G.number_of_edges():,} total)")

    # Metadata
    years = accidents["Year"].dropna() if "Year" in accidents.columns else pd.Series([])
    meta = {
        "data_source":       data_source,
        "accident_count":    len(accidents),
        "years_span":        YEARS_SPAN,
        "year_range":        [int(years.min()), int(years.max())] if len(years) else None,
        "severity_counts":   accidents["Severity"].value_counts().to_dict(),
        "edges_with_risk":   n_at_risk,
        "total_edges":       G.number_of_edges(),
    }
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[02] Loading graph from Step 1 …")
    G = load_graph(INPUT_GRAPH)

    # --- Choose data mode ---------------------------------------------------
    use_real = (
        IRAD_CSV_PATH is not None
        and IRAD_CSV_PATH.exists()
        and IRAD_CSV_PATH.stat().st_size > 0
    )

    if use_real:
        print(f"[02] MODE: REAL iRAD data  → {IRAD_CSV_PATH}")
        accidents = load_irad_real(IRAD_CSV_PATH)
        data_source = f"real_irad:{IRAD_CSV_PATH.name}"
    else:
        print("[02] MODE: SYNTHETIC accident data (iRAD not available)")
        accidents = generate_synthetic_accidents()
        data_source = "synthetic"

    # --- Compute r_ij -------------------------------------------------------
    meta = compute_risk(G, accidents, data_source)

    # --- Save ---------------------------------------------------------------
    save_graph(G, OUTPUT_GRAPH)

    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {METADATA_PATH}")
    print(f"[02] Done. r_ij added. Source = [{data_source}]\n")


if __name__ == "__main__":
    main()
