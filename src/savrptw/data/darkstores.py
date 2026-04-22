"""Dark-store ingestion — Blinkit only, from the committed KML snapshot.

Source
------
    data/raw/darkstoremap_in_2026-04-09.kml
    (snapshot of https://darkstoremap.in/dark_store.kml on 2026-04-09)

Pipeline
--------
1. Parse KML, extract every `<Placemark>` from Blinkit folders listed in the
   city config.  Each placemark yields one `(lat, lon, store_name, raw_id)`.
2. Filter to the city bounding box (defensive — tolerates folder drift).
3. K-means cluster to `n_clusters` centroids.
4. Return centroids as depot candidates; OSM node snapping is performed by
   `savrptw.graph.osm.snap_to_node` downstream.

No hand-picked coordinates.  Only real data ever crosses this boundary.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sklearn.cluster import KMeans

KML_NS = {"k": "http://www.opengis.net/kml/2.2"}

# The Blinkit-brand folder names present in the 2026-04-09 snapshot.
_BLINKIT_FOLDER_PREFIXES: tuple[str, ...] = (
    "Blinkit ",
    "Pune Blinkit",
)


@dataclass(frozen=True)
class StoreRecord:
    """A single dark-store record as parsed from the KML."""

    lat: float
    lon: float
    store_name: str
    raw_id: str
    folder: str


def _extract_desc_field(desc: str, key: str) -> str | None:
    """Pull `Key: value` from KML description CDATA (tolerant to <br> soup)."""
    m = re.search(rf"{re.escape(key)}\s*:\s*([^<\n\r]+)", desc, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def _folder_is_blinkit(folder_name: str) -> bool:
    low = folder_name.lower()
    return any(folder_name.startswith(p) for p in _BLINKIT_FOLDER_PREFIXES) or "blinkit" in low


def parse_blinkit_placemarks(kml_path: Path) -> list[StoreRecord]:
    """Parse the whole KML file and return every Blinkit store placemark.

    Placemark geometry is always `<Point>` in this snapshot; other geometries
    (if any) are ignored.
    """
    if not kml_path.exists():
        raise FileNotFoundError(
            f"KML snapshot missing: {kml_path}. Run tools/refresh_darkstores.py "
            "or restore the committed file."
        )

    tree = ET.parse(kml_path)
    document = tree.getroot().find("k:Document", KML_NS)
    if document is None:
        raise ValueError(f"{kml_path} is not a well-formed KML Document")

    out: list[StoreRecord] = []
    for folder in document.findall("k:Folder", KML_NS):
        fname = folder.findtext("k:name", default="", namespaces=KML_NS) or ""
        if not _folder_is_blinkit(fname):
            continue
        for pm in folder.findall("k:Placemark", KML_NS):
            coords_el = pm.find(".//k:Point/k:coordinates", KML_NS)
            if coords_el is None or not coords_el.text:
                continue
            # KML stores "lon,lat[,alt]" — note the ordering trap.
            parts = coords_el.text.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
            except ValueError:
                continue
            name = pm.findtext("k:name", default="", namespaces=KML_NS) or ""
            desc = pm.findtext("k:description", default="", namespaces=KML_NS) or ""
            raw_id = _extract_desc_field(desc, "merchant_id") or name
            out.append(
                StoreRecord(
                    lat=lat, lon=lon, store_name=name.strip(), raw_id=str(raw_id), folder=fname
                )
            )
    return out


def filter_by_bbox(records: list[StoreRecord], bbox: DictConfig | dict) -> list[StoreRecord]:
    """Keep only stores that fall inside `bbox`.

    `bbox` is a mapping with keys `lat_min`, `lat_max`, `lon_min`, `lon_max`.
    """
    lat_min = float(bbox["lat_min"])
    lat_max = float(bbox["lat_max"])
    lon_min = float(bbox["lon_min"])
    lon_max = float(bbox["lon_max"])
    return [
        r
        for r in records
        if lat_min <= r.lat <= lat_max and lon_min <= r.lon <= lon_max
    ]


def cluster_depots(
    records: list[StoreRecord], n_clusters: int, seed: int = 42
) -> list[tuple[float, float]]:
    """K-means on `(lat, lon)` space, return centroid `(lat, lon)` tuples.

    K-means in degree space is adequate for within-city clustering (<25 km):
    the isotropy error at Indian latitudes is well below OSM node spacing.
    If the city ever exceeds ~40 km diameter, switch to equirectangular-m
    projection here.
    """
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if len(records) < n_clusters:
        raise ValueError(
            f"only {len(records)} stores available — cannot cluster into {n_clusters} depots"
        )
    X = np.array([[r.lat, r.lon] for r in records], dtype=float)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(X)
    return [(float(c[0]), float(c[1])) for c in km.cluster_centers_]


def load_blinkit_stores(
    kml_path: Path, city_cfg: DictConfig, n_clusters: int, seed: int = 42
) -> dict:
    """End-to-end: parse → filter → cluster.

    Returns a structured dict with both the raw store list (for the
    cross-validation step) and the clustered depot centroids (for the
    instance generator).
    """
    all_records = parse_blinkit_placemarks(kml_path)

    # First, any explicit folder allow-list from the city config wins.
    allow = set(city_cfg.get("kml_folders", []) or [])
    if allow:
        records = [r for r in all_records if r.folder in allow]
    else:
        records = list(all_records)

    # Defensive bbox filter (catches any folder drift or cross-city contamination).
    records = filter_by_bbox(records, city_cfg["bbox"])

    if not records:
        raise RuntimeError(
            f"No Blinkit stores for city {city_cfg['slug']!r} after folder+bbox filter "
            f"(kml_folders={list(allow)}, bbox={dict(city_cfg['bbox'])})"
        )

    centroids = cluster_depots(records, n_clusters=n_clusters, seed=seed)
    return {
        "city": city_cfg["slug"],
        "n_stores_raw": len(records),
        "n_clusters": n_clusters,
        "stores": [
            {"lat": r.lat, "lon": r.lon, "name": r.store_name, "raw_id": r.raw_id}
            for r in records
        ],
        "depot_centroids": [{"lat": la, "lon": lo} for la, lo in centroids],
    }
