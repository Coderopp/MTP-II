"""Verify the Blinkit KML ingestion against the committed snapshot."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from savrptw.data.darkstores import (
    cluster_depots,
    filter_by_bbox,
    load_blinkit_stores,
    parse_blinkit_placemarks,
)

REPO = Path(__file__).resolve().parents[1]
KML = REPO / "data" / "raw" / "darkstoremap_in_2026-04-09.kml"
CONF = str(REPO / "conf")


@pytest.fixture(scope="module")
def all_blinkit():
    assert KML.exists(), f"KML snapshot missing at {KML}"
    return parse_blinkit_placemarks(KML)


def test_kml_parse_gets_expected_count(all_blinkit):
    # The audit established 461 Blinkit placemarks across all 4 folders.
    # Tolerate minor parser skips (malformed rows) but flag large drift.
    assert 440 <= len(all_blinkit) <= 475, (
        f"expected ~461 Blinkit placemarks, got {len(all_blinkit)}"
    )


@pytest.mark.parametrize(
    # Thresholds pinned to observed counts in the 2026-04-09 snapshot:
    # bengaluru=113, delhi=111, gurugram=38, mumbai=96, pune=39.
    "city,min_count",
    [
        ("bengaluru", 100),
        ("delhi", 100),
        ("gurugram", 30),
        ("mumbai", 80),
        ("pune", 30),
    ],
)
def test_city_has_enough_blinkit_depots(city, min_count):
    """Every primary experiment city has enough Blinkit stores to cluster."""
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"city={city}"])
    all_rec = parse_blinkit_placemarks(KML)
    # Use the same folder filter the loader applies.
    allow = set(cfg.city.get("kml_folders") or [])
    filtered = [r for r in all_rec if not allow or r.folder in allow]
    filtered = filter_by_bbox(filtered, cfg.city.bbox)
    assert len(filtered) >= min_count, (
        f"{city}: only {len(filtered)} Blinkit stores — cannot cluster reliably"
    )


def test_hyderabad_has_no_blinkit_in_snapshot():
    """DATA GAP: the 2026-04-09 KML has no Blinkit dedicated folder for
    Hyderabad.  This test is intentionally an assertion of zero — it is a
    regression tripwire. Hyderabad is excluded from the primary single-brand
    experiments unless a future reproducible snapshot closes the gap.
    """
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["city=hyderabad"])
    all_rec = parse_blinkit_placemarks(KML)
    # With no folder allow-list, only bbox filter applies.
    allow = set(cfg.city.get("kml_folders") or [])
    pool = all_rec if not allow else [r for r in all_rec if r.folder in allow]
    filtered = filter_by_bbox(pool, cfg.city.bbox)
    assert len(filtered) == 0, (
        "Hyderabad Blinkit coverage unexpectedly nonzero — "
        "update the experiment city list; data gap has closed"
    )


def test_experiment_default_excludes_hyderabad():
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        cfg = compose(config_name="config")
    assert list(cfg.experiment.cities) == [
        "bengaluru",
        "delhi",
        "gurugram",
        "mumbai",
        "pune",
    ]


@pytest.mark.parametrize("k", [3, 5, 8])
def test_clustering_emits_k_centroids_inside_bbox(k):
    """K-means must return exactly k centroids that land inside the bbox."""
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["city=bengaluru"])
    out = load_blinkit_stores(KML, cfg.city, n_clusters=k, seed=42)
    assert out["n_clusters"] == k
    assert len(out["depot_centroids"]) == k
    bbox = cfg.city.bbox
    for c in out["depot_centroids"]:
        assert bbox.lat_min <= c["lat"] <= bbox.lat_max
        assert bbox.lon_min <= c["lon"] <= bbox.lon_max


def test_clustering_is_deterministic():
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["city=bengaluru"])
    a = load_blinkit_stores(KML, cfg.city, n_clusters=5, seed=42)
    b = load_blinkit_stores(KML, cfg.city, n_clusters=5, seed=42)
    assert a["depot_centroids"] == b["depot_centroids"]


def test_clustering_rejects_undersized_request():
    tiny = [
        {"lat": 12.97, "lon": 77.59},
        {"lat": 12.98, "lon": 77.60},
    ]
    from savrptw.data.darkstores import StoreRecord

    records = [StoreRecord(lat=r["lat"], lon=r["lon"], store_name="x", raw_id="x", folder="f") for r in tiny]
    with pytest.raises(ValueError):
        cluster_depots(records, n_clusters=5)
