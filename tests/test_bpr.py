"""Verify BPR congestion behaviour in isolation (no OSM download needed)."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
from hydra import compose, initialize_config_dir

from savrptw.congestion.bpr import attach_congestion

REPO = Path(__file__).resolve().parents[1]
CONF = str(REPO / "conf")


def _toy_graph() -> nx.MultiDiGraph:
    """Five edges, one of each highway class, all 1 km / 40 kph free-flow."""
    G = nx.MultiDiGraph()
    classes = ["motorway", "primary", "secondary", "residential", "living_street"]
    for i, hw in enumerate(classes):
        u, v = i, i + 1
        G.add_node(u)
        G.add_node(v)
        G.add_edge(
            u,
            v,
            key=0,
            length=1000.0,
            speed_kph=40.0,
            t_ij_free=(1.0 / 40.0) * 60.0,   # = 1.5 min
            highway=hw,
            lanes=2,
            h_ij=1 if hw in ("residential", "living_street") else 0,
        )
    return G


def _cfg():
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        return compose(config_name="config")


def test_attach_congestion_populates_t_ij():
    G = _toy_graph()
    cfg = _cfg()
    attach_congestion(G, cfg.congestion)
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        assert "c_ij" in data and data["c_ij"] >= 0.0
        assert "t_ij" in data and data["t_ij"] >= data["t_ij_free"] - 1e-9


def test_congestion_is_higher_on_arterials_than_residential_at_peak():
    G = _toy_graph()
    cfg = _cfg()
    cfg.congestion.dispatch_hour = 19  # peak
    attach_congestion(G, cfg.congestion)
    edges_by_class = {
        data["highway"]: data for _u, _v, _k, data in G.edges(keys=True, data=True)
    }
    # Primary (0.90 VC peak) should inflate more than residential (0.30).
    assert edges_by_class["primary"]["c_ij"] > edges_by_class["residential"]["c_ij"]
    assert edges_by_class["motorway"]["c_ij"] > edges_by_class["living_street"]["c_ij"]


def test_offpeak_gives_lower_congestion_than_peak():
    G_peak = _toy_graph()
    G_off = _toy_graph()
    cfg = _cfg()
    cfg.congestion.dispatch_hour = 19  # peak
    attach_congestion(G_peak, cfg.congestion)
    cfg.congestion.dispatch_hour = 2   # deep off-peak
    attach_congestion(G_off, cfg.congestion)
    pp = next(iter(G_peak.edges(keys=True, data=True)))[3]
    oo = next(iter(G_off.edges(keys=True, data=True)))[3]
    assert pp["c_ij"] > oo["c_ij"]
    assert pp["t_ij"] > oo["t_ij"]
