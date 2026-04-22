from __future__ import annotations

import networkx as nx
from omegaconf import OmegaConf

from savrptw.risk import attach_risk, expected_annual_events, relative_error


def _risk_cfg():
    return OmegaConf.create(
        {
            "source": "morth_mohan_osm_proxy_v1",
            "lambda_class": {
                "primary": 6.11e-08,
                "secondary": 8.65e-08,
                "residential": 2.61e-08,
                "unclassified": 9.41e-09,
            },
            "severity_multiplier": {
                "primary": 2.086,
                "secondary": 1.481,
                "residential": 0.324,
                "unclassified": 0.324,
            },
            "proxy": {
                "betweenness_k": 0,
                "max_betweenness_reference": 0.10,
                "max_signal_count": 6,
                "max_crossing_count": 6,
                "w_betweenness": 0.40,
                "w_signal_density": 0.30,
                "w_crossing_density": 0.30,
                "min_weight": 0.50,
                "max_weight": 2.00,
            },
            "cross_validation": {
                "annual_edge_traversals_reference": {
                    "primary": 1_000_000,
                    "secondary": 600_000,
                    "residential": 120_000,
                    "unclassified": 100_000,
                }
            },
            "r_max_clip": 0.99,
        }
    )


def _build_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node(0, x=77.0, y=12.0, highway="traffic_signals")
    G.add_node(1, x=77.001, y=12.001, crossing="marked")
    G.add_node(2, x=77.002, y=12.002)
    G.add_node(3, x=77.003, y=12.003)

    G.add_edge(0, 1, key=0, highway="primary", length=800.0, t_ij=2.0)
    G.add_edge(1, 2, key=0, highway="secondary", length=600.0, t_ij=1.8)
    G.add_edge(2, 3, key=0, highway="residential", length=500.0, t_ij=1.5)
    G.add_edge(3, 0, key=0, highway="residential", length=450.0, t_ij=1.4)
    return G


def test_r_ij_in_range():
    G = attach_risk(_build_graph(), _risk_cfg())
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        assert 0.0 <= data["r_ij"] <= 0.99


def test_arterial_riskier_than_residential():
    G = attach_risk(_build_graph(), _risk_cfg())
    arterial = G[0][1][0]["r_ij"]
    local = G[2][3][0]["r_ij"]
    assert arterial > local


def test_calibration_matches_city_total_on_synthetic_graph():
    G = attach_risk(_build_graph(), _risk_cfg())
    expected = expected_annual_events(G, _risk_cfg())
    target = expected * 1.10
    assert relative_error(expected, target) <= 0.25
