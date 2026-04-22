"""Bengaluru (and Indian-city) Accident Surrogate Model — BASM.

Task #16 closes the earlier placeholder by wiring a documented, auditable
calibration path:

    r_ij = clip(
        λ_class(highway) · len_km · severity_multiplier(highway) · proxy_edge,
        0,
        r_max_clip,
    )

`λ_class` and `severity_multiplier` come from `conf/risk/basm_v1.yaml`; the
derivation and citations live in `docs/BASM_CALIBRATION.md`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import networkx as nx
from omegaconf import DictConfig, OmegaConf

_ARTERIAL = frozenset({"motorway", "trunk", "primary"})
_COLLECTOR = frozenset({"secondary", "tertiary"})
_LOCAL = frozenset({"residential", "living_street", "service", "unclassified"})


def _canonical_highway(raw: object) -> str:
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    hw = str(raw or "unclassified")
    if hw.endswith("_link"):
        hw = hw[:-5]
    if hw in _ARTERIAL | _COLLECTOR | _LOCAL:
        return hw
    return "unclassified"


def _bounded(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _neighbor_tag_count(
    G: nx.MultiDiGraph,
    node_ids: Iterable[int],
    *,
    predicate,
) -> int:
    seen: set[int] = set()
    count = 0
    for nid in node_ids:
        if nid in seen or nid not in G:
            continue
        seen.add(nid)
        data = G.nodes[nid]
        if predicate(data):
            count += 1
    return count


def _edge_proxy_weight(
    G: nx.MultiDiGraph,
    u: int,
    v: int,
    edge_betweenness: dict[tuple[int, int], float],
    risk_cfg: DictConfig,
) -> float:
    """Return a bounded local proxy multiplier for edge `(u, v)`.

    The proxy is intentionally simple and deterministic:
    1. approximate edge betweenness on a simplified directed view,
    2. local traffic-signal density around the edge endpoints,
    3. local crossing density around the edge endpoints.

    The multiplier is clipped into the configured range to avoid overpowering
    the sourced class rates.
    """
    proxy_cfg = risk_cfg["proxy"]
    max_bet = float(proxy_cfg["max_betweenness_reference"])
    max_signals = float(proxy_cfg["max_signal_count"])
    max_crossings = float(proxy_cfg["max_crossing_count"])

    local_nodes = {u, v}
    local_nodes.update(G.predecessors(u))
    local_nodes.update(G.successors(u))
    local_nodes.update(G.predecessors(v))
    local_nodes.update(G.successors(v))

    signal_count = _neighbor_tag_count(
        G,
        local_nodes,
        predicate=lambda data: str(data.get("highway", "")) == "traffic_signals",
    )
    crossing_count = _neighbor_tag_count(
        G,
        local_nodes,
        predicate=lambda data: (
            str(data.get("highway", "")) == "crossing"
            or "crossing" in data
            or str(data.get("footway", "")) == "crossing"
        ),
    )

    bet = edge_betweenness.get((u, v), 0.0)
    bet_norm = _bounded(bet / max_bet, 0.0, 1.0)
    signal_norm = _bounded(signal_count / max_signals, 0.0, 1.0)
    crossing_norm = _bounded(crossing_count / max_crossings, 0.0, 1.0)

    proxy = (
        1.0
        + float(proxy_cfg["w_betweenness"]) * (2.0 * bet_norm - 1.0)
        + float(proxy_cfg["w_signal_density"]) * signal_norm
        + float(proxy_cfg["w_crossing_density"]) * crossing_norm
    )
    return _bounded(
        proxy,
        float(proxy_cfg["min_weight"]),
        float(proxy_cfg["max_weight"]),
    )


def _approx_edge_betweenness(
    G: nx.MultiDiGraph,
    risk_cfg: DictConfig,
) -> dict[tuple[int, int], float]:
    """Approximate edge betweenness on a min-time simple directed graph."""
    simple = nx.DiGraph()
    for u, v, _k, data in G.edges(keys=True, data=True):
        weight = float(data.get("t_ij", data.get("t_ij_free", 1.0)))
        if simple.has_edge(u, v) and simple[u][v]["weight"] <= weight:
            continue
        simple.add_edge(u, v, weight=weight)

    if simple.number_of_edges() == 0:
        return {}

    k = int(risk_cfg["proxy"]["betweenness_k"])
    if 0 < k < simple.number_of_nodes():
        raw = nx.edge_betweenness_centrality(simple, k=k, weight="weight", seed=17)
    else:
        raw = nx.edge_betweenness_centrality(simple, weight="weight")
    return {(int(u), int(v)): float(val) for (u, v), val in raw.items()}


def expected_annual_events(
    G: nx.MultiDiGraph,
    risk_cfg: DictConfig,
) -> float:
    """Aggregate expected annual crash events from calibrated edge probabilities.

    `annual_edge_traversals_reference` is a per-class exposure proxy: the
    representative number of traversals on a one-kilometre segment in one year.
    For a single edge traversal, `r_ij` already scales with length, so the
    annual expectation is simply `Σ r_ij × traversals(class(edge))`.
    """
    traversals = OmegaConf.to_container(
        risk_cfg["cross_validation"]["annual_edge_traversals_reference"],
        resolve=True,
    )
    total = 0.0
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        hw = _canonical_highway(data.get("highway"))
        total += float(data.get("r_ij", 0.0)) * float(
            traversals.get(hw, traversals.get("unclassified", 0.0))
        )
    return total


def relative_error(expected: float, target: float) -> float:
    if target <= 0.0:
        raise ValueError("target must be positive")
    return abs(expected - target) / target


def attach_risk(G: nx.MultiDiGraph, risk_cfg: DictConfig) -> nx.MultiDiGraph:
    """Attach `r_ij` to every edge of `G`, returning the same graph for chaining.

    Fails loudly if `source == "uncalibrated"` — prevents accidental production
    runs with placeholder rates.
    """
    if G is None or risk_cfg is None:
        raise RuntimeError("uncalibrated — BASM requires a graph and a populated config")

    if str(risk_cfg.get("source", "uncalibrated")) == "uncalibrated":
        raise RuntimeError("uncalibrated — pick a calibrated BASM source")

    lambda_class = OmegaConf.to_container(risk_cfg["lambda_class"], resolve=True)
    severity_multiplier = OmegaConf.to_container(
        risk_cfg["severity_multiplier"], resolve=True
    )
    edge_betweenness = _approx_edge_betweenness(G, risk_cfg)
    r_max = float(risk_cfg["r_max_clip"])

    for u, v, _k, data in G.edges(keys=True, data=True):
        hw = _canonical_highway(data.get("highway"))
        data["highway"] = hw
        length_km = max(float(data.get("length", 0.0)) / 1000.0, 0.0)
        base = float(lambda_class.get(hw, lambda_class.get("unclassified", 0.0)))
        sev = float(
            severity_multiplier.get(hw, severity_multiplier.get("unclassified", 1.0))
        )
        proxy = _edge_proxy_weight(G, u, v, edge_betweenness, risk_cfg)
        r_ij = _bounded(base * length_km * sev * proxy, 0.0, r_max)
        data["r_ij"] = r_ij
        data["risk_proxy_weight"] = proxy

    return G
