"""BPR congestion inflation.

FORMULATION.md §10.3.

For each edge:

    V/C = class_peak_vc * hour_multiplier(hour)
    ρ  = α (V/C)^β
    c_ij = ρ
    t_ij = t_ij_free · (1 + ρ)

Notes
-----
* The Highway Capacity Manual (HCM 2010) lane capacities and the
  time-of-day multipliers live in `conf/congestion/bpr.yaml` — not here.
  Anything hand-pickable is forced into config so reviewers can audit the
  constants at a glance.
* No paid-API calls.  An optional `here_validate` helper will be added later
  for one-city sanity checking.
"""

from __future__ import annotations

import networkx as nx
from omegaconf import DictConfig, OmegaConf


def _hour_multiplier(hour: int, hour_profile: DictConfig) -> float:
    peak = set(hour_profile["peak_hours"])
    return (
        float(hour_profile["peak_multiplier"])
        if int(hour) in peak
        else float(hour_profile["offpeak_multiplier"])
    )


def _vc_ratio(hw: str, lanes: int, hour_mult: float, cfg: DictConfig) -> float:
    """Estimate (V/C) for a street of the given class at `hour`."""
    # Baseline V/C at peak for the class.
    vc_peak_table: dict = OmegaConf.to_container(cfg["vc_class_peak"], resolve=True)  # type: ignore[assignment]
    base_vc_peak = float(vc_peak_table.get(hw, vc_peak_table.get("unclassified", 0.5)))
    # Capacity scales with lane count; V/C does not (lanes increase both V and C).
    _ = lanes  # retained for future directional/asymmetric scaling
    return base_vc_peak * hour_mult


def attach_congestion(G: nx.MultiDiGraph, cong_cfg: DictConfig) -> nx.MultiDiGraph:
    """Populate `c_ij` and `t_ij` (congestion-inflated) on every edge."""
    alpha = float(cong_cfg["alpha"])
    beta = int(cong_cfg["beta"])
    hour = int(cong_cfg["dispatch_hour"])
    hour_mult = _hour_multiplier(hour, cong_cfg["hour_profile"])

    for _u, _v, _k, data in G.edges(keys=True, data=True):
        hw = data.get("highway", "unclassified")
        lanes = int(data.get("lanes", 1))
        vc = _vc_ratio(hw, lanes, hour_mult, cong_cfg)
        rho = alpha * (vc ** beta)
        data["c_ij"] = float(rho)
        t_free = float(data.get("t_ij_free", 0.0))
        data["t_ij"] = t_free * (1.0 + rho)
    return G
