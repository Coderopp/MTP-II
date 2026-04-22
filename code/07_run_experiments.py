"""
07_run_experiments.py - Automated benchmark harness for the SA-VRPTW study.

Orchestrates:
  1. generate_mock_instance(N, seed) - a reproducible MD-VRPTW instance that
     * scales K = max(ceil(N / 1.5), 1),
     * distributes K riders across 3 dark-store depots as evenly as possible,
     * uses Q = 2 thermal-bag capacity,
     * applies the 15-minute SLA window on every customer,
     * assigns customers to their nearest depot,
     * returns travel-time (T), risk (R) and congestion (C) matrices in the
       depot-first node ordering expected by every solver.
  2. Runs MILP, GA, ALNS, and DRL over N in {10, 20, 50, 100}.
  3. Prints a table and dumps results to data/07_experiment_results.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from solver_milp import solve_md_vrptw
from solver_metaheuristics import solve_ga, solve_alns
from solver_drl import solve_drl


# ---------------------------------------------------------------------------
# Paper parameters
# ---------------------------------------------------------------------------
Q_CAP = 2                           # Q = 2 thermal-bag units per rider
TW_WINDOW = 15                      # 15-minute SLA
LAMBDA = (0.40, 0.40, 0.20)         # (travel, risk, congestion/STW)
N_DEPOTS = 3
DEPOT_COORDS = [
    {"name": "DarkStore_TechMarket"},
    {"name": "DarkStore_Prembazar"},
    {"name": "DarkStore_GoleBazaar"},
]
BENCH_SIZES = [10, 20, 50, 100]

RESULTS_CSV = Path(__file__).resolve().parent.parent / "data" / "07_experiment_results.csv"


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------
def fleet_size(n: int) -> int:
    """K = max(ceil(N / 1.5), 1)."""
    return max(math.ceil(n / 1.5), 1)


def distribute_riders(k_total: int, n_depots: int) -> List[List[int]]:
    """Split K rider IDs across n_depots as evenly as possible."""
    base, rem = divmod(k_total, n_depots)
    counts = [base + (1 if i < rem else 0) for i in range(n_depots)]
    out, cursor = [], 0
    for cnt in counts:
        out.append(list(range(cursor, cursor + cnt)))
        cursor += cnt
    return out


def generate_mock_instance(n: int, seed: int = 42) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a MD-VRPTW benchmark instance *without* needing the OSM pipeline.

    Returns (instance_dict, T_mat, R_mat, C_mat) with node layout
    [depots..., customers...] so solver_* modules can ingest it directly.

    Customers are uniformly sampled in a 10x10 km plane; depots are placed
    in a triangle around the centroid. Travel time T_ij = euclidean_km * 3
    minutes (assumes 20 km/h), risk R_ij sampled log-uniform in [0.05, 0.8],
    congestion C_ij = 0.5 * T_ij with multiplicative noise.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # ------------------------------------------------------------- depot setup
    depot_xy = np.array([[3.0, 5.0], [7.0, 3.0], [5.0, 8.0]])
    k_total = fleet_size(n)
    rider_groups = distribute_riders(k_total, N_DEPOTS)

    depots = []
    for d_idx in range(N_DEPOTS):
        depots.append({
            "name":     DEPOT_COORDS[d_idx]["name"],
            "node_id":  f"D{d_idx}",
            "x":        float(depot_xy[d_idx, 0]),
            "y":        float(depot_xy[d_idx, 1]),
            "e_0":      0,
            "l_0":      120,
            "riders":   rider_groups[d_idx],
            "num_riders": len(rider_groups[d_idx]),
        })

    # ----------------------------------------------------------- customer setup
    cust_xy = rng.uniform(0, 10, (n, 2))
    customers = []
    current_tick = 0
    for i in range(n):
        # nearest depot by Euclidean distance
        dists = np.linalg.norm(depot_xy - cust_xy[i], axis=1)
        d_idx = int(np.argmin(dists))
        e_i = current_tick
        l_i = e_i + TW_WINDOW
        q_i = py_rng.randint(1, Q_CAP)          # q_i in {1, 2}, always <= Q
        customers.append({
            "node_id":        f"C{i}",
            "assigned_depot": f"D{d_idx}",
            "x":              float(cust_xy[i, 0]),
            "y":              float(cust_xy[i, 1]),
            "e_i":            e_i,
            "l_i":            l_i,
            "q_i":            q_i,
        })
        current_tick += py_rng.randint(1, 3)

    # ----------------------------------------------------------- matrices
    coords = np.vstack([depot_xy, cust_xy])      # depots first, customers after
    V = coords.shape[0]
    T = np.zeros((V, V)); R = np.zeros((V, V)); C = np.zeros((V, V))
    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            d_km = float(np.linalg.norm(coords[i] - coords[j]))
            T[i, j] = d_km * 3.0                                 # 20 km/h
            # Risk log-uniform [0.05, 0.8], then aggregated along shortest-arc
            R[i, j] = float(np.exp(rng.uniform(math.log(0.05), math.log(0.8))))
            C[i, j] = 0.5 * T[i, j] * rng.uniform(0.7, 1.3)

    instance = {
        "metadata": {
            "description":      "Mock MD-VRPTW benchmark instance",
            "n_customers":      n,
            "k_riders":         k_total,
            "vehicle_capacity": Q_CAP,
            "lambda":           list(LAMBDA),
            "seed":             seed,
        },
        "depots":        depots,
        "customers":     customers,
        "depot_riders":  {f"D{i}": rider_groups[i] for i in range(N_DEPOTS)},
        "parameters": {
            "Q":       Q_CAP,
            "K":       k_total,
            "N":       n,
            "lambda1": LAMBDA[0],
            "lambda2": LAMBDA[1],
            "lambda3": LAMBDA[2],
            "tw_window": TW_WINDOW,
        },
    }
    return instance, T, R, C


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_once(n: int, seed: int, milp_time_limit: int, drl_epochs: int,
             skip_milp: bool, verbose: bool) -> List[dict]:
    print(f"\n=== N = {n} (seed={seed}) ===")
    inst, T, R, C = generate_mock_instance(n, seed)
    K = inst["parameters"]["K"]
    print(f"  K = {K}, Q = {inst['parameters']['Q']}, "
          f"depots = {len(inst['depots'])}, "
          f"rider split = {[d['num_riders'] for d in inst['depots']]}")

    rows: List[dict] = []

    if not skip_milp and n <= 25:       # MILP tractability threshold
        t0 = time.time()
        try:
            res = solve_md_vrptw(inst, T, R, C_mat=C, time_limit=milp_time_limit,
                                 verbose=verbose)
            rows.append({"N": n, "K": K, "Algorithm": "MILP", **_summarise(res)})
        except Exception as exc:
            print(f"  MILP failed: {exc}")
            rows.append({"N": n, "K": K, "Algorithm": "MILP",
                         "runtime": time.time() - t0, "objective": float("nan"),
                         "travel_time": float("nan"), "risk": float("nan"),
                         "lateness": float("nan")})

    # GA
    res_ga = solve_ga(inst, T, R, C_mat=C, pop_size=40,
                      generations=100 if n <= 50 else 60, seed=seed,
                      verbose=verbose)
    rows.append({"N": n, "K": K, "Algorithm": "GA", **_summarise(res_ga)})

    # ALNS
    res_alns = solve_alns(inst, T, R, C_mat=C,
                          iterations=300 if n <= 50 else 200,
                          seed=seed, verbose=verbose)
    rows.append({"N": n, "K": K, "Algorithm": "ALNS", **_summarise(res_alns)})

    # DRL
    res_drl = solve_drl(inst, T, R, C_mat=C,
                        epochs=drl_epochs if n <= 50 else max(50, drl_epochs // 2),
                        seed=seed, verbose=verbose)
    rows.append({"N": n, "K": K, "Algorithm": "DRL", **_summarise(res_drl)})

    for r in rows:
        print(f"  {r['Algorithm']:<6} obj={r['objective']:.3f}  "
              f"t={r['travel_time']:.2f}  r={r['risk']:.3f}  "
              f"late={r['lateness']:.2f}  runtime={r['runtime']:.2f}s")
    return rows


def _summarise(res: dict) -> dict:
    return {
        "runtime":     res.get("runtime", float("nan")),
        "objective":   res.get("objective", float("nan")),
        "travel_time": res.get("travel_time", float("nan")),
        "risk":        res.get("risk", float("nan")),
        "lateness":    res.get("lateness", float("nan")),
        "unserved":    res.get("unserved", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=BENCH_SIZES,
                    help="Customer-count sweep (default: 10 20 50 100).")
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--milp-time-limit", type=int, default=30)
    ap.add_argument("--drl-epochs", type=int, default=200)
    ap.add_argument("--skip-milp", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    all_rows: List[dict] = []
    for n in args.sizes:
        all_rows.extend(run_once(
            n, args.seed, args.milp_time_limit, args.drl_epochs,
            args.skip_milp, args.verbose,
        ))

    df = pd.DataFrame(all_rows)
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
