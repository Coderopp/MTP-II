"""
solver_milp.py - Exact Multi-Depot VRPTW solver (PuLP / CBC).

Enforces ALL paper constraints:
  * Q = 2 per rider (read from instance["parameters"]["Q"])
  * K dynamically scaled to N and distributed across the available depots
  * Every rider k is bound to a single depot d(k): the flow conservation
    constraints guarantee rider k enters and exits d(k) exactly once, and
    cannot use any other depot as origin or terminus.
  * Soft Time Windows: big-M time-propagation constraints, early arrivals
    wait (tau_i >= e_i), late arrivals pay a linear lateness penalty.
  * Multi-objective: L1 * t_ij + L2 * r_ij + L3 * (lateness + congestion).

The decision variable is x[i, j, k] in {0, 1}: rider k traverses arc (i, j).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np
import pulp


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def solve_md_vrptw(
    instance: dict,
    T_mat: np.ndarray,
    R_mat: np.ndarray,
    C_mat: np.ndarray | None = None,
    time_limit: int = 60,
    verbose: bool = False,
) -> dict:
    """
    Solve a MD-VRPTW instance exactly.

    Parameters
    ----------
    instance : dict produced by 04_instance_generator.py or generate_mock_instance
    T_mat    : (V, V) travel-time matrix in minutes (symmetric is fine)
    R_mat    : (V, V) risk matrix (e.g. -log survival exposure)
    C_mat    : (V, V) congestion / slack cost matrix; optional (defaults to 0)
    time_limit : CBC wall-clock budget (seconds)

    Node ordering convention
    ------------------------
    Node indices [0 .. D-1]        : depots d_0, ..., d_{D-1}
    Node indices [D .. D+N-1]      : customers c_0, ..., c_{N-1}
    T_mat, R_mat, C_mat must be built against this ordering.
    """
    depots = instance["depots"]
    customers = instance["customers"]
    params = instance["parameters"]

    D = len(depots)
    N = len(customers)
    V = D + N
    Q = int(params["Q"])                    # hard Q=2
    K = int(params["K"])                    # dynamic fleet
    L1 = float(params["lambda1"])
    L2 = float(params["lambda2"])
    L3 = float(params["lambda3"])
    tw_len = int(params.get("tw_window", 15))

    assert T_mat.shape == (V, V), f"T_mat shape {T_mat.shape} != ({V},{V})"
    assert R_mat.shape == (V, V)
    if C_mat is None:
        C_mat = np.zeros((V, V))

    # --- Rider -> depot mapping (must partition {0..K-1}) -------------------
    depot_riders: Dict[int, List[int]] = {}   # depot_idx -> list of rider ids
    rider_depot: Dict[int, int] = {}          # rider_id  -> depot_idx
    for d_idx, depot in enumerate(depots):
        ids = depot.get("riders", [])
        depot_riders[d_idx] = list(ids)
        for k in ids:
            rider_depot[k] = d_idx
    assert set(rider_depot) == set(range(K)), (
        f"depot_riders partition must cover all K={K} rider ids; got {sorted(rider_depot)}"
    )

    # --- Demand per node (depots have q=0) ---------------------------------
    q = np.zeros(V, dtype=float)
    e = np.zeros(V, dtype=float)
    l = np.zeros(V, dtype=float)
    for d_idx, depot in enumerate(depots):
        q[d_idx] = 0.0
        e[d_idx] = float(depot.get("e_0", 0))
        l[d_idx] = float(depot.get("l_0", 120))
    for c_idx, cust in enumerate(customers):
        v = D + c_idx
        q[v] = float(cust["q_i"])
        e[v] = float(cust["e_i"])
        l[v] = float(cust["l_i"])

    customer_nodes = list(range(D, V))
    depot_nodes = list(range(D))

    # ------------------------------------------------------------------ MILP
    prob = pulp.LpProblem("SA-MD-VRPTW", pulp.LpMinimize)

    # Arc variable: rider k uses arc (i, j). Forbid self-loops and
    # depot-to-depot moves (a rider stays on its depot).
    def arc_allowed(i: int, j: int, k: int) -> bool:
        if i == j:
            return False
        if i in depot_nodes and j in depot_nodes:
            return False
        # Depot-binding: rider k only enters/leaves its own depot.
        dk = rider_depot[k]
        if i in depot_nodes and i != dk:
            return False
        if j in depot_nodes and j != dk:
            return False
        return True

    x = {
        (i, j, k): pulp.LpVariable(f"x_{i}_{j}_{k}", cat="Binary")
        for i in range(V) for j in range(V) for k in range(K)
        if arc_allowed(i, j, k)
    }

    # Arrival time at node i by rider k (>= e_i so early arrivals simply wait)
    tau = {
        (i, k): pulp.LpVariable(f"tau_{i}_{k}", lowBound=float(e[i]), upBound=float(l[i]) + 10 * tw_len)
        for i in range(V) for k in range(K)
    }

    # Lateness (soft): nonnegative slack over l_i.
    late = {
        (i, k): pulp.LpVariable(f"late_{i}_{k}", lowBound=0)
        for i in customer_nodes for k in range(K)
    }

    # Cumulative load on rider k after visiting i (MTZ-style, 0..Q).
    u = {
        (i, k): pulp.LpVariable(f"u_{i}_{k}", lowBound=0, upBound=Q)
        for i in range(V) for k in range(K)
    }

    big_M = float(max(l.max(), 1.0) + T_mat.max() + tw_len * 10)

    # --- Objective ----------------------------------------------------------
    travel = pulp.lpSum(T_mat[i, j] * x[i, j, k] for (i, j, k) in x)
    risk   = pulp.lpSum(R_mat[i, j] * x[i, j, k] for (i, j, k) in x)
    cong   = pulp.lpSum(C_mat[i, j] * x[i, j, k] for (i, j, k) in x)
    lateness = pulp.lpSum(late[i, k] for i in customer_nodes for k in range(K))
    prob += L1 * travel + L2 * risk + L3 * (cong + lateness)

    # --- Each customer visited exactly once, by exactly one rider ----------
    for j in customer_nodes:
        prob += pulp.lpSum(
            x[i, j, k] for i in range(V) for k in range(K) if (i, j, k) in x
        ) == 1, f"visit_once_{j}"

    # --- Depot-binding & route cardinality ---------------------------------
    # Each rider k leaves its assigned depot d(k) at most once, and returns
    # to d(k) the same number of times (0 if unused, 1 if used).
    for k in range(K):
        dk = rider_depot[k]
        out_dk = pulp.lpSum(x[dk, j, k] for j in customer_nodes if (dk, j, k) in x)
        in_dk  = pulp.lpSum(x[i, dk, k] for i in customer_nodes if (i, dk, k) in x)
        prob += out_dk <= 1, f"depot_out_{k}"
        prob += in_dk == out_dk, f"depot_balance_{k}"
        # Belt-and-braces: rider k cannot visit any other depot (already
        # excluded by arc_allowed, but we restate for clarity).
        for d_other in depot_nodes:
            if d_other == dk:
                continue
            prob += pulp.lpSum(
                x[i, d_other, k] for i in range(V) if (i, d_other, k) in x
            ) == 0, f"no_other_depot_in_{k}_{d_other}"

    # --- Flow conservation at every customer -------------------------------
    for j in customer_nodes:
        for k in range(K):
            inflow  = pulp.lpSum(x[i, j, k] for i in range(V) if (i, j, k) in x)
            outflow = pulp.lpSum(x[j, i, k] for i in range(V) if (j, i, k) in x)
            prob += inflow == outflow, f"flow_{j}_{k}"

    # --- Capacity via MTZ (u_j >= u_i + q_j - Q(1 - x_ijk)) -----------------
    for k in range(K):
        dk = rider_depot[k]
        prob += u[dk, k] == 0, f"u_depot_{k}"
    for (i, j, k), var in x.items():
        if j in depot_nodes:
            continue                         # load only tracked on customers
        prob += (
            u[j, k] >= u[i, k] + q[j] - Q * (1 - var)
        ), f"mtz_{i}_{j}_{k}"
    for k in range(K):
        for j in customer_nodes:
            prob += u[j, k] <= Q, f"cap_upper_{j}_{k}"
            prob += u[j, k] >= q[j] * pulp.lpSum(
                x[i, j, k] for i in range(V) if (i, j, k) in x
            ), f"cap_lower_{j}_{k}"

    # --- Time propagation with big-M; soft lateness -------------------------
    for (i, j, k), var in x.items():
        if j in depot_nodes:
            continue  # returning to depot has no SLA
        prob += (
            tau[j, k] >= tau[i, k] + T_mat[i, j] - big_M * (1 - var)
        ), f"time_{i}_{j}_{k}"

    for j in customer_nodes:
        for k in range(K):
            prob += tau[j, k] >= e[j], f"tw_early_{j}_{k}"
            # Lateness: late_jk >= tau_jk - l_j  (active only if customer is
            # visited; otherwise tau_jk is free and late defaults to 0).
            prob += late[j, k] >= tau[j, k] - l[j], f"tw_late_{j}_{k}"

    # --- Solve --------------------------------------------------------------
    t0 = time.time()
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=1 if verbose else 0)
    prob.solve(solver)
    runtime = time.time() - t0
    status = pulp.LpStatus[prob.status]

    if status not in ("Optimal", "Not Solved"):
        return {"status": status, "runtime": runtime, "objective": math.nan,
                "routes": {}, "travel_time": math.nan, "risk": math.nan,
                "lateness": math.nan}

    # --- Extract routes -----------------------------------------------------
    routes: Dict[int, List[int]] = {}
    for k in range(K):
        dk = rider_depot[k]
        # Walk the route by following x=1 arcs starting from depot d(k).
        cur, route = dk, [dk]
        visited_local = set()
        while True:
            nxt = None
            for j in range(V):
                if (cur, j, k) in x and pulp.value(x[cur, j, k]) > 0.5:
                    nxt = j
                    break
            if nxt is None or nxt in visited_local:
                break
            route.append(nxt)
            visited_local.add(nxt)
            if nxt in depot_nodes:
                break
            cur = nxt
        routes[k] = route

    pure_t = sum(T_mat[i, j] for r in routes.values() for i, j in zip(r, r[1:]))
    pure_r = sum(R_mat[i, j] for r in routes.values() for i, j in zip(r, r[1:]))
    pure_late = sum(
        max(0.0, (pulp.value(tau[i, k]) or 0.0) - l[i])
        for k in range(K) for i in customer_nodes
        if (i, k) in tau and (pulp.value(tau[i, k]) or 0.0) > 0
    )

    return {
        "status":      status,
        "runtime":     runtime,
        "objective":   pulp.value(prob.objective),
        "routes":      routes,
        "travel_time": pure_t,
        "risk":        pure_r,
        "lateness":    pure_late,
        "K":           K,
        "Q":           Q,
    }


if __name__ == "__main__":
    # Smoke test with a tiny synthetic instance.
    from pprint import pprint
    import random as _rnd
    _rnd.seed(0)
    depots_mock = [
        {"name": "d0", "node_id": "0", "e_0": 0, "l_0": 120, "riders": [0, 1]},
        {"name": "d1", "node_id": "1", "e_0": 0, "l_0": 120, "riders": [2]},
    ]
    custs = [
        {"node_id": str(2 + i), "assigned_depot": "0" if i < 3 else "1",
         "e_i": i * 2, "l_i": i * 2 + 15, "q_i": _rnd.choice([1, 2])}
        for i in range(5)
    ]
    inst = {
        "depots": depots_mock, "customers": custs,
        "depot_riders": {"0": [0, 1], "1": [2]},
        "parameters": {"Q": 2, "K": 3, "N": 5,
                       "lambda1": 0.4, "lambda2": 0.4, "lambda3": 0.2,
                       "tw_window": 15},
    }
    V = 2 + 5
    rng = np.random.default_rng(0)
    T = rng.uniform(1, 10, (V, V)); np.fill_diagonal(T, 0)
    R = rng.uniform(0.01, 0.5, (V, V)); np.fill_diagonal(R, 0)
    res = solve_md_vrptw(inst, T, R, time_limit=10)
    pprint(res)
