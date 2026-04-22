"""
solver_metaheuristics.py - GA and ALNS for the Multi-Depot SA-VRPTW.

Representation
--------------
A solution is a dict {rider_id: [depot, c1, c2, ..., depot]}. Every route:
  * starts and ends at the rider's assigned depot (multi-depot respect);
  * carries at most Q = 2 demand units (sum(q_i) <= Q along the route);
  * early arrivals wait (no penalty), late arrivals incur a soft penalty.

Fitness
-------
  f = L1 * sum(t_ij) + L2 * sum(r_ij) + L3 * (sum(c_ij) + lateness_sum)
    + UNSERVED_PENALTY * |customers_not_served|

where (L1, L2, L3) = (0.4, 0.4, 0.2) by default.
"""

from __future__ import annotations

import copy
import math
import random
import time
from typing import Dict, List, Tuple

import numpy as np


UNSERVED_PENALTY = 1e4


# ---------------------------------------------------------------------------
# Problem container
# ---------------------------------------------------------------------------
class MDVRPTW:
    """Lightweight view of an instance + matrices. Node layout: depots, then customers."""

    def __init__(self, instance: dict, T_mat: np.ndarray, R_mat: np.ndarray,
                 C_mat: np.ndarray | None = None):
        self.depots = instance["depots"]
        self.customers = instance["customers"]
        params = instance["parameters"]
        self.Q = int(params["Q"])
        self.K = int(params["K"])
        self.L1 = float(params["lambda1"])
        self.L2 = float(params["lambda2"])
        self.L3 = float(params["lambda3"])

        self.D = len(self.depots)
        self.N = len(self.customers)
        self.V = self.D + self.N
        self.T = T_mat
        self.R = R_mat
        self.C = C_mat if C_mat is not None else np.zeros_like(T_mat)

        # q, e, l arrays aligned to matrix indexing
        self.q = np.zeros(self.V); self.e = np.zeros(self.V); self.l = np.zeros(self.V)
        for d_idx, d in enumerate(self.depots):
            self.e[d_idx] = float(d.get("e_0", 0))
            self.l[d_idx] = float(d.get("l_0", 120))
        for c_idx, c in enumerate(self.customers):
            v = self.D + c_idx
            self.q[v] = float(c["q_i"])
            self.e[v] = float(c["e_i"])
            self.l[v] = float(c["l_i"])

        # Rider -> depot index mapping
        self.rider_depot: Dict[int, int] = {}
        for d_idx, d in enumerate(self.depots):
            for rk in d.get("riders", []):
                self.rider_depot[rk] = d_idx
        assert set(self.rider_depot) == set(range(self.K)), \
            "Instance must provide a depot mapping for every rider 0..K-1."

        # Customer -> native assigned depot index (by node_id lookup)
        self.customer_depot: Dict[int, int] = {}
        depot_id_to_idx = {d["node_id"]: i for i, d in enumerate(self.depots)}
        for c_idx, c in enumerate(self.customers):
            self.customer_depot[self.D + c_idx] = depot_id_to_idx[c["assigned_depot"]]

    # --- Route cost & feasibility ------------------------------------------
    def route_cost(self, route: List[int]) -> Tuple[float, float, float, float]:
        """Return (weighted_obj_contrib, travel_time, risk, lateness) for a route."""
        if len(route) < 2:
            return 0.0, 0.0, 0.0, 0.0
        total_t = total_r = total_c = 0.0
        clock = float(self.e[route[0]])       # start at depot's opening time
        lateness = 0.0
        for i, j in zip(route, route[1:]):
            dt = float(self.T[i, j])
            total_t += dt
            total_r += float(self.R[i, j])
            total_c += float(self.C[i, j])
            clock += dt
            if j < self.D:                    # arrival at depot - no SLA
                continue
            if clock < self.e[j]:             # early -> wait
                clock = self.e[j]
            if clock > self.l[j]:             # late -> soft penalty
                lateness += clock - self.l[j]
        contrib = self.L1 * total_t + self.L2 * total_r + self.L3 * (total_c + lateness)
        return contrib, total_t, total_r, lateness

    def feasible_capacity(self, route: List[int]) -> bool:
        """Sum of demands along the route must not exceed Q."""
        load = sum(self.q[v] for v in route if v >= self.D)
        return load <= self.Q + 1e-9

    def evaluate(self, solution: Dict[int, List[int]]) -> dict:
        """Full-solution evaluation: objective + per-route stats + unserved penalty."""
        obj = tt = rr = lat = 0.0
        served = set()
        for k, route in solution.items():
            dk = self.rider_depot[k]
            # Force depot framing; if broken, penalize heavily.
            if not route or route[0] != dk or route[-1] != dk:
                obj += UNSERVED_PENALTY
                continue
            if not self.feasible_capacity(route):
                obj += UNSERVED_PENALTY
                continue
            c, t, r, l = self.route_cost(route)
            obj += c; tt += t; rr += r; lat += l
            for v in route[1:-1]:
                served.add(v)
        unserved = self.N - len(served)
        obj += UNSERVED_PENALTY * unserved
        return {"objective": obj, "travel_time": tt, "risk": rr,
                "lateness": lat, "unserved": unserved}


# ---------------------------------------------------------------------------
# Initial solution - depot-respecting, capacity-respecting greedy split
# ---------------------------------------------------------------------------
def greedy_initial(p: MDVRPTW, rng: random.Random) -> Dict[int, List[int]]:
    """
    Assign each customer to a rider at its native depot, chaining up to Q demand
    units per trip. Customers processed in order of earliest time window.
    """
    # Group customers by native depot, sort by e_i so time windows cascade
    by_depot: Dict[int, List[int]] = {d: [] for d in range(p.D)}
    for v, d_idx in p.customer_depot.items():
        by_depot[d_idx].append(v)
    for d_idx in by_depot:
        by_depot[d_idx].sort(key=lambda v: (p.e[v], -p.q[v]))

    riders_by_depot: Dict[int, List[int]] = {d: [] for d in range(p.D)}
    for k, d_idx in p.rider_depot.items():
        riders_by_depot[d_idx].append(k)

    solution = {k: [p.rider_depot[k], p.rider_depot[k]] for k in range(p.K)}
    # For each depot, fill its riders one at a time respecting Q.
    for d_idx, custs in by_depot.items():
        rider_ring = riders_by_depot[d_idx]
        if not rider_ring:
            continue
        slot = 0
        load = {k: 0.0 for k in rider_ring}
        for v in custs:
            placed = False
            # try riders in round-robin; prefer capacity-fit slot.
            for _ in range(len(rider_ring)):
                k = rider_ring[slot]
                slot = (slot + 1) % len(rider_ring)
                if load[k] + p.q[v] <= p.Q + 1e-9:
                    # insert before final depot
                    solution[k].insert(-1, v)
                    load[k] += p.q[v]
                    placed = True
                    break
            if not placed:
                # All riders at this depot full - drop (will be picked up by
                # local search / ALNS insertion) by leaving v unserved.
                pass
    return solution


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------
def solve_ga(
    instance: dict,
    T_mat: np.ndarray,
    R_mat: np.ndarray,
    C_mat: np.ndarray | None = None,
    pop_size: int = 40,
    generations: int = 200,
    mutation_rate: float = 0.15,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    t0 = time.time()
    rng = random.Random(seed)
    p = MDVRPTW(instance, T_mat, R_mat, C_mat)

    # --- Initial population: greedy base + perturbations -------------------
    population: List[Dict[int, List[int]]] = []
    base = greedy_initial(p, rng)
    population.append(base)
    for _ in range(pop_size - 1):
        cand = _perturb(base, p, rng)
        population.append(cand)

    def fitness(sol): return p.evaluate(sol)["objective"]

    best = min(population, key=fitness)
    best_fit = fitness(best)
    convergence = [best_fit]

    for gen in range(generations):
        population.sort(key=fitness)
        # elitism: top 20%
        elite_n = max(2, pop_size // 5)
        next_gen = [copy.deepcopy(s) for s in population[:elite_n]]

        while len(next_gen) < pop_size:
            p1 = _tournament(population, rng, fitness, k=3)
            p2 = _tournament(population, rng, fitness, k=3)
            child = _crossover(p1, p2, p, rng)
            if rng.random() < mutation_rate:
                child = _mutate(child, p, rng)
            next_gen.append(child)

        population = next_gen
        cur_best = min(population, key=fitness)
        cur_fit = fitness(cur_best)
        if cur_fit < best_fit:
            best, best_fit = copy.deepcopy(cur_best), cur_fit
        convergence.append(best_fit)
        if verbose and gen % 20 == 0:
            print(f"  [GA gen {gen}] best={best_fit:.3f}")

    stats = p.evaluate(best)
    return {
        "algorithm": "GA",
        "runtime": time.time() - t0,
        "objective": stats["objective"],
        "travel_time": stats["travel_time"],
        "risk": stats["risk"],
        "lateness": stats["lateness"],
        "unserved": stats["unserved"],
        "routes": best,
        "convergence": convergence,
        "K": p.K, "Q": p.Q,
    }


def _tournament(pop, rng, fitness, k=3):
    return min(rng.sample(pop, min(k, len(pop))), key=fitness)


def _crossover(a: Dict[int, List[int]], b: Dict[int, List[int]],
               p: MDVRPTW, rng: random.Random) -> Dict[int, List[int]]:
    """
    Route-level crossover: pick a random subset of riders from parent a to
    inherit their routes wholesale; refill the remaining customers using b's
    ordering while obeying Q=2 and depot affinity.
    """
    rider_ids = list(range(p.K))
    inherit_mask = {k: rng.random() < 0.5 for k in rider_ids}
    child = {k: [p.rider_depot[k], p.rider_depot[k]] for k in rider_ids}

    used = set()
    for k in rider_ids:
        if inherit_mask[k]:
            route = list(a[k])
            # Check capacity; if violated, trim
            load = 0.0
            trimmed = [route[0]]
            for v in route[1:-1]:
                if load + p.q[v] <= p.Q + 1e-9 and v not in used:
                    trimmed.append(v)
                    used.add(v)
                    load += p.q[v]
            trimmed.append(route[-1])
            child[k] = trimmed

    # Collect remaining customers in b's serving order
    leftover = []
    seen = set()
    for k in rider_ids:
        for v in b.get(k, []):
            if v >= p.D and v not in used and v not in seen:
                leftover.append(v); seen.add(v)
    for v in range(p.D, p.V):                # catch any missed by b
        if v not in used and v not in seen:
            leftover.append(v); seen.add(v)

    # Insert into child respecting depot + capacity
    for v in leftover:
        d_idx = p.customer_depot[v]
        candidates = [k for k in rider_ids if p.rider_depot[k] == d_idx]
        rng.shuffle(candidates)
        placed = False
        for k in candidates:
            load = sum(p.q[x] for x in child[k] if x >= p.D)
            if load + p.q[v] <= p.Q + 1e-9:
                child[k].insert(-1, v)
                placed = True
                break
        # else dropped; handled via UNSERVED_PENALTY
    return child


def _mutate(sol: Dict[int, List[int]], p: MDVRPTW, rng: random.Random) -> Dict[int, List[int]]:
    """2-opt swap within a single route, or relocate a customer to another
    rider of the same depot (keeps depot + capacity invariants)."""
    sol = copy.deepcopy(sol)
    op = rng.choice(["2opt", "relocate"])
    riders_with_work = [k for k, r in sol.items() if len(r) >= 4]

    if op == "2opt" and riders_with_work:
        k = rng.choice(riders_with_work)
        r = sol[k]
        i, j = sorted(rng.sample(range(1, len(r) - 1), 2))
        r[i:j + 1] = r[i:j + 1][::-1]
    elif op == "relocate":
        donors = [k for k, r in sol.items() if len(r) > 2]
        if not donors:
            return sol
        k_from = rng.choice(donors)
        v = rng.choice(sol[k_from][1:-1])
        d_idx = p.rider_depot[k_from]
        receivers = [k for k in range(p.K)
                     if p.rider_depot[k] == d_idx and k != k_from]
        rng.shuffle(receivers)
        for k_to in receivers:
            load = sum(p.q[x] for x in sol[k_to] if x >= p.D)
            if load + p.q[v] <= p.Q + 1e-9:
                sol[k_from].remove(v)
                sol[k_to].insert(-1, v)
                break
    return sol


def _perturb(base: Dict[int, List[int]], p: MDVRPTW, rng: random.Random):
    cand = copy.deepcopy(base)
    for _ in range(rng.randint(1, 3)):
        cand = _mutate(cand, p, rng)
    return cand


# ---------------------------------------------------------------------------
# Adaptive Large Neighborhood Search (ALNS)
# ---------------------------------------------------------------------------
def solve_alns(
    instance: dict,
    T_mat: np.ndarray,
    R_mat: np.ndarray,
    C_mat: np.ndarray | None = None,
    iterations: int = 500,
    destroy_frac: float = 0.25,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    t0 = time.time()
    rng = random.Random(seed)
    p = MDVRPTW(instance, T_mat, R_mat, C_mat)

    current = greedy_initial(p, rng)
    best = copy.deepcopy(current)
    best_obj = p.evaluate(best)["objective"]
    convergence = [best_obj]

    for it in range(iterations):
        # --- Destroy: remove ~destroy_frac of customers (random removal) ----
        cand = copy.deepcopy(current)
        served = [(k, v) for k, r in cand.items() for v in r[1:-1]]
        n_remove = max(1, int(destroy_frac * len(served)))
        to_remove = rng.sample(served, min(n_remove, len(served)))
        removed_customers = []
        for k, v in to_remove:
            if v in cand[k]:
                cand[k].remove(v)
                removed_customers.append(v)

        # --- Repair: regret-based insertion at matching depot ---------------
        rng.shuffle(removed_customers)
        for v in removed_customers:
            d_idx = p.customer_depot[v]
            rider_pool = [k for k in range(p.K) if p.rider_depot[k] == d_idx]
            best_delta = math.inf
            best_slot = None
            for k in rider_pool:
                load = sum(p.q[x] for x in cand[k] if x >= p.D)
                if load + p.q[v] > p.Q + 1e-9:
                    continue
                route = cand[k]
                for pos in range(1, len(route)):
                    trial = route[:pos] + [v] + route[pos:]
                    c, *_ = p.route_cost(trial)
                    c0, *_ = p.route_cost(route)
                    delta = c - c0
                    if delta < best_delta:
                        best_delta = delta
                        best_slot = (k, pos)
            if best_slot is not None:
                k, pos = best_slot
                cand[k] = cand[k][:pos] + [v] + cand[k][pos:]

        # --- Accept: simulated-annealing style on objective ----------------
        cand_obj = p.evaluate(cand)["objective"]
        cur_obj  = p.evaluate(current)["objective"]
        temperature = max(1.0, 100 * (1 - it / iterations))
        if cand_obj < cur_obj or rng.random() < math.exp(-(cand_obj - cur_obj) / temperature):
            current = cand
        if cand_obj < best_obj:
            best, best_obj = copy.deepcopy(cand), cand_obj
        convergence.append(best_obj)
        if verbose and it % 50 == 0:
            print(f"  [ALNS it {it}] best={best_obj:.3f}")

    stats = p.evaluate(best)
    return {
        "algorithm": "ALNS",
        "runtime": time.time() - t0,
        "objective": stats["objective"],
        "travel_time": stats["travel_time"],
        "risk": stats["risk"],
        "lateness": stats["lateness"],
        "unserved": stats["unserved"],
        "routes": best,
        "convergence": convergence,
        "K": p.K, "Q": p.Q,
    }
