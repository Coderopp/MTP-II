"""Adaptive Large Neighbourhood Search (Ropke & Pisinger, 2006).

Representation matches the GA: a list of routes (each a sequence of customer
ids flanked by a depot).  An iteration destroys `q` customers via a randomly
selected destroy operator, reinserts them via a randomly selected repair
operator, then accepts/rejects via simulated annealing.  Operator weights
adapt with the Ropke-Pisinger sigma scheme.

Operators implemented
---------------------
Destroy
    * random_removal     — uniform random customer removal
    * worst_removal      — removes customers whose arc cost contributes most
    * shaw_removal       — related-ness via travel-time proximity
    * risk_cluster       — targets customers on arcs with high R_uv

Repair
    * greedy_insert      — cheapest feasible insertion per customer
    * regret_2_insert    — max-regret over second-best positions
    * regret_3_insert    — max-regret over third-best positions

The final incumbent is validated strictly; no fake feasibility hints.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Iterable

from savrptw.eval.feasibility import validate
from savrptw.eval.objective import objective as eval_objective
from savrptw.solvers._common import build_route
from savrptw.solvers.base import InfeasibleError, Solver, register
from savrptw.types import Customer, Depot, Instance, Route, Solution


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class ALNSConfig:
    iterations: int = 5000
    segment_length: int = 100
    reaction_factor: float = 0.1
    sigma: tuple[float, float, float, float] = (33.0, 9.0, 1.0, 0.0)
    destroy_ops: tuple[str, ...] = (
        "random_removal",
        "worst_removal",
        "shaw_removal",
        "risk_cluster_removal",
    )
    repair_ops: tuple[str, ...] = ("greedy_insert", "regret_2_insert", "regret_3_insert")
    T_start_ratio: float = 0.05
    cooling_rate: float = 0.99975
    seed: int = 42
    # Fraction of customers to destroy per iteration — uniform in [lo, hi).
    destroy_frac_lo: float = 0.15
    destroy_frac_hi: float = 0.45


def _cfg_from(cfg) -> ALNSConfig:
    get = (lambda k, d: cfg.get(k, d)) if hasattr(cfg, "get") else (lambda k, d: d)
    sigma = get("sigma", None)
    if sigma is not None:
        sigma = tuple(float(s) for s in sigma)
    else:
        sigma = (33.0, 9.0, 1.0, 0.0)
    acceptance = get("acceptance", None)
    if acceptance is not None and hasattr(acceptance, "get"):
        T_start_ratio = float(acceptance.get("T_start_ratio", 0.05))
        cooling_rate = float(acceptance.get("cooling_rate", 0.99975))
    else:
        T_start_ratio, cooling_rate = 0.05, 0.99975

    def _tup(k, fallback):
        v = get(k, fallback)
        if v is None:
            return fallback
        return tuple(str(x) for x in v)

    return ALNSConfig(
        iterations=int(get("iterations", 5000)),
        segment_length=int(get("segment_length", 100)),
        reaction_factor=float(get("reaction_factor", 0.1)),
        sigma=sigma,  # type: ignore[arg-type]
        destroy_ops=_tup(
            "destroy_operators",
            ("random_removal", "worst_removal", "shaw_removal", "risk_cluster_removal"),
        ),
        repair_ops=_tup(
            "repair_operators", ("greedy_insert", "regret_2_insert", "regret_3_insert")
        ),
        T_start_ratio=T_start_ratio,
        cooling_rate=cooling_rate,
        seed=int(get("seed", 42)),
    )


# -----------------------------------------------------------------------------
# Solution representation used internally — list of per-depot route customer
# sequences.  Depot bookends are implicit and attached when exporting.
# -----------------------------------------------------------------------------


@dataclass
class _State:
    """Mutable ALNS working state."""

    routes: list[list[Customer]] = field(default_factory=list)      # one list per active rider
    route_depots: list[int] = field(default_factory=list)           # parallel: depot_id per route
    unassigned: list[Customer] = field(default_factory=list)        # customers pulled out by destroy

    def copy(self) -> "_State":
        return _State(
            routes=[list(r) for r in self.routes],
            route_depots=list(self.route_depots),
            unassigned=list(self.unassigned),
        )


# -----------------------------------------------------------------------------
# Cost model — a single function all operators must use to score moves.
# -----------------------------------------------------------------------------


def _route_cost(instance: Instance, depot: Depot, custs: list[Customer]) -> tuple[float, bool]:
    """Return (F1 contribution + hard-violation penalty, feasible-flag)."""
    rb = build_route(instance, depot, custs)
    # Infeasibility gets a huge but finite penalty so the SA can still move.
    pen = 0.0 if rb.feasible else 1e6 + rb.violation_mass
    return rb.f1_contribution + pen, rb.feasible


def _state_cost(instance: Instance, state: _State, depot_by_id: dict[int, Depot]) -> float:
    total = 0.0
    for depot_id, custs in zip(state.route_depots, state.routes, strict=True):
        c, _ = _route_cost(instance, depot_by_id[depot_id], custs)
        total += c
    # Unassigned customers heavily penalised — repair must place them.
    total += 1e9 * len(state.unassigned)
    return total


# -----------------------------------------------------------------------------
# Initial construction — greedy sequential insertion per depot.
# -----------------------------------------------------------------------------


def _initial_solution(
    instance: Instance, depot_by_id: dict[int, Depot], rng: random.Random
) -> _State:
    state = _State()
    grouped: dict[int, list[Customer]] = {d.depot_id: [] for d in instance.depots}
    for c in instance.customers:
        grouped[c.home_depot].append(c)

    for depot_id, custs in grouped.items():
        rng.shuffle(custs)
        depot = depot_by_id[depot_id]
        current: list[Customer] = []
        for c in custs:
            trial = current + [c]
            _, feas = _route_cost(instance, depot, trial)
            if feas:
                current = trial
            else:
                if current:
                    state.routes.append(current)
                    state.route_depots.append(depot_id)
                # Start a new route with this customer; if even a singleton
                # is infeasible (e.g. depot→customer arc alone exceeds the
                # residential cap), keep it as a singleton anyway and let SA
                # try to repair — final validate() is the strict gate.
                current = [c]
        if current:
            state.routes.append(current)
            state.route_depots.append(depot_id)
    return state


# -----------------------------------------------------------------------------
# Destroy operators
# -----------------------------------------------------------------------------


def _pick_destroy_count(instance: Instance, cfg: ALNSConfig, rng: random.Random) -> int:
    frac = rng.uniform(cfg.destroy_frac_lo, cfg.destroy_frac_hi)
    q = max(1, int(round(frac * len(instance.customers))))
    return min(q, max(1, len(instance.customers) - 1))


def _iter_state_positions(
    state: _State,
) -> Iterable[tuple[int, int, Customer]]:
    for ri, route in enumerate(state.routes):
        for pi, c in enumerate(route):
            yield ri, pi, c


def _remove_positions(state: _State, positions: list[tuple[int, int]]) -> None:
    """Remove (route, pos) pairs, handling index shifts correctly."""
    # Group by route index.
    by_route: dict[int, list[int]] = {}
    for ri, pi in positions:
        by_route.setdefault(ri, []).append(pi)
    for ri, pis in by_route.items():
        for pi in sorted(pis, reverse=True):
            state.unassigned.append(state.routes[ri].pop(pi))


def destroy_random(instance: Instance, state: _State, q: int, rng: random.Random) -> None:
    candidates = [(ri, pi) for ri, pi, _ in _iter_state_positions(state)]
    if not candidates:
        return
    picks = rng.sample(candidates, min(q, len(candidates)))
    _remove_positions(state, picks)


def destroy_worst(instance: Instance, state: _State, q: int, rng: random.Random) -> None:
    """Remove customers whose local contribution to arrival-time + lateness is
    largest.  Approximation: per-customer (e_i-late-mass)."""
    scored: list[tuple[float, int, int]] = []
    depot_by_id = {d.depot_id: d for d in instance.depots}
    for ri, route in enumerate(state.routes):
        if not route:
            continue
        depot = depot_by_id[state.route_depots[ri]]
        rb = build_route(instance, depot, route)
        # Score = tau at each customer (zero if on-time) + wait mass.
        for pi, c in enumerate(route):
            # `rb.arrivals` includes depot-at-0 and depot-at-end; customer pi
            # is at index pi+1.
            a = rb.arrivals[pi + 1]
            tau = max(0.0, a - c.eta_i)
            wait = max(0.0, c.e_i - a)
            score = tau + 0.1 * wait + 0.01 * rng.random()  # jitter
            scored.append((-score, ri, pi))  # negative so largest first with heap-less sort
    scored.sort()
    picks = [(ri, pi) for _, ri, pi in scored[:q]]
    _remove_positions(state, picks)


def destroy_shaw(instance: Instance, state: _State, q: int, rng: random.Random) -> None:
    """Relatedness removal — pick a seed, then remove closest customers by T_uv."""
    flat = [(ri, pi, c) for ri, pi, c in _iter_state_positions(state)]
    if not flat:
        return
    seed_idx = rng.randrange(len(flat))
    _, _, seed = flat[seed_idx]

    def d(other: Customer) -> float:
        arc = instance.super_arcs.get((seed.customer_id, other.customer_id))
        return arc.T_uv if arc is not None else 1e9

    ranked = sorted(flat, key=lambda t: d(t[2]))
    picks = [(ri, pi) for ri, pi, _ in ranked[:q]]
    _remove_positions(state, picks)


def destroy_risk_cluster(
    instance: Instance, state: _State, q: int, rng: random.Random
) -> None:
    """Remove customers on arcs carrying the highest R_uv."""
    scored: list[tuple[float, int, int]] = []
    depot_by_id = {d.depot_id: d for d in instance.depots}
    for ri, route in enumerate(state.routes):
        if not route:
            continue
        depot = depot_by_id[state.route_depots[ri]]
        prev = depot.depot_id
        for pi, c in enumerate(route):
            arc = instance.super_arcs.get((prev, c.customer_id))
            score = float(arc.R_uv) if arc is not None else 0.0
            scored.append((-score + 0.01 * rng.random(), ri, pi))
            prev = c.customer_id
    scored.sort()
    picks = [(ri, pi) for _, ri, pi in scored[:q]]
    _remove_positions(state, picks)


_DESTROY = {
    "random_removal": destroy_random,
    "worst_removal": destroy_worst,
    "shaw_removal": destroy_shaw,
    "risk_cluster_removal": destroy_risk_cluster,
}


# -----------------------------------------------------------------------------
# Repair operators
# -----------------------------------------------------------------------------


def _best_insertions(
    instance: Instance,
    state: _State,
    cust: Customer,
    depot_by_id: dict[int, Depot],
) -> list[tuple[float, int, int]]:
    """Return list of `(delta_cost, route_idx, position)` sorted ascending.

    Only considers routes whose depot matches `cust.home_depot`.  Emits the
    option of creating a new single-customer route (route_idx = len(routes)).
    """
    options: list[tuple[float, int, int]] = []
    depot = depot_by_id[cust.home_depot]
    for ri, route in enumerate(state.routes):
        if state.route_depots[ri] != cust.home_depot:
            continue
        base, _ = _route_cost(instance, depot, route)
        for pos in range(len(route) + 1):
            trial = route[:pos] + [cust] + route[pos:]
            new_cost, feas = _route_cost(instance, depot, trial)
            if not feas:
                continue
            options.append((new_cost - base, ri, pos))
    # Option: open a new route.
    new_cost, feas_new = _route_cost(instance, depot, [cust])
    if feas_new:
        options.append((new_cost, len(state.routes), 0))
    options.sort(key=lambda t: t[0])
    return options


def _commit_insertion(state: _State, cust: Customer, ri: int, pos: int) -> None:
    if ri == len(state.routes):
        state.routes.append([cust])
        state.route_depots.append(cust.home_depot)
    else:
        state.routes[ri].insert(pos, cust)


def repair_greedy(
    instance: Instance, state: _State, rng: random.Random
) -> bool:
    depot_by_id = {d.depot_id: d for d in instance.depots}
    rng.shuffle(state.unassigned)
    remaining: list[Customer] = []
    while state.unassigned:
        c = state.unassigned.pop()
        opts = _best_insertions(instance, state, c, depot_by_id)
        if not opts:
            remaining.append(c)
            continue
        _, ri, pos = opts[0]
        _commit_insertion(state, c, ri, pos)
    state.unassigned = remaining
    return not state.unassigned


def _regret_repair(
    instance: Instance, state: _State, rng: random.Random, k: int
) -> bool:
    depot_by_id = {d.depot_id: d for d in instance.depots}
    while state.unassigned:
        best_c: Customer | None = None
        best_reg = -math.inf
        best_opt: tuple[float, int, int] | None = None
        for c in state.unassigned:
            opts = _best_insertions(instance, state, c, depot_by_id)
            if not opts:
                # This customer cannot be placed — treat as infinite regret
                # so we deal with it first.
                reg = math.inf
                opt: tuple[float, int, int] | None = None
            else:
                top = opts[0][0]
                kth = opts[min(k - 1, len(opts) - 1)][0]
                reg = kth - top
                opt = opts[0]
            if reg > best_reg:
                best_reg = reg
                best_c = c
                best_opt = opt
        assert best_c is not None
        if best_opt is None:
            # Nothing feasible for this customer — drop to unassigned end and stop
            # trying to repair; caller will fail.
            state.unassigned.remove(best_c)
            state.unassigned.insert(0, best_c)
            return False
        _commit_insertion(state, best_c, best_opt[1], best_opt[2])
        state.unassigned.remove(best_c)
    return True


def repair_regret_2(instance: Instance, state: _State, rng: random.Random) -> bool:
    return _regret_repair(instance, state, rng, 2)


def repair_regret_3(instance: Instance, state: _State, rng: random.Random) -> bool:
    return _regret_repair(instance, state, rng, 3)


_REPAIR = {
    "greedy_insert": repair_greedy,
    "regret_2_insert": repair_regret_2,
    "regret_3_insert": repair_regret_3,
}


# -----------------------------------------------------------------------------
# Roulette-wheel operator selection with adaptive weights.
# -----------------------------------------------------------------------------


class _Roulette:
    def __init__(self, ops: tuple[str, ...], reaction_factor: float):
        self.ops = list(ops)
        self.weights = [1.0] * len(ops)
        self.scores = [0.0] * len(ops)
        self.used = [0] * len(ops)
        self.r = reaction_factor

    def pick(self, rng: random.Random) -> int:
        total = sum(self.weights)
        x = rng.uniform(0.0, total)
        cum = 0.0
        for i, w in enumerate(self.weights):
            cum += w
            if x <= cum:
                return i
        return len(self.weights) - 1

    def reward(self, idx: int, score: float) -> None:
        self.scores[idx] += score
        self.used[idx] += 1

    def adapt(self) -> None:
        for i in range(len(self.weights)):
            if self.used[i]:
                self.weights[i] = (1 - self.r) * self.weights[i] + self.r * (
                    self.scores[i] / self.used[i]
                )
            self.scores[i] = 0.0
            self.used[i] = 0


# -----------------------------------------------------------------------------
# Solver entry point
# -----------------------------------------------------------------------------


def _state_to_solution(
    instance: Instance, state: _State, depot_by_id: dict[int, Depot], solver_name: str
) -> Solution:
    routes_out: list[Route] = []
    for rider_id, (depot_id, custs) in enumerate(
        zip(state.route_depots, state.routes, strict=True)
    ):
        rb = build_route(instance, depot_by_id[depot_id], custs)
        routes_out.append(
            Route(
                rider_id=rider_id,
                depot_id=depot_id,
                nodes=rb.nodes,
                arrivals=rb.arrivals,
            )
        )
    sol = Solution(
        routes=routes_out, objective=0.0, constraint_summary={}, solver=solver_name
    )
    sol.objective = eval_objective(instance, sol)
    return sol


@register
class ALNSSolver(Solver):
    name = "alns"

    def solve(self, instance: Instance) -> Solution:  # noqa: C901
        cfg = _cfg_from(self.cfg)
        rng = random.Random(cfg.seed ^ instance.seed)
        t0 = time.perf_counter()
        depot_by_id = {d.depot_id: d for d in instance.depots}

        # 1. Initial construction.
        current = _initial_solution(instance, depot_by_id, rng)
        if current.unassigned:
            # Try regret repair to place leftovers before falling back.
            _regret_repair(instance, current, rng, 2)
        if current.unassigned:
            # Last resort: drop each remaining customer into its own
            # singleton route (possibly infeasible).  The SA loop will try
            # to relocate it, and the strict validate() at the end is the
            # final gate.  This is preferable to aborting before SA has
            # any chance to repair pathological seeds.
            for c in current.unassigned:
                current.routes.append([c])
                current.route_depots.append(c.home_depot)
            current.unassigned = []
        best = current.copy()
        best_cost = _state_cost(instance, best, depot_by_id)
        cur_cost = best_cost
        T = max(1e-6, cfg.T_start_ratio * max(abs(best_cost), 1.0))

        destroy_r = _Roulette(cfg.destroy_ops, cfg.reaction_factor)
        repair_r = _Roulette(cfg.repair_ops, cfg.reaction_factor)

        for it in range(cfg.iterations):
            candidate = current.copy()
            di = destroy_r.pick(rng)
            ri = repair_r.pick(rng)
            q = _pick_destroy_count(instance, cfg, rng)
            _DESTROY[destroy_r.ops[di]](instance, candidate, q, rng)
            repaired = _REPAIR[repair_r.ops[ri]](instance, candidate, rng)
            if not repaired:
                continue
            cand_cost = _state_cost(instance, candidate, depot_by_id)

            reward = 0.0
            if cand_cost < best_cost - 1e-9:
                best = candidate.copy()
                best_cost = cand_cost
                current = candidate
                cur_cost = cand_cost
                reward = cfg.sigma[0]
            elif cand_cost < cur_cost - 1e-9:
                current = candidate
                cur_cost = cand_cost
                reward = cfg.sigma[1]
            else:
                # Simulated-annealing acceptance for worse candidates.
                delta = cand_cost - cur_cost
                if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                    current = candidate
                    cur_cost = cand_cost
                    reward = cfg.sigma[2]
                else:
                    reward = cfg.sigma[3]

            destroy_r.reward(di, reward)
            repair_r.reward(ri, reward)
            if (it + 1) % cfg.segment_length == 0:
                destroy_r.adapt()
                repair_r.adapt()
            T *= cfg.cooling_rate

        # 2. Convert best state to Solution and validate.
        sol = _state_to_solution(instance, best, depot_by_id, self.name)
        sol.run_meta = {
            "wall_clock_s": time.perf_counter() - t0,
            "iterations": cfg.iterations,
            "seed": cfg.seed ^ instance.seed,
            "final_cost": best_cost,
            "destroy_weights": dict(zip(destroy_r.ops, destroy_r.weights, strict=True)),
            "repair_weights": dict(zip(repair_r.ops, repair_r.weights, strict=True)),
        }
        report = validate(instance, sol)
        if not report.feasible:
            raise InfeasibleError(
                "ALNS best incumbent failed validation: "
                + "; ".join(str(v) for v in report.violations[:5])
            )
        sol.constraint_summary["feasible"] = 1.0
        return sol
