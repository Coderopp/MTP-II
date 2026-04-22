"""Genetic Algorithm for SA-VRPTW.

Representation
--------------
Giant-tour chromosome: a permutation of customer ids across ALL depots.
Decoding groups customers by their pre-assigned `home_depot`, then each
depot-group is split by Bellman-Ford (optimal) — see `_common.bellman_ford_split`.

Operators
---------
* Selection: tournament (k configurable, default 5)
* Crossover: OX1 (ordered crossover) — preserves permutation validity
* Mutations: swap, insertion, 2-opt-style reversal (random choice)
* Elitism: top-K individuals survive unchanged each generation

Search uses a surrogate objective `F₁ + violation_mass` (see `_common.build_route`
for violation accounting).  The final incumbent is validated strictly; if no
feasible solution exists the solver raises `InfeasibleError`.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from savrptw.eval.feasibility import validate
from savrptw.solvers._common import assemble_solution
from savrptw.solvers.base import InfeasibleError, Solver, register
from savrptw.types import Instance, Solution


@dataclass
class GAConfig:
    pop_size: int = 100
    generations: int = 200
    cx_prob: float = 0.85
    mut_prob: float = 0.20
    elite_size: int = 4
    tournament_k: int = 5
    penalties: dict = None  # type: ignore[assignment]
    seed: int = 42


def _cfg_from(cfg) -> GAConfig:
    """Build a GAConfig, tolerating Hydra DictConfig or plain dict."""
    get = (lambda k, d: cfg.get(k, d)) if hasattr(cfg, "get") else (lambda k, d: d)
    penalties = get("penalties", None)
    if penalties is not None and hasattr(penalties, "items"):
        penalties = dict(penalties)
    return GAConfig(
        pop_size=int(get("pop_size", 100)),
        generations=int(get("generations", 200)),
        cx_prob=float(get("cx_prob", 0.85)),
        mut_prob=float(get("mut_prob", 0.20)),
        elite_size=int(get("elite_size", 4)),
        tournament_k=int(get("tournament_k", 5)),
        penalties=penalties or {
            "tw_late": 50.0,
            "tw_early": 5.0,
            "capacity": 1000.0,
            "duration": 1000.0,
            "risk_budget": 1000.0,
            "residential_route_cap": 500.0,
            "missing_arc": 1e6,
        },
        seed=int(get("seed", 42)),
    )


def _ox1(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
    """Ordered crossover OX1 — returns a single child as a permutation."""
    n = len(p1)
    if n < 2:
        return list(p1)
    a, b = sorted(rng.sample(range(n), 2))
    hole = set(p1[a:b])
    child = [None] * n
    child[a:b] = p1[a:b]
    # Fill remaining slots in p2-order, starting from position b.
    p2_cycle = p2[b:] + p2[:b]
    idx = b
    for gene in p2_cycle:
        if gene in hole:
            continue
        if idx >= n:
            idx = 0
        while child[idx] is not None:
            idx += 1
            if idx >= n:
                idx = 0
        child[idx] = gene
        idx += 1
    # Safety net (should never trip): fill any remaining with p2 order.
    if None in child:
        leftover = [g for g in p2 if g not in child]
        for i, _ in enumerate(child):
            if child[i] is None:
                child[i] = leftover.pop(0)
    return child  # type: ignore[return-value]


def _mutate(ind: list[int], rng: random.Random) -> None:
    """Random choice of swap / insertion / reversal."""
    n = len(ind)
    if n < 2:
        return
    op = rng.choice(("swap", "insert", "reverse"))
    if op == "swap":
        i, j = rng.sample(range(n), 2)
        ind[i], ind[j] = ind[j], ind[i]
    elif op == "insert":
        i = rng.randrange(n)
        j = rng.randrange(n)
        gene = ind.pop(i)
        ind.insert(j, gene)
    else:  # reverse
        i, j = sorted(rng.sample(range(n), 2))
        ind[i : j + 1] = list(reversed(ind[i : j + 1]))


@register
class GASolver(Solver):
    name = "ga"

    def solve(self, instance: Instance) -> Solution:
        gcfg = _cfg_from(self.cfg)
        rng = random.Random(gcfg.seed ^ instance.seed)
        customers = instance.customers
        if not customers:
            # Edge case: nothing to do — return an empty validated solution.
            empty = Solution(routes=[], objective=0.0, constraint_summary={}, solver=self.name)
            rep = validate(instance, empty)
            if not rep.feasible:
                raise InfeasibleError(str(rep.violations))
            return empty

        t0 = time.perf_counter()

        cust_ids = [c.customer_id for c in customers]

        def fresh_chrom() -> list[int]:
            x = cust_ids[:]
            rng.shuffle(x)
            return x

        # Initial population.
        population: list[tuple[float, list[int]]] = []
        for _ in range(gcfg.pop_size):
            chrom = fresh_chrom()
            _, surrogate, _ = assemble_solution(
                instance, chrom, solver_name=self.name, penalty_weights=gcfg.penalties
            )
            population.append((surrogate, chrom))
        population.sort(key=lambda t: t[0])

        best_surrogate = population[0][0]
        best_chrom = list(population[0][1])

        def tournament() -> list[int]:
            contenders = rng.sample(population, min(gcfg.tournament_k, len(population)))
            return list(min(contenders, key=lambda t: t[0])[1])

        for _gen in range(gcfg.generations):
            elites = [list(c) for _, c in population[: gcfg.elite_size]]
            offspring: list[list[int]] = []

            while len(offspring) + len(elites) < gcfg.pop_size:
                p1 = tournament()
                p2 = tournament()
                child = _ox1(p1, p2, rng) if rng.random() < gcfg.cx_prob else list(p1)
                if rng.random() < gcfg.mut_prob:
                    _mutate(child, rng)
                offspring.append(child)

            new_pop: list[tuple[float, list[int]]] = []
            for chrom in elites + offspring:
                _, surrogate, _ = assemble_solution(
                    instance, chrom, solver_name=self.name, penalty_weights=gcfg.penalties
                )
                new_pop.append((surrogate, chrom))
            new_pop.sort(key=lambda t: t[0])
            population = new_pop[: gcfg.pop_size]

            if population[0][0] < best_surrogate:
                best_surrogate = population[0][0]
                best_chrom = list(population[0][1])

        # Decode the incumbent and validate it strictly.
        best_solution, _surrogate, _hint = assemble_solution(
            instance, best_chrom, solver_name=self.name, penalty_weights=gcfg.penalties
        )
        best_solution.run_meta = {
            "wall_clock_s": time.perf_counter() - t0,
            "generations": gcfg.generations,
            "pop_size": gcfg.pop_size,
            "seed": gcfg.seed ^ instance.seed,
            "surrogate": best_surrogate,
        }
        report = validate(instance, best_solution)
        if not report.feasible:
            raise InfeasibleError(
                "GA produced no feasible incumbent — "
                f"{len(report.violations)} violations: "
                + "; ".join(str(v) for v in report.violations[:5])
            )
        best_solution.constraint_summary["feasible"] = 1.0
        return best_solution
