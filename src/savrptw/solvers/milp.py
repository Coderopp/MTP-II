"""MILP solver — exact (or best-effort time-limited) baseline.

Implements FORMULATION.md §8 via PuLP.  Backend: Gurobi if the `gurobipy`
module is available and licensed; otherwise CBC (bundled with PuLP).

Objective
---------
The exponential lateness penalty `exp(β·τ) − 1` is convex and is replaced in
the MILP by a tangent-line outer-approximation — the PWL value the MILP
minimises is a *lower bound* on the true F₁.  Actual F₁ is recomputed
post-solve from the realised arrival times (via `savrptw.eval.objective`).
The paper reports BOTH: the MILP objective (bound) and the true F₁.

Variables (per FORMULATION.md §4)
---------------------------------
* `x[i, j, k] ∈ {0, 1}` — rider k uses super-arc (i, j).
* `a[i, k] ≥ 0` — arrival time at node i by rider k.
* `w[i, k] ≥ 0` — idle-wait at customer i by rider k.
* `tau[i, k] ≥ 0` — lateness at customer i by rider k (capped at τ_max).
* `pen[i, k] ≥ 0` — PWL approximation of exp(β·τ[i,k]) − 1.
* `u[i, k] ∈ ℤ_{≥1}` — MTZ sequence variable.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import pulp

from savrptw.eval.feasibility import validate
from savrptw.eval.objective import objective as eval_objective
from savrptw.solvers._common import build_route
from savrptw.solvers.base import InfeasibleError, Solver, register
from savrptw.types import Customer, Depot, Instance, Route, Solution


@dataclass
class MILPConfig:
    backend: str = "auto"          # "gurobi" | "cbc" | "auto"
    time_limit_s: int = 300
    mip_gap: float = 0.01
    threads: int = 8
    log_to_console: bool = False
    stw_pwl_breakpoints: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0)


def _cfg_from(cfg) -> MILPConfig:
    get = (lambda k, d: cfg.get(k, d)) if hasattr(cfg, "get") else (lambda k, d: d)
    bps = get("stw_pwl_breakpoints", None)
    if bps is not None:
        bps = tuple(float(b) for b in bps)
    else:
        bps = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0)
    return MILPConfig(
        backend=str(get("backend", "auto")),
        time_limit_s=int(get("time_limit_s", 300)),
        mip_gap=float(get("mip_gap", 0.01)),
        threads=int(get("threads", 8)),
        log_to_console=bool(get("log_to_console", False)),
        stw_pwl_breakpoints=bps,
    )


def _pick_solver(mcfg: MILPConfig) -> pulp.LpSolver:
    """Pick PuLP backend respecting the "gurobi if possible" preference."""
    prefer_gurobi = mcfg.backend in ("auto", "gurobi")
    if prefer_gurobi:
        try:
            import gurobipy  # noqa: F401

            return pulp.GUROBI_CMD(
                msg=int(mcfg.log_to_console),
                timeLimit=mcfg.time_limit_s,
                gapRel=mcfg.mip_gap,
                threads=mcfg.threads,
            )
        except Exception:
            if mcfg.backend == "gurobi":
                raise
    # CBC fallback — disable aggressive presolve that can declare
    # big-M formulations spuriously infeasible.
    return pulp.PULP_CBC_CMD(
        msg=int(mcfg.log_to_console),
        timeLimit=mcfg.time_limit_s,
        gapRel=mcfg.mip_gap,
        threads=mcfg.threads,
        options=["presolve off", "ratioGap 0.05"],
    )


def _stw_pwl_rows(breakpoints: tuple[float, ...], beta: float) -> list[tuple[float, float, float]]:
    """Tangent lines of f(τ) = exp(β·τ) − 1 at each breakpoint.

    Returns tuples `(τ_k, f_k, f'_k)` so constraints can be written
        pen ≥ f_k + f'_k · (τ − τ_k).
    """
    rows = []
    for tk in breakpoints:
        fk = math.exp(beta * tk) - 1.0
        dfk = beta * math.exp(beta * tk)
        rows.append((tk, fk, dfk))
    return rows


def _fleet_size_per_depot(instance: Instance, depot_id: int) -> int:
    """Over-provision riders per depot per FORMULATION.md §3.4."""
    assigned = sum(1 for c in instance.customers if c.home_depot == depot_id)
    if assigned == 0:
        return 0
    # ⌈1.2 · N_d / Q⌉
    return max(1, math.ceil(1.2 * assigned / max(1, instance.Q)))


@register
class MILPSolver(Solver):
    name = "milp"

    def solve(self, instance: Instance) -> Solution:  # noqa: C901
        mcfg = _cfg_from(self.cfg)
        t0 = time.perf_counter()

        depots = instance.depots
        customers = instance.customers
        if not customers:
            empty = Solution(
                routes=[], objective=0.0, constraint_summary={}, solver=self.name
            )
            rep = validate(instance, empty)
            if not rep.feasible:
                raise InfeasibleError(str(rep.violations))
            return empty

        # Index space — single integer namespace across depots + customers.
        node_ids: list[int] = [d.depot_id for d in depots] + [c.customer_id for c in customers]
        cust_ids: list[int] = [c.customer_id for c in customers]
        depot_ids: list[int] = [d.depot_id for d in depots]
        cust_by_id: dict[int, Customer] = {c.customer_id: c for c in customers}
        depot_by_id: dict[int, Depot] = {d.depot_id: d for d in depots}

        # Each depot gets its own rider set; rider k is tagged with its home depot.
        rider_to_depot: dict[int, int] = {}
        k = 0
        for d in depots:
            for _ in range(_fleet_size_per_depot(instance, d.depot_id)):
                rider_to_depot[k] = d.depot_id
                k += 1
        if not rider_to_depot:
            raise InfeasibleError("fleet size is zero — no customers to serve?")
        riders: list[int] = list(rider_to_depot.keys())

        # Prune arcs per rider:
        #  - rider can only traverse (i, j) if both endpoints are its home depot
        #    or customers whose home_depot matches (FORMULATION.md constraint 6).
        def rider_allows(kidx: int, i: int, j: int) -> bool:
            d_home = rider_to_depot[kidx]
            if i == j:
                return False
            for nid in (i, j):
                if nid in depot_by_id:
                    if nid != d_home:
                        return False
                else:
                    if cust_by_id[nid].home_depot != d_home:
                        return False
            return True

        prob = pulp.LpProblem("SA_VRPTW", pulp.LpMinimize)

        # --- decision variables ---------------------------------------------------
        x = {}
        for kidx in riders:
            for i in node_ids:
                for j in node_ids:
                    if not rider_allows(kidx, i, j):
                        continue
                    if (i, j) not in instance.super_arcs:
                        continue
                    x[(i, j, kidx)] = pulp.LpVariable(
                        f"x_{i}_{j}_{kidx}", lowBound=0, upBound=1, cat="Binary"
                    )

        bigM = max(instance.T_max * 2.0, 240.0)
        tau_max = float(mcfg.stw_pwl_breakpoints[-1])

        # Arrival-time variables.  NOTE: we DO NOT reuse one `a[depot, k]` for
        # both dispatch and return — that creates a self-contradiction whenever
        # a rider actually uses its tour.  Instead:
        #   * `a[i, k]`  for i ∈ C is arrival at customer i by rider k,
        #   * `a_return[k]` is the return time at rider k's home depot,
        #   * dispatch time is an implicit 0 (not a variable).
        a = {
            (i, kidx): pulp.LpVariable(f"a_{i}_{kidx}", lowBound=0, upBound=bigM)
            for i in cust_ids
            for kidx in riders
        }
        a_return = {
            kidx: pulp.LpVariable(f"aret_{kidx}", lowBound=0, upBound=bigM)
            for kidx in riders
        }
        w = {
            (i, kidx): pulp.LpVariable(f"w_{i}_{kidx}", lowBound=0, upBound=bigM)
            for i in cust_ids
            for kidx in riders
        }
        tau = {
            (i, kidx): pulp.LpVariable(f"tau_{i}_{kidx}", lowBound=0, upBound=tau_max)
            for i in cust_ids
            for kidx in riders
        }
        pen = {
            (i, kidx): pulp.LpVariable(f"pen_{i}_{kidx}", lowBound=0)
            for i in cust_ids
            for kidx in riders
        }
        u = {
            (i, kidx): pulp.LpVariable(f"u_{i}_{kidx}", lowBound=1, upBound=len(cust_ids))
            for i in cust_ids
            for kidx in riders
        }

        # --- objective ------------------------------------------------------------
        prob += pulp.lpSum(
            instance.w_early * w[(i, kidx)] + pen[(i, kidx)]
            for i in cust_ids
            for kidx in riders
        )

        # --- visit-once (constraint 1) -------------------------------------------
        for i in cust_ids:
            prob += (
                pulp.lpSum(
                    x[(j, i, kidx)]
                    for kidx in riders
                    for j in node_ids
                    if (j, i, kidx) in x
                )
                == 1,
                f"visit_once_{i}",
            )

        # --- flow conservation (2) ------------------------------------------------
        for kidx in riders:
            for j in cust_ids:
                prob += (
                    pulp.lpSum(x[(i, j, kidx)] for i in node_ids if (i, j, kidx) in x)
                    == pulp.lpSum(x[(j, i, kidx)] for i in node_ids if (j, i, kidx) in x),
                    f"flow_{j}_{kidx}",
                )

        # --- depot leaves/returns (4)(5) -----------------------------------------
        for kidx in riders:
            d_home = rider_to_depot[kidx]
            leaves = pulp.lpSum(
                x[(d_home, j, kidx)] for j in cust_ids if (d_home, j, kidx) in x
            )
            returns = pulp.lpSum(
                x[(i, d_home, kidx)] for i in cust_ids if (i, d_home, kidx) in x
            )
            prob += (leaves <= 1, f"leave_once_{kidx}")
            prob += (leaves == returns, f"close_tour_{kidx}")

        # --- capacity (3) --------------------------------------------------------
        for kidx in riders:
            prob += (
                pulp.lpSum(
                    cust_by_id[j].demand * x[(i, j, kidx)]
                    for i in node_ids
                    for j in cust_ids
                    if (i, j, kidx) in x
                )
                <= instance.Q,
                f"capacity_{kidx}",
            )

        # --- MTZ sub-tour elimination (7)(8) -------------------------------------
        Ncap = len(cust_ids)
        for kidx in riders:
            for i in cust_ids:
                for j in cust_ids:
                    if i == j:
                        continue
                    if (i, j, kidx) not in x:
                        continue
                    prob += (
                        u[(i, kidx)] - u[(j, kidx)] + Ncap * x[(i, j, kidx)] <= Ncap - 1,
                        f"mtz_{i}_{j}_{kidx}",
                    )

        # --- arrival-time linking (9) --------------------------------------------
        # Four cases — dispatch (depot→customer), customer→customer,
        # customer→depot (return), and depot→depot (disallowed via
        # rider_allows + self-loop skipping, so no constraint needed).
        for kidx in riders:
            d_home = rider_to_depot[kidx]
            for (i, j, kk), xij in list(x.items()):
                if kk != kidx:
                    continue
                arc = instance.super_arcs[(i, j)]
                if i == d_home and j in cust_by_id:
                    # dispatch — implicit a_start = 0.
                    prob += (
                        a[(j, kidx)] >= arc.T_uv - bigM * (1 - xij),
                        f"arr_dispatch_{j}_{kidx}",
                    )
                elif i in cust_by_id and j in cust_by_id:
                    s_i = cust_by_id[i].service_time
                    prob += (
                        a[(j, kidx)] >= a[(i, kidx)] + s_i + arc.T_uv - bigM * (1 - xij),
                        f"arr_{i}_{j}_{kidx}",
                    )
                elif i in cust_by_id and j == d_home:
                    s_i = cust_by_id[i].service_time
                    prob += (
                        a_return[kidx] >= a[(i, kidx)] + s_i + arc.T_uv - bigM * (1 - xij),
                        f"arr_return_{i}_{kidx}",
                    )
                # depot→depot arcs are not in x (self-loop excluded).

        # --- soft TW linking (11)(12) --------------------------------------------
        # The linking constraints must be deactivated for riders that do not
        # visit a given customer — otherwise a=0 (the default bound) forces
        # w >= e_i > 0 everywhere and renders the MILP spuriously infeasible.
        for i in cust_ids:
            c = cust_by_id[i]
            for kidx in riders:
                visit_ki = pulp.lpSum(
                    x[(j, i, kidx)] for j in node_ids if (j, i, kidx) in x
                )
                prob += (
                    w[(i, kidx)] >= c.e_i - a[(i, kidx)] - bigM * (1 - visit_ki),
                    f"w_{i}_{kidx}",
                )
                prob += (
                    tau[(i, kidx)] >= a[(i, kidx)] - c.eta_i - bigM * (1 - visit_ki),
                    f"tau_{i}_{kidx}",
                )

        # --- PWL outer approximation of exp(β·τ) − 1 -----------------------------
        rows = _stw_pwl_rows(mcfg.stw_pwl_breakpoints, instance.beta_stw)
        for i in cust_ids:
            for kidx in riders:
                for tk, fk, dfk in rows:
                    prob += (
                        pen[(i, kidx)] >= fk + dfk * (tau[(i, kidx)] - tk),
                        f"pwl_{i}_{kidx}_{tk}",
                    )

        # --- ε-constraints (§6) --------------------------------------------------
        for kidx in riders:
            prob += (
                pulp.lpSum(
                    instance.super_arcs[(i, j)].R_uv * x[(i, j, kidx)]
                    for (i, j, kk) in x.keys()
                    if kk == kidx
                )
                <= instance.R_bar,
                f"risk_budget_{kidx}",
            )
            prob += (
                pulp.lpSum(
                    instance.super_arcs[(i, j)].H_uv * x[(i, j, kidx)]
                    for (i, j, kk) in x.keys()
                    if kk == kidx
                )
                <= instance.H_cap_route,
                f"hres_route_{kidx}",
            )
        prob += (
            pulp.lpSum(
                instance.super_arcs[(i, j)].H_uv * x[(i, j, kidx)]
                for (i, j, kidx) in x.keys()
            )
            <= instance.H_bar,
            "hres_fleet",
        )

        # --- route-duration (10) -------------------------------------------------
        for kidx in riders:
            # Duration = return time at home depot; see a_return linkage above.
            prob += (a_return[kidx] <= instance.T_max, f"duration_{kidx}")

        # --- solve ---------------------------------------------------------------
        solver = _pick_solver(mcfg)
        status = prob.solve(solver)
        status_str = pulp.LpStatus.get(status, "Unknown")
        mip_gap = None

        if status_str == "Infeasible":
            raise InfeasibleError(f"MILP proven infeasible for R̄={instance.R_bar}, H̄={instance.H_bar}")
        if status_str in ("Undefined", "Unbounded", "Not Solved"):
            # No incumbent to extract — bail out clearly rather than emitting garbage.
            raise InfeasibleError(
                f"MILP terminated with status {status_str!r} — no usable incumbent"
            )

        # --- extract routes ------------------------------------------------------
        routes_out: list[Route] = []
        for kidx in riders:
            # Reconstruct sequence from x[(i,j,k)] == 1.
            successors: dict[int, int] = {}
            for (i, j, kk), var in x.items():
                if kk != kidx:
                    continue
                val = pulp.value(var)
                if val is not None and val > 0.5:
                    successors[i] = j
            if not successors:
                continue
            d_home = rider_to_depot[kidx]
            # Rider must leave the depot for at least one customer — if the
            # only arc starts/ends at the depot, it's an artefact of slack and
            # the rider is effectively idle.
            if d_home not in successors:
                continue
            if successors[d_home] == d_home:
                continue
            seq = [d_home]
            visited = {d_home}
            cur = d_home
            while cur in successors:
                nxt = successors[cur]
                seq.append(nxt)
                if nxt == d_home:
                    break
                if nxt in visited:
                    raise InfeasibleError(
                        f"MILP returned a cycle among customers for rider {kidx}"
                    )
                visited.add(nxt)
                cur = nxt
            if seq[-1] != d_home:
                raise InfeasibleError(
                    f"rider {kidx}: path {seq} did not return to depot {d_home}"
                )
            # Skip degenerate depot→depot tours (no customers visited).
            if len(seq) <= 2:
                continue

            # Recompute arrivals via the canonical builder to avoid drift from
            # the MILP's `a` variables (PWL slack can offset).
            custs = [cust_by_id[nid] for nid in seq[1:-1]]
            rb = build_route(instance, depot_by_id[d_home], custs)
            routes_out.append(
                Route(
                    rider_id=kidx,
                    depot_id=d_home,
                    nodes=rb.nodes,
                    arrivals=rb.arrivals,
                )
            )

        # --- evaluate, validate, return -----------------------------------------
        sol = Solution(
            routes=routes_out,
            objective=0.0,
            constraint_summary={
                "milp_status": float(status),
                "milp_pwl_bound": float(pulp.value(prob.objective) or 0.0),
            },
            solver=self.name,
        )
        sol.objective = eval_objective(instance, sol)
        sol.run_meta = {
            "wall_clock_s": time.perf_counter() - t0,
            "backend": solver.name if hasattr(solver, "name") else type(solver).__name__,
            "status": status_str,
            "pwl_bound": sol.constraint_summary["milp_pwl_bound"],
            "mip_gap": mip_gap,
        }
        report = validate(instance, sol)
        if not report.feasible:
            raise InfeasibleError(
                "MILP solution failed validation: "
                + "; ".join(str(v) for v in report.violations[:5])
            )
        sol.constraint_summary["feasible"] = 1.0
        return sol


__all__: list[str] = ["MILPSolver"]

# Keep `Any` alias stable for type-checkers reading this module.
_Any = Any  # noqa: N816
