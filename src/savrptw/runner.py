"""End-to-end experiment runner — instance → solver → evaluation → results.

Called from `scripts/run_experiment.py` (the Hydra entrypoint) or directly
with a composed DictConfig for programmatic use.

Results are emitted as a single row per (seed, solver, city, N, R̄, H̄) into
`<paths.results>/<run_id>.json` — ready for downstream aggregation and plotting.
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from savrptw.eval.feasibility import validate
from savrptw.eval.objective import breakdown
from savrptw.instance.generator import build_instance
from savrptw.sim.behavioral import sensitivity_sweep
from savrptw.sim.crash_mc import analytic
from savrptw.solvers import build as build_solver
from savrptw.types import Instance, Solution


def _default(obj: Any) -> Any:
    """JSON default — dataclasses and numpy scalars become plain dicts/floats."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Not JSON-serializable: {type(obj).__name__}")


def run_one(
    cfg: DictConfig,
    *,
    instance: Instance | None = None,
) -> dict[str, Any]:
    """Run a single experiment row and emit a result dict.

    If `instance` is supplied, skip `build_instance` (used by tests to feed
    synthetic fixtures without OSM/BASM dependencies).
    """
    run_id = str(uuid.uuid4())[:8]
    t_instance_start = time.perf_counter()
    if instance is None:
        instance = build_instance(cfg)
    t_instance = time.perf_counter() - t_instance_start

    t_solve = time.perf_counter()
    solver = build_solver(cfg.solver.name, cfg.solver)
    solution: Solution = solver.solve(instance)
    solve_wallclock = time.perf_counter() - t_solve

    validation = validate(instance, solution)
    if not validation.feasible:
        # A solver that returns here is buggy — every solver validates before
        # returning.  We still capture the report for forensics.
        pass

    # Post-hoc evaluators.
    crash = analytic(instance, solution, n_trips=10_000)
    behav = sensitivity_sweep(
        instance,
        solution,
        alphas=(3.0, 5.0, 8.0),
        betas=(1.0, 2.0, 3.0),
        n_riders_per_route=50,
        n_trips=5_000,
        seed=int(cfg.instance.seed),
    )

    row: dict[str, Any] = {
        "run_id": run_id,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "objective_breakdown": breakdown(instance, solution),
        "instance_meta": instance.meta,
        "N": instance.N,
        "n_depots": instance.n_depots,
        "Q": instance.Q,
        "T_max": instance.T_max,
        "R_bar": instance.R_bar,
        "H_bar": instance.H_bar,
        "solver": solution.solver,
        "F1": solution.objective,
        "n_routes": len(solution.routes),
        "feasible": validation.feasible,
        "violations": validation.as_dict()["violations"],
        "solve_wallclock_s": solve_wallclock,
        "instance_build_s": t_instance,
        "solution_run_meta": solution.run_meta,
        "crash_mc": crash.as_dict(),
        "behavioral_mc": behav.as_dict(),
    }

    out_dir = Path(cfg.paths.results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with out_path.open("w") as f:
        json.dump(row, f, default=_default, indent=2)
    row["output_path"] = str(out_path)
    return row
