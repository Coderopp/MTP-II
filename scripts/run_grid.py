"""Batch driver for the full SA-VRPTW experimental grid.

Iteration order is chosen to amortise the slowest steps:

    for city in CITIES:                   # OSM graph load (slow, 1x per city)
        for N in N_VALUES:                # instance build (BPR + risk + sample)
            for seed in SEEDS:            # 1 customer-resample per seed
                for solver in SOLVERS:    # solve + evaluate

Per-row results land in `<paths.results>/<run_id>.json`. A progress
manifest at `<paths.results>/_manifest.jsonl` records each completed
(city, N, seed, solver) tuple so the driver can resume from a crash by
skipping rows already present.

Usage:

    PYTHONPATH=src python scripts/run_grid.py
    PYTHONPATH=src python scripts/run_grid.py --solvers ga,alns --n-seeds 3
    PYTHONPATH=src python scripts/run_grid.py --pareto  # ε-sweep at N=50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from savrptw.runner import run_one  # noqa: E402


def already_done(manifest: Path, key: tuple) -> bool:
    if not manifest.exists():
        return False
    with manifest.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if tuple(rec.get("key", [])) == list(key):
                return True
    return False


def append_manifest(manifest: Path, key: tuple, output_path: str, **extra) -> None:
    rec = {"key": list(key), "output_path": output_path, "ts": time.time(), **extra}
    with manifest.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def compose_cfg(
    conf_dir: Path,
    *,
    city: str,
    solver: str,
    N: int,
    seed: int,
    n_depots: int,
    R_bar: float | None = None,
    H_bar: int | None = None,
    milp_time_limit: int | None = None,
) -> DictConfig:
    overrides = [
        f"city={city}",
        f"solver={solver}",
        f"instance.N={N}",
        f"instance.seed={seed}",
        f"instance.n_depots={n_depots}",
    ]
    if R_bar is not None:
        overrides.append(f"instance.R_bar={R_bar}")
    if H_bar is not None:
        overrides.append(f"instance.H_bar={H_bar}")
    if milp_time_limit is not None and solver == "milp":
        overrides.append(f"solver.time_limit_s={milp_time_limit}")
    with initialize_config_dir(config_dir=str(conf_dir.resolve()), version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def run_default_grid(
    *,
    conf_dir: Path,
    cities: Iterable[str],
    n_values: Iterable[int],
    seeds: Iterable[int],
    solvers: Iterable[str],
    depot_count_by_N: dict,
    results_dir: Path,
    milp_max_N: int,
) -> None:
    """Default ε (R_bar=∞, H_bar=∞) sweep over (city, N, seed, solver)."""
    manifest = results_dir / "_manifest.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for city in cities:
        for N in n_values:
            for seed in seeds:
                for solver in solvers:
                    if solver == "milp" and N > milp_max_N:
                        continue
                    total += 1

    done = 0
    failed = 0
    for city in cities:
        city_t0 = time.time()
        for N in n_values:
            for seed in seeds:
                for solver in solvers:
                    if solver == "milp" and N > milp_max_N:
                        continue
                    key = ("default", city, N, seed, solver)
                    if already_done(manifest, key):
                        done += 1
                        continue
                    print(
                        f"[{done + 1}/{total}] {city} N={N} seed={seed} solver={solver}",
                        flush=True,
                    )
                    t0 = time.time()
                    try:
                        cfg = compose_cfg(
                            conf_dir,
                            city=city,
                            solver=solver,
                            N=N,
                            seed=seed,
                            n_depots=int(depot_count_by_N[N]),
                            R_bar=10000.0,
                            H_bar=10**6,
                            milp_time_limit=120,
                        )
                        row = run_one(cfg)
                        append_manifest(
                            manifest,
                            key,
                            row["output_path"],
                            wall=time.time() - t0,
                            F1=row.get("F1"),
                            feasible=row.get("feasible"),
                        )
                        done += 1
                        print(
                            f"   ok  F1={row.get('F1'):.3f} feasible={row.get('feasible')} "
                            f"({time.time() - t0:.1f}s)",
                            flush=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        failed += 1
                        traceback.print_exc()
                        append_manifest(
                            manifest,
                            key,
                            "",
                            wall=time.time() - t0,
                            error=str(e),
                        )
                        print(f"   FAIL {e}", flush=True)
        print(
            f"== {city} done in {(time.time() - city_t0) / 60:.1f}min "
            f"(cumulative ok={done}, fail={failed})",
            flush=True,
        )


def run_pareto_sweep(
    *,
    conf_dir: Path,
    cities: Iterable[str],
    R_bar_values: Iterable[float],
    H_bar_values: Iterable[int],
    seeds: Iterable[int],
    solvers: Iterable[str],
    N: int,
    n_depots: int,
    results_dir: Path,
) -> None:
    """ε-constraint Pareto sweep at fixed (N, n_depots) per city."""
    manifest = results_dir / "_manifest.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)

    for city in cities:
        for R_bar in R_bar_values:
            for H_bar in H_bar_values:
                for seed in seeds:
                    for solver in solvers:
                        key = ("pareto", city, N, seed, solver, float(R_bar), int(H_bar))
                        if already_done(manifest, key):
                            continue
                        print(
                            f"PARETO {city} N={N} seed={seed} {solver} "
                            f"R_bar={R_bar} H_bar={H_bar}",
                            flush=True,
                        )
                        t0 = time.time()
                        try:
                            cfg = compose_cfg(
                                conf_dir,
                                city=city,
                                solver=solver,
                                N=N,
                                seed=seed,
                                n_depots=n_depots,
                                R_bar=float(R_bar),
                                H_bar=int(H_bar),
                            )
                            row = run_one(cfg)
                            append_manifest(
                                manifest,
                                key,
                                row["output_path"],
                                wall=time.time() - t0,
                                F1=row.get("F1"),
                                feasible=row.get("feasible"),
                            )
                            print(
                                f"   ok F1={row.get('F1'):.3f} ({time.time() - t0:.1f}s)",
                                flush=True,
                            )
                        except Exception as e:  # noqa: BLE001
                            traceback.print_exc()
                            append_manifest(
                                manifest,
                                key,
                                "",
                                wall=time.time() - t0,
                                error=str(e),
                            )
                            print(f"   FAIL {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="bengaluru,delhi,gurugram,mumbai,pune")
    ap.add_argument("--n-values", default="20,50,100,200")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--solvers", default="ga,alns,milp")
    ap.add_argument("--milp-max-N", type=int, default=50)
    ap.add_argument("--pareto", action="store_true",
                    help="Also run ε-sweep Pareto grid at N=50")
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--conf", default="conf")
    args = ap.parse_args()

    conf_dir = Path(args.conf)
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    n_values = [int(x) for x in args.n_values.split(",")]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]

    with initialize_config_dir(config_dir=str(conf_dir.resolve()), version_base="1.3"):
        cfg = compose(config_name="config")
    seeds = list(cfg.repro.seeds)[: args.n_seeds]
    depot_count_by_N = dict(cfg.experiment.depot_count_by_N)

    results_dir = Path(args.results_dir)
    print(f"Results -> {results_dir}/", flush=True)
    print(f"Cities: {cities}", flush=True)
    print(f"N values: {n_values}", flush=True)
    print(f"Seeds ({len(seeds)}): {seeds}", flush=True)
    print(f"Solvers: {solvers} (MILP capped at N≤{args.milp_max_N})", flush=True)

    grid_t0 = time.time()
    run_default_grid(
        conf_dir=conf_dir,
        cities=cities,
        n_values=n_values,
        seeds=seeds,
        solvers=solvers,
        depot_count_by_N=depot_count_by_N,
        results_dir=results_dir,
        milp_max_N=args.milp_max_N,
    )

    if args.pareto:
        run_pareto_sweep(
            conf_dir=conf_dir,
            cities=cities,
            R_bar_values=list(cfg.experiment.R_bar_values),
            H_bar_values=list(cfg.experiment.H_bar_values),
            seeds=seeds[:3],  # 3 seeds for the Pareto sweep
            solvers=[s for s in solvers if s != "milp"],
            N=50,
            n_depots=int(depot_count_by_N[50]),
            results_dir=results_dir,
        )

    print(f"\nGrid finished in {(time.time() - grid_t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
