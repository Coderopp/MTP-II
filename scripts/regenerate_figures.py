"""Regenerate all paper figures from data/results/*.json.

Reads every per-row result JSON written by `scripts/run_grid.py` and
emits the 9 figure types per city under `figures/<City>/`:

    1. scalability_runtime.png      — wall-clock vs N, by solver
    2. optimality_gap.png           — F1 normalised by best per (N, seed)
    3. pareto_frontier.png          — R_bar sweep at N=50, F1 vs survival
    4. ga_convergence.png           — best-so-far vs generation (if logged)
    5. density_risk_box.png         — F1 distribution per N, by city
    6. city_risk_box.png            — per-route survival distribution
    7. pareto_3d.png                — F1 / risk / residential 3-axis scatter
    8. smax_fatigue.png             — behavioural-MC rider survival sweep
    9. stw_scatter.png              — lateness vs F1 contribution

Cells with insufficient data are skipped with a printed warning. The
script is safe to run on a partial grid; figures regenerate on
subsequent passes as more rows land.

Usage:

    PYTHONPATH=src python3 scripts/regenerate_figures.py \\
        --results-dir data/results --out-dir figures
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CITIES_TITLE = {
    "bengaluru": "Bengaluru",
    "delhi": "Delhi",
    "gurugram": "Gurugram",
    "mumbai": "Mumbai",
    "pune": "Pune",
}


def load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for jf in sorted(results_dir.glob("*.json")):
        if jf.name.startswith("_"):
            continue
        try:
            rows.append(json.loads(jf.read_text()))
        except json.JSONDecodeError:
            print(f"  WARN: skipping malformed {jf.name}")
    return rows


def by_city(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        slug = (r.get("config", {}).get("city", {}) or {}).get("slug")
        if slug:
            out[slug].append(r)
    return out


def fig_scalability(rows: list[dict], out: Path, city: str) -> None:
    """Wall-clock vs N for each solver (mean + 95% CI band)."""
    by_solver: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_solver[r["solver"]][r["N"]].append(r["solve_wallclock_s"])
    if not by_solver:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for solver, by_N in by_solver.items():
        Ns = sorted(by_N.keys())
        means = [np.mean(by_N[n]) for n in Ns]
        sems = [np.std(by_N[n], ddof=1) / max(math.sqrt(len(by_N[n])), 1) for n in Ns]
        ax.errorbar(Ns, means, yerr=[1.96 * s for s in sems], marker="o",
                    capsize=3, label=solver.upper())
    ax.set_xlabel("Customers per instance N")
    ax.set_ylabel("Wall-clock seconds (mean ± 95% CI)")
    ax.set_yscale("log")
    ax.set_title(f"{CITIES_TITLE.get(city, city)} — Solver scalability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "1_scalability.png", dpi=150)
    plt.close(fig)


def fig_optimality_gap(rows: list[dict], out: Path, city: str) -> None:
    """F1 normalised by best per (N, seed) — relative gap to best-found."""
    grouped: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["N"], r["config"]["instance"]["seed"])
        grouped[key][r["solver"]] = r["F1"]
    by_solver_N: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (N, _seed), per_solver in grouped.items():
        if not per_solver:
            continue
        best = min(per_solver.values())
        if best <= 0:
            continue
        for s, v in per_solver.items():
            by_solver_N[s][N].append((v - best) / best * 100.0)
    if not by_solver_N:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for solver, by_N in by_solver_N.items():
        Ns = sorted(by_N.keys())
        means = [np.mean(by_N[n]) for n in Ns]
        ax.plot(Ns, means, marker="s", label=solver.upper())
    ax.set_xlabel("N")
    ax.set_ylabel("Relative gap to best F₁ (%)")
    ax.set_title(f"{CITIES_TITLE.get(city, city)} — Optimality gap")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "2_optimality_gap.png", dpi=150)
    plt.close(fig)


def fig_pareto(rows: list[dict], out: Path, city: str) -> None:
    """F1 vs per-route survival probability across the R_bar sweep."""
    points = [(r["R_bar"], r["F1"], r.get("crash_mc", {}).get("mean_route_survival"))
              for r in rows if r.get("R_bar") is not None]
    points = [(rb, f1, s) for rb, f1, s in points if s is not None and rb < 1e3]
    if len(points) < 4:
        return
    survival = [s for _, _, s in points]
    f1s = [f1 for _, f1, _ in points]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(survival, f1s, alpha=0.6)
    ax.set_xlabel("Per-route survival probability (Monte Carlo)")
    ax.set_ylabel("F₁")
    ax.set_title(f"{CITIES_TITLE.get(city, city)} — Risk–F₁ Pareto")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "3_pareto_frontier.png", dpi=150)
    plt.close(fig)


def fig_ga_convergence(rows: list[dict], out: Path, city: str) -> None:
    """Best-so-far vs generation if the solver logged a convergence trace."""
    traces = []
    for r in rows:
        if r["solver"] != "ga":
            continue
        trace = r.get("solution_run_meta", {}).get("convergence")
        if trace:
            traces.append(trace)
    if not traces:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for tr in traces[:10]:
        ax.plot(tr, alpha=0.4)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best F₁ so far")
    ax.set_title(f"{CITIES_TITLE.get(city, city)} — GA convergence (first 10 seeds)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "4_ga_convergence.png", dpi=150)
    plt.close(fig)


def fig_density_box(rows: list[dict], out: Path, city: str) -> None:
    """Box plot of F1 per N, ALNS only (most stable)."""
    by_N: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["solver"] != "alns":
            continue
        by_N[r["N"]].append(r["F1"])
    if not by_N:
        return
    Ns = sorted(by_N.keys())
    data = [by_N[n] for n in Ns]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, tick_labels=[str(n) for n in Ns])
    ax.set_xlabel("N")
    ax.set_ylabel("F₁")
    ax.set_title(f"{CITIES_TITLE.get(city, city)} — F₁ distribution by N (ALNS)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "5_density_risk_box.png", dpi=150)
    plt.close(fig)


def fig_city_risk_box(rows: list[dict], out: Path, city: str) -> None:
    """Survival distribution at fixed N=50 across solvers."""
    by_solver: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["N"] != 50:
            continue
        s = r.get("crash_mc", {}).get("mean_route_survival")
        if s is not None:
            by_solver[r["solver"]].append(s)
    if not by_solver:
        return
    solvers = sorted(by_solver.keys())
    data = [by_solver[s] for s in solvers]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, tick_labels=[s.upper() for s in solvers])
    ax.set_ylabel("Per-route survival")
    ax.set_title(f"{CITIES_TITLE.get(city, city)} — Survival distribution at N=50")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "6_fleet_risk.png", dpi=150)
    plt.close(fig)


def fig_pareto_3d(rows: list[dict], out: Path, city: str) -> None:
    """3-axis scatter F1 / survival / residential edges used."""
    points = []
    for r in rows:
        s = r.get("crash_mc", {}).get("mean_route_survival")
        h = r.get("objective_breakdown", {}).get("residential_edges_used")
        if s is None or h is None:
            continue
        points.append((r["F1"], s, h))
    if len(points) < 4:
        return
    f1s, surv, h = zip(*points, strict=True)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(f1s, surv, h, alpha=0.6)
    ax.set_xlabel("F₁")
    ax.set_ylabel("Survival")
    ax.set_zlabel("Residential edges")
    ax.set_title(f"{CITIES_TITLE.get(city, city)} — 3D Pareto")
    fig.tight_layout()
    fig.savefig(out / "7_pareto_3d.png", dpi=150)
    plt.close(fig)


def fig_smax_fatigue(rows: list[dict], out: Path, city: str) -> None:
    """Behavioural Monte Carlo: per-rider survival across (alpha, beta) grid."""
    grid: dict[tuple[float, float], list[float]] = defaultdict(list)
    for r in rows:
        sweep = (r.get("behavioral_mc") or {}).get("grid")
        if not sweep:
            continue
        for cell in sweep:
            grid[(cell["alpha"], cell["beta"])].append(cell.get("rider_survival", 0.0))
    if not grid:
        return
    cells = sorted(grid.keys())
    means = [np.mean(grid[k]) for k in cells]
    labels = [f"α={a},β={b}" for a, b in cells]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(cells)), means)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean per-rider survival")
    ax.set_title(f"{CITIES_TITLE.get(city, city)} — Behavioural sensitivity")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "8_smax_fatigue.png", dpi=150)
    plt.close(fig)


def fig_stw_scatter(rows: list[dict], out: Path, city: str) -> None:
    """Lateness contribution to F1 vs total F1."""
    points = []
    for r in rows:
        bd = r.get("objective_breakdown", {})
        late = bd.get("lateness_total")
        if late is None:
            continue
        points.append((r["F1"], late))
    if len(points) < 4:
        return
    f1, late = zip(*points, strict=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(f1, late, alpha=0.6)
    ax.set_xlabel("F₁")
    ax.set_ylabel("Lateness penalty (component of F₁)")
    ax.set_title(f"{CITIES_TITLE.get(city, city)} — STW penalty contribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "9_stw_scatter.png", dpi=150)
    plt.close(fig)


FIG_FUNCS = [
    fig_scalability,
    fig_optimality_gap,
    fig_pareto,
    fig_ga_convergence,
    fig_density_box,
    fig_city_risk_box,
    fig_pareto_3d,
    fig_smax_fatigue,
    fig_stw_scatter,
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="data/results")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    rows = load_rows(Path(args.results_dir))
    print(f"Loaded {len(rows)} result rows from {args.results_dir}")
    if not rows:
        print("No result rows found. Run scripts/run_grid.py first.")
        return

    grouped = by_city(rows)
    out_root = Path(args.out_dir)
    for city, city_rows in grouped.items():
        city_out = out_root / CITIES_TITLE.get(city, city)
        city_out.mkdir(parents=True, exist_ok=True)
        emitted = 0
        for fn in FIG_FUNCS:
            try:
                before = len(list(city_out.iterdir()))
                fn(city_rows, city_out, city)
                after = len(list(city_out.iterdir()))
                if after > before:
                    emitted += 1
            except Exception as e:  # noqa: BLE001
                print(f"  {city} {fn.__name__}: skipped ({e})")
        print(f"  {city}: {emitted}/9 figures written -> {city_out}")


if __name__ == "__main__":
    main()
