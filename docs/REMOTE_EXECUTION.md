# Remote Execution on iDRAC Compute

This note records the exact steps to run the full SA-VRPTW
experimental grid on the remote Dell iDRAC server, then pull the
results back for figure regeneration and the LaTeX paper.

## Prerequisites on the remote server

Python 3.11+ and the project dependencies. From the repo root on
the remote node:

```
git clone https://github.com/Coderopp/MTP-II.git mtp-final
cd mtp-final
python3 -m venv .venv
source .venv/bin/activate
pip install -e .          # all deps are in pyproject.toml
```

The OSMnx cache directory must be writable; first-time cold fetch of
each city's bbox graph takes 5--10 minutes. The repo ships an empty
`cache/` that the runner fills.

## Smoke test (one row)

Before launching the full grid, verify a single row completes:

```
PYTHONPATH=src python3 scripts/run_experiment.py \
  city=bengaluru solver=ga instance.N=20 instance.seed=42 \
  instance.n_depots=3 instance.R_bar=10000.0 instance.H_bar=1000000
```

Expected wall-clock on a modern server with a cold OSMnx cache:
roughly 6--7 minutes (mostly the OSM bbox download). Subsequent rows
in the same city use the cached graph and run in seconds for GA/ALNS,
up to 120 s for MILP.

## Full grid

```
PYTHONPATH=src nohup python3 scripts/run_grid.py \
  --n-seeds 10 \
  --solvers ga,alns,milp \
  --milp-max-N 50 \
  > data/results/_grid.log 2>&1 &
```

Grid size: 5 cities $\times$ 4 sizes ($N \in \{20, 50, 100, 200\}$)
$\times$ 10 seeds $\times$ 3 solvers, with MILP capped at $N \leq 50$.
Total = $5 \times (3 + 3 + 2 + 2) \times 10 = 500$ rows. Per-row
results land in `data/results/<run_id>.json`. The driver writes a
manifest (`data/results/_manifest.jsonl`) and resumes from
checkpoints if it crashes.

Monitor:

```
tail -f data/results/_grid.log
wc -l data/results/_manifest.jsonl
```

## Optional: Pareto sweep

After the default grid completes, regenerate the Pareto frontier at
$N = 50$ across the $\bar{R} \times \bar{H}$ grid:

```
PYTHONPATH=src python3 scripts/run_grid.py \
  --n-seeds 3 --solvers ga,alns --milp-max-N 0 --pareto
```

This adds $5 \times 7 \times 6 \times 3 \times 2 = 1260$ ε-sweep rows.

## Pulling results back

```
rsync -avz remote:mtp-final/data/results/ ./data/results/
```

The figure-generation script (Task #14) will consume
`data/results/*.json` and write to `figures/`.

## Updating the paper

After figures are regenerated, replace the placeholder framed boxes
in `ACCESS_latex_template_20240429/access.tex` with
`\includegraphics{figures/...}` and grep for `\TBD` to find every
empirical hole that still needs a number filled in.

```
grep -n '\\TBD' ACCESS_latex_template_20240429/access.tex
```
