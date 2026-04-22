# SA-VRPTW — Safety-Aware Vehicle Routing Problem with Time Windows

Reference implementation accompanying the Wiley submission on safety-aware routing for Indian quick-commerce.

## Status

Refactor in progress on branch `refactor/modularize`. The legacy pipeline lives under `code/` and is retained for cross-validation. The new canonical implementation lives under `src/savrptw/`.

Authoritative mathematical formulation: [`docs/FORMULATION.md`](docs/FORMULATION.md).

## Quickstart (valid once scaffolding is populated)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Layout

```
savrptw/
├── docs/                       # FORMULATION.md and methodology notes
├── src/savrptw/                # library code
├── conf/                       # Hydra configs (cities, solvers, experiments)
├── scripts/                    # thin CLI entry points
├── tools/                      # maintenance scripts (data refresh etc.)
├── tests/                      # pytest
├── data/
│   ├── raw/                    # committed inputs (e.g. KML snapshot)
│   ├── processed/              # derived city graphs (gitignored)
│   ├── results/                # experiment outputs (gitignored)
│   └── validation/             # cross-validation reports
└── code/                       # LEGACY pipeline, kept for cross-checking only
```

## License

MIT — see `LICENSE`.
