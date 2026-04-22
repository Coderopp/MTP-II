"""Hydra entry point for a single experiment row.

Examples
--------

    # full pipeline (requires OSMnx + calibrated r_ij):
    python scripts/run_experiment.py city=bengaluru solver=ga instance.N=50

    # override multiple groups:
    python scripts/run_experiment.py city=mumbai solver=alns \\
         instance.N=100 instance.seed=303 instance.R_bar=0.8
"""

from __future__ import annotations

import json
import sys

import hydra
from omegaconf import DictConfig

from savrptw.runner import run_one


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    result = run_one(cfg)
    print(json.dumps({k: v for k, v in result.items() if k not in {"config"}}, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
