"""Experiment runner tests.

We avoid triggering the OSM download / BASM calibration by supplying a
synthetic Instance directly — the runner's `instance=` parameter is the
official escape hatch documented in the module.
"""

from __future__ import annotations

import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from savrptw.runner import run_one
from tests.fixtures import build_mini_instance

REPO = Path(__file__).resolve().parents[1]
CONF = str(REPO / "conf")


def _compose(**overrides) -> object:
    with initialize_config_dir(config_dir=CONF, version_base="1.3"):
        ov = ["solver=ga", "instance.N=6", "instance.seed=11"]
        for k, v in overrides.items():
            ov.append(f"{k}={v}")
        return compose(config_name="config", overrides=ov)


def test_runner_end_to_end_with_synthetic_instance(tmp_path):
    cfg = _compose()
    cfg.paths.results = str(tmp_path)  # write into the temp dir
    # Quick GA config so the test finishes in well under a second.
    cfg.solver.pop_size = 20
    cfg.solver.generations = 30
    inst = build_mini_instance(N=6, R_bar=0.5)
    row = run_one(cfg, instance=inst)

    assert row["feasible"] is True
    assert row["solver"] == "ga"
    assert row["F1"] >= 0.0
    assert row["n_routes"] >= 1
    # crash + behavioural sections populated.
    assert "crash_mc" in row and row["crash_mc"]["mode"] == "analytic"
    assert "behavioral_mc" in row and len(row["behavioral_mc"]["points"]) == 9

    # File emitted.
    out = Path(row["output_path"])
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["run_id"] == row["run_id"]


def test_runner_with_alns(tmp_path):
    cfg = _compose(solver="alns")
    cfg.paths.results = str(tmp_path)
    cfg.solver.iterations = 200
    inst = build_mini_instance(N=6)
    row = run_one(cfg, instance=inst)
    assert row["feasible"] is True
    assert row["solver"] == "alns"


def test_runner_propagates_infeasibility(tmp_path):
    cfg = _compose()
    cfg.paths.results = str(tmp_path)
    cfg.solver.pop_size = 10
    cfg.solver.generations = 5
    # Impossibly tight ε constraint.
    inst = build_mini_instance(N=4, R_bar=0.001)
    import pytest

    from savrptw.solvers.base import InfeasibleError

    with pytest.raises(InfeasibleError):
        run_one(cfg, instance=inst)
