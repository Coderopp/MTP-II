"""
run_pipeline.py — Master runner for the SA-VRPTW data acquisition pipeline.

Usage
-----
  python run_pipeline.py                  # Run all steps (1-5)
  python run_pipeline.py --steps 1 2 3   # Run specific steps only
  python run_pipeline.py --congestion GOOGLE_MAPS  # Override mode

Steps
-----
  1  → 01_osm_graph.py          Download and build road graph
  2  → 02_irad_risk.py          Compute collision risk r_ij
  3  → 03_congestion.py         Compute congestion index c_ij
  4  → 04_instance_generator.py Generate VRPTW problem instance
  5  → 05_validate_instance.py  Validate all outputs
"""

import argparse
import importlib
import importlib.util
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

STEP_MAP = {
    1: "01_osm_graph",
    2: "02_irad_risk",
    3: "03_congestion",
    4: "04_instance_generator",
    5: "05_validate_instance",
}

STEP_LABELS = {
    1: "OSM Road Network",
    2: "iRAD Risk Scores (r_ij)",
    3: "Congestion Index (c_ij)",
    4: "VRPTW Instance Generation",
    5: "Instance Validation",
}


def run_step(step_num: int) -> bool:
    label = STEP_LABELS[step_num]
    module_name = STEP_MAP[step_num]
    bar = "─" * 55
    print(f"\n{bar}")
    print(f"  STEP {step_num}: {label}")
    print(f"{bar}")

    t0 = time.time()
    try:
        # Dynamically import and run the module's main()
        sys.path.insert(0, str(SCRIPT_DIR))
        spec = importlib.util.spec_from_file_location(
            module_name, SCRIPT_DIR / f"{module_name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        elapsed = time.time() - t0
        print(f"\n  ✅ Step {step_num} completed in {elapsed:.1f}s")
        return True
    except SystemExit as e:
        # validate script uses sys.exit — 0 = pass, 1 = fail
        if e.code == 0:
            elapsed = time.time() - t0
            print(f"\n  ✅ Step {step_num} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"\n  ❌ Step {step_num} FAILED (validation errors found)")
            return False
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌ Step {step_num} FAILED after {elapsed:.1f}s: {e}")
        import traceback; traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="SA-VRPTW data pipeline runner")
    parser.add_argument(
        "--steps", nargs="+", type=int,
        default=list(STEP_MAP.keys()),
        help="Steps to run (default: all 1-5)")
    parser.add_argument(
        "--congestion", choices=["SPEED_PROXY", "GOOGLE_MAPS"],
        default=None,
        help="Override congestion mode for Step 3")
    args = parser.parse_args()

    if args.congestion:
        os.environ["CONGESTION_MODE"] = args.congestion
        print(f"  CONGESTION_MODE overridden to: {args.congestion}")

    steps = sorted(args.steps)
    print(f"\n{'═' * 55}")
    print(f"  SA-VRPTW Data Acquisition Pipeline")
    print(f"  Running steps: {steps}")
    print(f"{'═' * 55}")

    t_start = time.time()
    results = {}
    for step in steps:
        if step not in STEP_MAP:
            print(f"  Unknown step: {step} — skipping")
            continue
        results[step] = run_step(step)
        if not results[step] and step < 5:
            print(f"\n  Pipeline halted at step {step}. Fix errors before continuing.")
            break

    total = time.time() - t_start
    print(f"\n{'═' * 55}")
    print(f"  Pipeline Summary  ({total:.1f}s total)")
    print(f"{'═' * 55}")
    for s, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  Step {s}: {STEP_LABELS[s]}")

    if all(results.values()):
        print(f"\n  🎉 All steps completed. Data is ready for the solver.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
