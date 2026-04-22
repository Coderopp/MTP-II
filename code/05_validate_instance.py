"""
05_validate_instance.py — Step 5: Sanity-check the full pipeline output.

Runs a suite of checks on the final graph and VRPTW instance before the
solver ingests them. All FAIL items must be resolved before proceeding to
Phase 3.

Checks
------
 1. Output files exist
 2. All edges have finite t_ij
 3. r_ij ∈ [0, 1] for all edges
 4. c_ij ∈ [0, 1] for all edges
 5. All customer nodes exist in the graph
 6. All customers are reachable from depot
 7. Time windows are feasible (tt_from_depot ≤ l_i)
 8. Total customer demand ≤ K * Q
 9. Lambda weights sum to 1.0
10. Prints a summary statistics table
"""

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import DATA_DIR, load_graph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRAPH_PATH    = DATA_DIR / "03_graph_final.graphml"
INSTANCE_PATH = DATA_DIR / "04_vrptw_instance.json"

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def check(label: str, passed: bool, detail: str = "") -> bool:
    status = PASS if passed else FAIL
    msg = f"  {status}  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return passed


def check_warn(label: str, cond: bool, detail: str = "") -> None:
    status = PASS if cond else WARN
    msg = f"  {status}  {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main() -> None:
    all_passed = True
    print("=" * 60)
    print("  SA-VRPTW Instance Validation")
    print("=" * 60)

    # --- Check 1: Files exist -----------------------------------------------
    print("\n[FILES]")
    g_ok = check("Graph file exists", GRAPH_PATH.exists(), str(GRAPH_PATH.name))
    i_ok = check("Instance file exists", INSTANCE_PATH.exists(), str(INSTANCE_PATH.name))
    all_passed &= g_ok and i_ok

    if not (g_ok and i_ok):
        print("\n  Cannot continue — run steps 01–04 first.")
        sys.exit(1)

    # --- Load ---------------------------------------------------------------
    print("\n[LOADING]")
    G = load_graph(GRAPH_PATH)
    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    with open(INSTANCE_PATH) as f:
        inst = json.load(f)
    depots = inst["depots"]
    customers = inst["customers"]
    params = inst["parameters"]

    # Map depots for easy lookup
    depot_map = {str(d["node_id"]): d for d in depots}
    print(f"  Instance: {len(customers)} customers, {len(depots)} depots, "
          f"K={params['K']}, Q={params['Q']}")

    # --- Check 2: t_ij finite on all edges ----------------------------------
    print("\n[EDGE ATTRIBUTES: t_ij]")
    bad_tij = [(u, v) for u, v, d in G.edges(data=True)
               if not (0.0 < d.get("t_ij", -1.0) < 1e6)]
    ok = check("All edges have finite t_ij > 0",
               len(bad_tij) == 0,
               f"{len(bad_tij)} bad edges" if bad_tij else "all OK")
    all_passed &= ok

    # --- Check 3: r_ij ∈ [0,1] ---------------------------------------------
    print("\n[EDGE ATTRIBUTES: r_ij]")
    bad_rij = [(u, v) for u, v, d in G.edges(data=True)
               if not (0.0 <= d.get("r_ij", -1.0) <= 1.0)]
    ok = check("r_ij ∈ [0,1] for all edges",
               len(bad_rij) == 0,
               f"{len(bad_rij)} out-of-range" if bad_rij else "all OK")
    all_passed &= ok

    # Distribution of r_ij
    rij_vals = [d.get("r_ij", 0.0) for _, _, d in G.edges(data=True)]
    n_risky = sum(1 for r in rij_vals if r > 0.0)
    check_warn(f"Edges with r_ij > 0",
               n_risky > 0,
               f"{n_risky} / {len(rij_vals)}")

    # --- Check 4: c_ij ∈ [0,1] ---------------------------------------------
    print("\n[EDGE ATTRIBUTES: c_ij]")
    bad_cij = [(u, v) for u, v, d in G.edges(data=True)
               if not (0.0 <= d.get("c_ij", -1.0) <= 1.0)]
    ok = check("c_ij ∈ [0,1] for all edges",
               len(bad_cij) == 0,
               f"{len(bad_cij)} out-of-range" if bad_cij else "all OK")
    all_passed &= ok

    # --- Check 5: Customer nodes in graph -----------------------------------
    print("\n[CUSTOMER NODES]")
    graph_nodes = set(G.nodes())
    missing_nodes = [c["node_id"] for c in customers if str(c["node_id"]) not in graph_nodes]
    ok = check("All customer nodes exist in graph",
               len(missing_nodes) == 0,
               f"missing: {missing_nodes}" if missing_nodes else "all present")
    all_passed &= ok

    for d_id in depot_map.keys():
        ok = check(f"Depot {d_id} exists in graph",
                   d_id in graph_nodes)
        all_passed &= ok

    # --- Check 6: Reachability from depot -----------------------------------
    print("\n[REACHABILITY]")
    unreachable = []
    for c in customers:
        cid = str(c["node_id"])
        did = str(c["assigned_depot"])
        if cid not in graph_nodes or did not in graph_nodes:
            continue   # already caught in check 5
        if not nx.has_path(G, did, cid):
            unreachable.append(f"{cid} from {did}")
    ok = check("All customers reachable from assigned depot",
               len(unreachable) == 0,
               f"unreachable: {unreachable}" if unreachable else "all reachable")
    all_passed &= ok

    # --- Check 7: Time windows feasible -------------------------------------
    print("\n[TIME WINDOWS]")
    infeasible_tw = []
    for c in customers:
        if c.get("tt_from_depot") is not None and c["tt_from_depot"] > c["l_i"]:
            infeasible_tw.append((c["node_id"], c["tt_from_depot"], c["l_i"]))
    ok = check("Feasible time windows (tt_from_depot ≤ l_i)",
               len(infeasible_tw) == 0,
               f"{len(infeasible_tw)} infeasible" if infeasible_tw else "all feasible")
    all_passed &= ok

    # --- Check 8: Demand feasibility ----------------------------------------
    print("\n[CAPACITY]")
    total_demand = sum(c["q_i"] for c in customers)
    max_load = params["K"] * params["Q"]
    ok = check("Total demand ≤ K × Q",
               total_demand <= max_load,
               f"demand={total_demand}, max={max_load}")
    all_passed &= ok

    # --- Check 9: Lambda weights sum to 1.0 ---------------------------------
    print("\n[OBJECTIVE WEIGHTS]")
    lam_sum = params["lambda1"] + params["lambda2"] + params["lambda3"]
    ok = check("λ₁ + λ₂ + λ₃ = 1.0",
               abs(lam_sum - 1.0) < 1e-6,
               f"sum={lam_sum:.4f}")
    all_passed &= ok

    # --- Summary statistics table -------------------------------------------
    print("\n[SUMMARY STATISTICS]")
    tij  = [d.get("t_ij",  0.0) for _, _, d in G.edges(data=True)]
    rij  = [d.get("r_ij",  0.0) for _, _, d in G.edges(data=True)]
    cij  = [d.get("c_ij",  1.0) for _, _, d in G.edges(data=True)]
    ei   = [c["e_i"]  for c in customers]
    li   = [c["l_i"]  for c in customers]
    qi   = [c["q_i"]  for c in customers]
    tts  = [c.get("tt_from_depot", 0) for c in customers]

    rows = [
        ["t_ij (min)",        f"{min(tij):.3f}", f"{max(tij):.3f}",
         f"{sum(tij)/len(tij):.3f}"],
        ["r_ij",              f"{min(rij):.3f}", f"{max(rij):.3f}",
         f"{sum(rij)/len(rij):.3f}"],
        ["c_ij",              f"{min(cij):.3f}", f"{max(cij):.3f}",
         f"{sum(cij)/len(cij):.3f}"],
        ["e_i (open, min)",   f"{min(ei)}",      f"{max(ei)}",
         f"{sum(ei)/len(ei):.1f}"],
        ["l_i (close, min)",  f"{min(li)}",      f"{max(li)}",
         f"{sum(li)/len(li):.1f}"],
        ["q_i (demand, units)",f"{min(qi)}",     f"{max(qi)}",
         f"{sum(qi)/len(qi):.1f}"],
        ["tt_from_depot (min)",f"{min(tts):.2f}",f"{max(tts):.2f}",
         f"{sum(tts)/len(tts):.2f}"],
    ]
    header = f"  {'Attribute':<26} {'Min':>8} {'Max':>8} {'Mean':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(f"  {r[0]:<26} {r[1]:>8} {r[2]:>8} {r[3]:>8}")

    # --- Final verdict -------------------------------------------------------
    print("\n" + "=" * 60)
    if all_passed:
        print(f"  {PASS}  All checks passed. Instance ready for solver.")
    else:
        print(f"  {FAIL}  Some checks failed. Review output above.")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
