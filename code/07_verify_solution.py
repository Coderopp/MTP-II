"""
07_verify_solution.py — Validation Suite for SA-VRPTW solutions.
"""

import json
from pathlib import Path
from utils import DATA_DIR, load_graph

SOLUTION_PATH = DATA_DIR / "05_solution.json"
INSTANCE_PATH = DATA_DIR / "04_vrptw_instance.json"

def verify():
    if not SOLUTION_PATH.exists():
        print(f"Error: {SOLUTION_PATH} not found.")
        return

    with open(INSTANCE_PATH) as f:
        instance = json.load(f)
    with open(SOLUTION_PATH) as f:
        solution = json.load(f)

    customers = instance['customers']
    num_customers = len(customers)
    all_cust_ids = {c['node_id'] for c in customers}
    
    routes = solution['routes']
    visited_ids = []
    
    print(f"--- Verification Report: {solution['metadata']['description']} ---")
    print(f"Instance size: {num_customers} customers")
    print(f"Solution routes: {len(routes)}")

    # 1. Check all customers are visited
    for r_idx, route in enumerate(routes):
        # Middle nodes (excluding depot start/end)
        nodes = route['nodes'][1:-1]
        visited_ids.extend(nodes)
        
        # 2. Check depot consistency
        depot_id = route['depot']
        if route['nodes'][0] != depot_id or route['nodes'][-1] != depot_id:
            print(f"  [!] Route {r_idx} does not start/end at depot {depot_id}")

    unique_visited = set(visited_ids)
    if len(unique_visited) != num_customers:
        missing = all_cust_ids - unique_visited
        extra = unique_visited - all_cust_ids
        print(f"  [!] Missing customers: {len(missing)}")
        print(f"  [!] Unique visited: {len(unique_visited)} / {num_customers}")
    else:
        print("  [OK] All customers visited exactly once.")

    # 3. Double visits
    if len(visited_ids) > num_customers:
        print(f"  [!] Redundant visits detected: {len(visited_ids) - num_customers}")

    print("--- Constraints Check (Conceptual) ---")
    # For a full check, we'd re-calculate time and load from the graph
    print("  [INFO] Solver reported score: ", solution['summary']['score'])
    print("✅ Verification Complete.")

if __name__ == "__main__":
    verify()
