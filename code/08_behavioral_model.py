import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from utils import DATA_DIR, load_graph

def run_behavioral_simulation():
    # 1. Load Data
    try:
        with open(DATA_DIR / '04_vrptw_instance.json', 'r') as f:
            instance = json.load(f)
        with open(DATA_DIR / '05_solution.json', 'r') as f:
            solution = json.load(f)
        G = load_graph(DATA_DIR / '03_graph_final.graphml')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Extract nodes
    depots = {d['node_id']: d for d in instance['depots']}
    customers = {c['node_id']: c for c in instance['customers']}
    
    # 3. Behavioral Simulation (Rider choice modeling)
    # Riders often take shortcuts or prioritize speed over safety if the incentive is high.
    # We simulate a "Compliance Rate": Higher compliance means staying on the suggested safe path.
    
    results = []
    
    print(f"Propagating Behavioral Profiles across {len(solution['routes'])} routes...")
    
    for route_idx, route in enumerate(solution['routes']):
        route_nodes = route['nodes']
        best_score = 0
        
        # Simulate 10 different riders for this route
        for rider_id in range(10):
            # Compliance Factor (0.0 to 1.0)
            # Low compliance = speed seeker (ignores risk weights)
            # High compliance = safety seeker (follows suggested path)
            compliance = np.random.beta(5, 2) # Biased towards higher compliance
            
            total_tt = 0
            total_risk = 0
            total_cong = 0
            
            for i in range(len(route_nodes) - 1):
                u, v = route_nodes[i], route_nodes[i+1]
                
                # Fetch edge data (realized costs based on compliance)
                try:
                    # In a real model, we'd search for alternative paths between u and v
                    # Here we simulate the effect of compliance on the outcome
                    # If compliance is low, they might go faster but double the risk
                    
                    # Assuming we follow the path but the *perceived* costs vary
                    path_tt = nx.shortest_path_length(G, u, v, weight='t_ij')
                    path_risk = nx.shortest_path_length(G, u, v, weight='r_ij')
                    path_cong = nx.shortest_path_length(G, u, v, weight='c_ij')
                    
                    # Realized cost = (Compliance * Suggested) + ((1-Compliance) * Risky Shortcut)
                    realized_tt = path_tt * (0.8 + 0.4 * (1-compliance)) # May be faster or slower
                    realized_risk = path_risk * (1.0 + 2.0 * (1-compliance)) # Risk spikes if non-compliant
                    
                    total_tt += realized_tt
                    total_risk += realized_risk
                    total_cong += path_cong
                except:
                    # Fallback if path not found
                    total_tt += 10
                    total_risk += 5
            
            results.append({
                "route_id": route_idx,
                "rider_id": rider_id,
                "compliance": round(compliance, 2),
                "total_tt": round(total_tt, 2),
                "total_risk": round(total_risk, 2),
                "total_cong": round(total_cong, 2)
            })

    # 4. Save analysis
    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / '08_behavioral_analysis.csv', index=False)
    
    # Generate summary stats
    summary = df.groupby('route_id').agg({
        'total_tt': ['mean', 'std'],
        'total_risk': ['mean', 'std'],
        'compliance': 'mean'
    })
    print("\nBehavioral Simulation Summary (per route):")
    print(summary.head())
    
    print(f"\n[08] Done. Behavioral analysis saved to {DATA_DIR / '08_behavioral_analysis.csv'}")

if __name__ == "__main__":
    run_behavioral_simulation()
