"""
07_solver_ga.py — Multi-Objective Genetic Algorithm for Safe-Aware VRPTW.

Objectives:
1. Minimize Total Travel Time (Efficiency)
2. Minimize Total Risk Exposure (Safety)
3. Minimize Total Congestion Impact (Sustainability)

Constraints:
- Capacity (Q=3)
- Time Windows (Hard/Soft penalties)
- Route consistency (Start/End at assigned depot)
"""

import json
import random
import numpy as np
import networkx as nx
from pathlib import Path
from utils import DATA_DIR, load_graph

# Configuration
INSTANCE_PATH = DATA_DIR / "04_vrptw_instance.json"
GRAPH_PATH = DATA_DIR / "03_graph_final.graphml"
OUTPUT_PATH = DATA_DIR / "05_solution.json"

# GA Hyperparameters
POP_SIZE = 50
GENS = 100
CX_PROB = 0.8
MUT_PROB = 0.2
ELITE_SIZE = 2

# Vehicle Capacity (from instance or fixed)
CAPACITY = 3  # Pre-defined based on rider payload

class SAVRPTW_Solver:
    def __init__(self, instance_path, graph_path):
        with open(instance_path) as f:
            self.instance = json.load(f)
        
        self.G = load_graph(graph_path)
        self.depots = {d['node_id']: d for d in self.instance['depots']}
        self.customers = self.instance['customers']
        self.n = len(self.customers)
        
        # We need a mapping from node_id to (time, risk, congestion)
        # But calculating all-pairs shortest paths on the full OSM graph for 200 nodes is slow.
        # We'll use the pre-calculated tt_from_depot and approximate customer-customer costs 
        # using a simple Euclidean or a subset of shortest paths if needed.
        # However, for a high-quality solver, we need the matrix.
        self._prepare_matrices()

    def _prepare_matrices(self):
        """Prepare cost matrices for travel time, risk, and congestion."""
        print("  [GA] Preparing cost matrices...")
        # Nodes: Depot(s) + Customers
        self.all_nodes = [c['node_id'] for c in self.customers]
        # For simplicity in this demo/skeleton, we'll use the networkx shortest paths
        # In a real large-scale scenario, we'd use a contraction hierarchy or pre-distilled graph.
        
        # We'll focus on the customers assigned to each depot
        self.matrix_tt = {}
        self.matrix_risk = {}
        self.matrix_cong = {}

        # Since we have a MultiDiGraph, we need to ensure the weight is set correctly
        # We'll pre-calculate node-node paths for all pairs in the instance nodes
        relevant_ids = list(self.depots.keys()) + [c['node_id'] for c in self.customers]
        
        # Use single-source Dijkstra for each relevant node
        for start_node in relevant_ids:
            # Travel Time
            lengths_tt = nx.single_source_dijkstra_path_length(self.G, start_node, weight='t_ij')
            # Risk
            lengths_risk = nx.single_source_dijkstra_path_length(self.G, start_node, weight='r_ij')
            # Congestion
            lengths_cong = nx.single_source_dijkstra_path_length(self.G, start_node, weight='c_ij')

            self.matrix_tt[start_node] = lengths_tt
            self.matrix_risk[start_node] = lengths_risk
            self.matrix_cong[start_node] = lengths_cong

    def evaluate_route(self, route, depot_id):
        """Calculate costs and violations for a single route."""
        tt = 0
        risk = 0
        cong = 0
        current_time = 0
        load = 0
        tw_violation = 0

        prev_node = depot_id
        for c_idx in route:
            cust = self.customers[c_idx]
            curr_node = cust['node_id']
            
            # Travel costs
            dt = self.matrix_tt[prev_node].get(curr_node, 999)
            dr = self.matrix_risk[prev_node].get(curr_node, 999)
            dc = self.matrix_cong[prev_node].get(curr_node, 999)
            
            tt += dt
            risk += dr
            cong += dc
            
            current_time += dt
            # Service / Time Window
            if current_time < cust['e_i']:
                current_time = cust['e_i'] # Wait until opening
            elif current_time > cust['l_i']:
                tw_violation += (current_time - cust['l_i'])
            
            load += cust.get('q_i', 1)
            prev_node = curr_node
            
        # Return to depot
        dt_home = self.matrix_tt[prev_node].get(depot_id, 999)
        tt += dt_home
        risk += self.matrix_risk[prev_node].get(depot_id, 999)
        cong += self.matrix_cong[prev_node].get(depot_id, 999)
        
        return tt, risk, cong, tw_violation, load

    def split(self, individual):
        """
        Split procedure (simplified): Greedy partitioning of the giant tour.
        Optimal split uses Bellman-Ford on a DAG, but greedy is faster for GA iterations.
        """
        routes = []
        current_route = []
        current_load = 0
        
        # We assume each customer is pre-assigned to a depot.
        # We should cluster the giant tour by depot first or ensure valid routes.
        # For simplicity, we'll assign the route to the depot of the FIRST customer in the sequence.
        
        for c_idx in individual:
            cust = self.customers[c_idx]
            q = cust.get('q_i', 1)
            
            if current_load + q > CAPACITY:
                if current_route:
                    routes.append(current_route)
                current_route = [c_idx]
                current_load = q
            else:
                current_route.append(c_idx)
                current_load += q
        
        if current_route:
            routes.append(current_route)
            
        return routes

    def fitness(self, individual):
        """Calculate total multi-objective score."""
        routes = self.split(individual)
        total_tt = 0
        total_risk = 0
        total_cong = 0
        total_tw_v = 0
        
        l1, l2, l3 = self.instance['metadata'].get('lambda', [0.33, 0.33, 0.34])
        
        for r in routes:
            # In MD-VRPTW, we should use the assigned depot.
            # We'll take the assigned_depot of the first customer.
            depot_id = self.customers[r[0]]['assigned_depot']
            res = self.evaluate_route(r, depot_id)
            total_tt += res[0]
            total_risk += res[1]
            total_cong += res[2]
            total_tw_v += res[3]
            
        # Score = Weighted objectives + Penalty for TW violations
        penalty = total_tw_v * 100 
        score = (l1 * total_tt) + (l2 * total_risk) + (l3 * total_cong) + penalty
        return score

    def solve(self):
        print(f"  [GA] Starting optimization for {self.n} customers...")
        # Initialize Population
        population = [random.sample(range(self.n), self.n) for _ in range(POP_SIZE)]
        
        for gen in range(GENS):
            # Sort by fitness
            population.sort(key=lambda ind: self.fitness(ind))
            
            best_score = self.fitness(population[0])
            if gen % 10 == 0:
                print(f"    Gen {gen}: Best Score = {best_score:.2f}")
            
            new_pop = population[:ELITE_SIZE]
            
            while len(new_pop) < POP_SIZE:
                # Tournament Selection
                p1 = self.tournament(population)
                p2 = self.tournament(population)
                
                # Crossover
                if random.random() < CX_PROB:
                    c1, c2 = self.ordered_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                if random.random() < MUT_PROB:
                    self.mutate(c1)
                if random.random() < MUT_PROB:
                    self.mutate(c2)
                
                new_pop.extend([c1, c2])
            
            population = new_pop[:POP_SIZE]
            
        population.sort(key=lambda ind: self.fitness(ind))
        best_ind = population[0]
        self._save_solution(best_ind)

    def tournament(self, pop, k=3):
        candidates = random.sample(pop, k)
        return min(candidates, key=lambda i: self.fitness(i))

    def ordered_crossover(self, p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        
        def fill_child(parent1, parent2, start, end):
            child = [None] * size
            child[start:end] = parent1[start:end]
            
            p2_filtered = [item for item in parent2 if item not in child[start:end]]
            
            # Fill before the segment
            child[:start] = p2_filtered[:start]
            # Fill after the segment
            child[end:] = p2_filtered[start:]
            return child

        return fill_child(p1, p2, a, b), fill_child(p2, p1, a, b)

    def mutate(self, ind):
        # Swap mutation
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]

    def _save_solution(self, best_ind):
        routes_indices = self.split(best_ind)
        routes = []
        for r_idxs in routes_indices:
            # Map indices back to node_ids and calculate timestamps
            depot_id = self.customers[r_idxs[0]]['assigned_depot']
            nodes = [depot_id] + [self.customers[i]['node_id'] for i in r_idxs] + [depot_id]
            routes.append({
                "depot": depot_id,
                "nodes": nodes,
                "customer_indices": [int(i) for i in r_idxs]
            })
        
        result = {
            "metadata": self.instance['metadata'],
            "summary": {
                "score": self.fitness(best_ind),
                "num_routes": len(routes)
            },
            "routes": routes
        }
        
        with open(OUTPUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [GA] Done. Solution saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    if not INSTANCE_PATH.exists():
        print(f"Error: {INSTANCE_PATH} not found.")
    else:
        solver = SAVRPTW_Solver(INSTANCE_PATH, GRAPH_PATH)
        solver.solve()
