"""
solver_milp.py — Mixed-Integer Linear Programming Exact Solver

Formulates the multi-objective SA-VRPTW balancing the time penalty (real arrival vs. estimated arrival)
against the probabilistic route survival, including capacity (Q) constraints and soft time window (STW) penalties.
"""

import pulp
import networkx as nx
import numpy as np

def solve_milp(instance: dict, G: nx.MultiDiGraph):
    """
    Solve the SA-VRPTW using PuLP exact solver.
    """
    # Create the LP problem
    prob = pulp.LpProblem("SA_VRPTW_MultiObjective", pulp.LpMinimize)

    # 1. Sets and Parameters
    customers = instance["customers"]
    depots = instance["depots"]
    nodes = depots + customers
    
    V = [n["node_id"] for n in nodes]
    C = [c["node_id"] for c in customers]
    D = [d["node_id"] for d in depots]
    
    K = instance["parameters"]["K"]  # number of riders
    Q = instance["parameters"]["Q"]  # vehicle capacity
    
    l1 = instance["parameters"].get("lambda1", 0.4)
    l2 = instance["parameters"].get("lambda2", 0.4)
    l3 = instance["parameters"].get("lambda3", 0.2)

    # 2. Extract edge attributes from the graph
    # Helper to get edge attrs mapping directly to the primary objective
    def get_attr(u, v, attr):
        if not G.has_edge(u, v): return 1000.0 # Large penalty if no path
        # Select best parallel edge by travel time (t_ij)
        return min(G[u][v].values(), key=lambda e: e.get('t_ij', float('inf'))).get(attr, 1000.0)

    # Precompute travel time and risk matrices for all node pairs
    t_ij = { (u, v): get_attr(u, v, 't_ij') for u in V for v in V if u != v }
    r_ij = { (u, v): get_attr(u, v, 'r_ij') for u in V for v in V if u != v }
    
    # 3. Decision Variables
    # x[i,j,k] = 1 if rider k travels from node i to j
    x = pulp.LpVariable.dicts("x", ((i, j, k) for i in V for j in V if i != j for k in range(K)), cat='Binary')
    
    # t[i,k] = Arrival time of rider k at node i
    t = pulp.LpVariable.dicts("t", ((i, k) for i in V for k in range(K)), lowBound=0, cat='Continuous')
    
    # q[i,k] = Remaining capacity of rider k after serving node i
    q = pulp.LpVariable.dicts("q", ((i, k) for i in V for k in range(K)), lowBound=0, upBound=Q, cat='Continuous')
    
    # STW Penalty Variables
    # e_penalty[i] = Earliness penalty
    # l_penalty[i] = Lateness penalty
    e_pen = pulp.LpVariable.dicts("e_pen", C, lowBound=0, cat='Continuous')
    l_pen = pulp.LpVariable.dicts("l_pen", C, lowBound=0, cat='Continuous')

    # 4. Multi-Objective Function
    # Obj 1: Total Route Travel Time (Efficiency)
    obj_time = pulp.lpSum(t_ij.get((i,j), 1000) * x[i,j,k] for i in V for j in V if i != j for k in range(K))
    
    # Obj 2: Route Survival / Risk Exposure (Safety)
    obj_risk = pulp.lpSum(r_ij.get((i,j), 1.0) * x[i,j,k] for i in V for j in V if i != j for k in range(K))
    
    # Obj 3: Soft Time Window Penalties
    obj_stw = pulp.lpSum(e_pen[i] + l_pen[i] for i in C)

    # Combined Objective
    prob += l1 * obj_time + l2 * obj_risk + l3 * obj_stw

    # 5. Constraints
    # Constraint: Every customer must be visited exactly once
    for i in C:
        prob += pulp.lpSum(x[i, j, k] for j in V if i != j for k in range(K)) == 1
        
    # Constraint: Flow conservation for vehicles
    for k in range(K):
        for j in C:
            prob += pulp.lpSum(x[i, j, k] for i in V if i != j) == pulp.lpSum(x[j, h, k] for h in V if h != j)
            
    # Constraint: Capacity and Time tracking (MTZ subtour elimination logic)
    M_t = 10000  # Large constant for time
    for i in V:
        for j in C:
            if i != j:
                for k in range(K):
                    # Arrival time
                    prob += t[j,k] >= t[i,k] + t_ij.get((i,j), 0) - M_t * (1 - x[i,j,k])
                    
                    # Capacity
                    demand_j = next((cust['q_i'] for cust in customers if cust['node_id'] == j), 1)
                    prob += q[j,k] <= q[i,k] - demand_j + Q * (1 - x[i,j,k])

    # Constraint: Soft Time Windows logic
    for i in C:
        customer_info = next(cust for cust in customers if cust['node_id'] == i)
        e_i = customer_info['e_i']
        l_i = customer_info['l_i']
        
        # We define arrival time t_arr as sum(t[i,k] * x_used)
        t_arr = pulp.lpSum(t[i,k] for k in range(K)) # approximation for STW constraints
        
        prob += e_pen[i] >= e_i - t_arr
        prob += l_pen[i] >= t_arr - l_i
        
    # 6. Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))
    
    status = pulp.LpStatus[prob.status]
    objective_val = pulp.value(prob.objective) if status == "Optimal" else float('inf')
    
    return {
        "status": status,
        "objective": objective_val,
        "method": "MILP"
    }
