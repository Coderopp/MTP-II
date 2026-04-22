import math
import pulp
import networkx as nx

class MILPSolver:
    def __init__(self, instance_data, graph):
        self.instance = instance_data
        self.G = graph
        self.prob = None

    def solve(self, R_bar=float('inf'), H_bar=float('inf')):
        # Extract metadata
        customers = self.instance["customers"]
        depots = self.instance["depots"]

        V_c = [c["node_id"] for c in customers]
        V_d = [d["node_id"] for d in depots]
        V = V_d + V_c

        # Build K list mapping to specific depots
        K = []
        for d in depots:
            num_k = d.get('K_d', 0)
            depot_id = d["node_id"]
            for i in range(num_k):
                K.append({"id": f"k_{depot_id}_{i}", "depot": depot_id})

        Q = self.instance["parameters"]["Q"]
        T_max = 35.0

        # Build node dictionaries
        node_data = {}
        for d in depots:
            node_data[d["node_id"]] = {
                "e": d["e_0"], "l": d["l_0"], "eta": 0, "s": 0, "q": 0, "type": "depot"
            }
        for c in customers:
            node_data[c["node_id"]] = {
                "e": c["e_i"], "l": c["l_i"], "eta": c.get("eta_i", c["e_i"]+10),
                "s": c.get("s_i", 2), "q": c["q_i"], "type": "customer",
                "assigned_depot": c["assigned_depot"]
            }

        # Initialize problem
        self.prob = pulp.LpProblem("SAVRPTW_ConvexSTW", pulp.LpMinimize)

        # -----------------------------------------------------
        # 1. Variables
        # -----------------------------------------------------
        # x_{ij}^k: 1 if vehicle k travels from i to j, 0 otherwise
        x = pulp.LpVariable.dicts("x",
            ((i, j, k["id"]) for i in V for j in V for k in K if i != j),
            cat=pulp.LpBinary)

        # a_i^k: arrival time of vehicle k at node i
        a = pulp.LpVariable.dicts("a",
            ((i, k["id"]) for i in V for k in K),
            lowBound=0, cat=pulp.LpContinuous)

        # w_i^k: wait time (early arrival) at node i
        w = pulp.LpVariable.dicts("w",
            ((i, k["id"]) for i in V_c for k in K),
            lowBound=0, cat=pulp.LpContinuous)

        # tau_i^k: lateness at node i
        tau = pulp.LpVariable.dicts("tau",
            ((i, k["id"]) for i in V_c for k in K),
            lowBound=0, cat=pulp.LpContinuous)

        # P_L_i^k: convex penalty at node i
        P_L = pulp.LpVariable.dicts("P_L",
            ((i, k["id"]) for i in V_c for k in K),
            lowBound=0, cat=pulp.LpContinuous)

        # Load variable for MTZ capacity sub-tour elimination
        y = pulp.LpVariable.dicts("y",
            ((i, k["id"]) for i in V for k in K),
            lowBound=0, upBound=Q, cat=pulp.LpContinuous)

        # -----------------------------------------------------
        # 2. Objective Function: F_1
        # -----------------------------------------------------
        # Minimize sum of early wait times and the exponential lateness penalty
        # F_1 = sum( 1.0 * w_i^k + P_L_i^k )
        self.prob += pulp.lpSum(w[i, k["id"]] + P_L[i, k["id"]] for i in V_c for k in K)

        # -----------------------------------------------------
        # 3. Constraints
        # -----------------------------------------------------

        # A) Lateness and Early Wait Time Linking
        beta_stw = 0.12
        for i in V_c:
            for k in K:
                # w_i^k >= e_i - a_i^k
                self.prob += w[i, k["id"]] >= node_data[i]["e"] - a[i, k["id"]]
                # tau_i^k >= a_i^k - ETA_i
                self.prob += tau[i, k["id"]] >= a[i, k["id"]] - node_data[i]["eta"]

                # Piecewise Linear Approximation for (exp(0.12 * tau) - 1)
                # Tangent lines at tau = 0, 5, 10, 15, 20
                for tau_pt in [0, 5, 10, 15, 20]:
                    f_val = math.exp(beta_stw * tau_pt) - 1
                    f_prime = beta_stw * math.exp(beta_stw * tau_pt)
                    # P_L >= f(tau_pt) + f'(tau_pt) * (tau - tau_pt)
                    self.prob += P_L[i, k["id"]] >= f_val + f_prime * (tau[i, k["id"]] - tau_pt)

        # B) Flow Conservation and Depot Coupling
        for k in K:
            d_k = k["depot"]
            # A vehicle must start and end at its specific depot
            self.prob += pulp.lpSum(x[d_k, j, k["id"]] for j in V_c) == 1
            self.prob += pulp.lpSum(x[j, d_k, k["id"]] for j in V_c) == 1

            # Cannot visit other depots
            for d_other in V_d:
                if d_other != d_k:
                    for j in V:
                        if d_other != j:
                            self.prob += x[d_other, j, k["id"]] == 0
                            self.prob += x[j, d_other, k["id"]] == 0

            # Flow conservation for customers
            for j in V_c:
                self.prob += pulp.lpSum(x[i, j, k["id"]] for i in V if i != j) == \
                             pulp.lpSum(x[j, l, k["id"]] for l in V if l != j)

        # Each customer is visited exactly once
        for i in V_c:
            self.prob += pulp.lpSum(x[i, j, k["id"]] for j in V for k in K if i != j) == 1

        # C) Capacity Constraint (Q = 2)
        for k in K:
            for i in V:
                for j in V_c:
                    if i != j:
                        # y_j >= y_i + q_j - Q(1 - x_ij)
                        self.prob += y[j, k["id"]] >= y[i, k["id"]] + node_data[j]["q"] - Q * (1 - x[i, j, k["id"]])

        # D) Time-Linking Constraints & Service Time
        M = 1000
        def get_t_ij(i, j):
            # approximate travel time
            if i == j: return 0
            if self.G.has_edge(i, j):
                edges = self.G[i][j]
                return min(e.get("t_ij", float('inf')) for e in edges.values())
            return M

        for k in K:
            for i in V:
                for j in V_c:
                    if i != j:
                        t_ij = get_t_ij(i, j)
                        s_i = node_data[i]["s"]
                        # a_j >= a_i + s_i + t_ij - M(1 - x_ij)
                        self.prob += a[j, k["id"]] >= a[i, k["id"]] + s_i + t_ij - M * (1 - x[i, j, k["id"]])

        # E) Maximum Route Duration (T_max <= 35)
        for k in K:
            d_k = k["depot"]
            # Arrival back at depot must be <= 35
            for i in V_c:
                t_id = get_t_ij(i, d_k)
                s_i = node_data[i]["s"]
                self.prob += a[d_k, k["id"]] >= a[i, k["id"]] + s_i + t_id - M * (1 - x[i, d_k, k["id"]])
            self.prob += a[d_k, k["id"]] <= T_max

        # F) Risk and Residential Budgets (epsilon-constraints)
        def get_r_ij(i, j):
            if i == j: return 0
            if self.G.has_edge(i, j):
                edges = self.G[i][j]
                return min(e.get("r_ij", float('inf')) for e in edges.values())
            return 1.0

        def get_h_ij(i, j):
            if i == j: return 0
            if self.G.has_edge(i, j):
                edges = self.G[i][j]
                return min(e.get("h_ij", 0) for e in edges.values())
            return 1

        for k in K:
            # Route Risk Budget
            if R_bar != float('inf'):
                self.prob += pulp.lpSum(get_r_ij(i, j) * x[i, j, k["id"]] for i in V for j in V if i != j) <= R_bar

            # Route Residential Budget (Hard Cap <= 8 per route)
            self.prob += pulp.lpSum(get_h_ij(i, j) * x[i, j, k["id"]] for i in V for j in V if i != j) <= 8

        # Fleet Residential Budget
        if H_bar != float('inf'):
            self.prob += pulp.lpSum(get_h_ij(i, j) * x[i, j, k["id"]] for i in V for j in V for k in K if i != j) <= H_bar

        # -----------------------------------------------------
        # 4. Solve
        # -----------------------------------------------------
        self.prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))

        status = pulp.LpStatus[self.prob.status]
        if status == "Optimal":
            return pulp.value(self.prob.objective)
        else:
            return float('inf')

if __name__ == "__main__":
    pass
