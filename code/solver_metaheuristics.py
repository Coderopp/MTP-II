import math

class MetaheuristicSolver:
    def __init__(self, instance_data, graph):
        self.instance = instance_data
        self.G = graph
        self.Q = self.instance["parameters"]["Q"]
        self.T_max = 35.0

        # Build lookup dictionaries
        self.node_data = {}
        for d in self.instance["depots"]:
            self.node_data[d["node_id"]] = {"type": "depot"}
        for c in self.instance["customers"]:
            self.node_data[c["node_id"]] = {
                "e": c["e_i"], "l": c["l_i"], "eta": c.get("eta_i", c["e_i"]+10),
                "s": c.get("s_i", 2), "q": c["q_i"], "type": "customer",
                "assigned_depot": c["assigned_depot"]
            }

    def _get_t_ij(self, i, j):
        if i == j: return 0
        if self.G.has_edge(i, j):
            return min(e.get("t_ij", float('inf')) for e in self.G[i][j].values())
        return float('inf')

    def _get_r_ij(self, i, j):
        if i == j: return 0
        if self.G.has_edge(i, j):
            return min(e.get("r_ij", float('inf')) for e in self.G[i][j].values())
        return 1.0

    def _get_h_ij(self, i, j):
        if i == j: return 0
        if self.G.has_edge(i, j):
            return min(e.get("h_ij", 0) for e in self.G[i][j].values())
        return 1

    def _evaluate_solution(self, route):
        """
        Calculates the exact F1 scoring for a given single micro-batch route.
        Route is a list of node_ids starting and ending at the same depot.
        """
        if not route or len(route) < 2:
            return 0.0

        depot = route[0]
        arrival_time = 0.0
        F1_score = 0.0
        beta_stw = 0.12

        for idx in range(1, len(route) - 1): # Exclude start and end depots
            prev_node = route[idx-1]
            curr_node = route[idx]

            t_ij = self._get_t_ij(prev_node, curr_node)
            s_prev = self.node_data[prev_node].get("s", 0) if prev_node != depot else 0

            arrival_time += s_prev + t_ij

            # Calculate wait time (w) and lateness (tau)
            e_i = self.node_data[curr_node]["e"]
            eta_i = self.node_data[curr_node]["eta"]

            w_i = max(0, e_i - arrival_time)
            tau_i = max(0, arrival_time - eta_i)

            # Convex STW Penalty
            F1_score += (1.0 * w_i) + (math.exp(beta_stw * tau_i) - 1)

            # If arriving early, vehicle waits until e_i
            arrival_time = max(arrival_time, e_i)

        return F1_score

    def _check_constraints(self, route, R_bar=float('inf'), H_bar=float('inf')):
        """
        Enforces strict constraints: Q<=2, depot mismatch, T_max<=35, Risk and H Budgets.
        Returns a tuple: (is_valid: bool, penalty: float)
        """
        if not route or len(route) < 2:
            return True, 0.0

        penalty = 0.0
        is_valid = True

        depot = route[0]
        if route[-1] != depot:
            is_valid = False
            penalty += 10000

        total_q = 0
        total_risk = 0.0
        total_h = 0
        arrival_time = 0.0

        for idx in range(1, len(route) - 1):
            prev_node = route[idx-1]
            curr_node = route[idx]

            # Depot mismatch check
            if self.node_data[curr_node]["assigned_depot"] != depot:
                is_valid = False
                penalty += 10000

            # Capacity
            total_q += self.node_data[curr_node]["q"]

            # Attributes
            t_ij = self._get_t_ij(prev_node, curr_node)
            r_ij = self._get_r_ij(prev_node, curr_node)
            h_ij = self._get_h_ij(prev_node, curr_node)

            total_risk += r_ij
            total_h += h_ij

            s_prev = self.node_data[prev_node].get("s", 0) if prev_node != depot else 0
            arrival_time += s_prev + t_ij

            e_i = self.node_data[curr_node]["e"]
            arrival_time = max(arrival_time, e_i)

        # Return to depot
        last_cust = route[-2]
        s_last = self.node_data[last_cust].get("s", 0)
        t_id = self._get_t_ij(last_cust, depot)
        arrival_time += s_last + t_id

        total_risk += self._get_r_ij(last_cust, depot)
        total_h += self._get_h_ij(last_cust, depot)

        # Check bounds
        if total_q > self.Q:
            is_valid = False
            penalty += 5000 * (total_q - self.Q)

        if arrival_time > self.T_max:
            is_valid = False
            penalty += 5000 * (arrival_time - self.T_max)

        if total_risk > R_bar:
            is_valid = False
            penalty += 5000 * (total_risk - R_bar)

        # Hard cap per route is 8 for residential
        if total_h > 8 or total_h > H_bar:
            is_valid = False
            penalty += 5000 * max((total_h - 8), (total_h - H_bar))

        return is_valid, penalty

if __name__ == "__main__":
    pass
