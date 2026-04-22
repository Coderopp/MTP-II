"""
solver_drl.py - REINFORCE Pointer-Network baseline for MD-SA-VRPTW.

The environment is hard-masked so the agent can never emit a capacity-infeasible
or depot-mismatched action:

  * Capacity: if carrying_load + q_i > Q=2, customer i is masked to -inf.
  * Depot affinity: on the first step, only the rider's assigned depot d(k)
    is available as origin; returning to any other depot is forbidden.
  * After serving at least one customer, returning to d(k) is always allowed
    (rider may end the tour).
  * STW: the reward subtracts a soft-lateness penalty for arriving after l_i.
  * Multi-objective weights lambda = (0.4, 0.4, 0.2) matching the paper.

Training: vanilla REINFORCE with return normalization.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class MDVRPTW_Env:
    """
    One rider at a time: we iterate over riders in order and let the policy
    build each rider's route. A single "episode" dispatches all K riders.
    """

    def __init__(self, instance: dict, T_mat: np.ndarray, R_mat: np.ndarray,
                 C_mat: np.ndarray | None = None):
        self.depots = instance["depots"]
        self.customers = instance["customers"]
        params = instance["parameters"]
        self.Q = int(params["Q"])                       # hard Q = 2
        self.K = int(params["K"])
        self.L1 = float(params["lambda1"])
        self.L2 = float(params["lambda2"])
        self.L3 = float(params["lambda3"])

        self.D = len(self.depots); self.N = len(self.customers); self.V = self.D + self.N
        self.T = torch.tensor(T_mat, dtype=torch.float32)
        self.R = torch.tensor(R_mat, dtype=torch.float32)
        self.C = torch.tensor(C_mat if C_mat is not None else np.zeros_like(T_mat), dtype=torch.float32)

        self.q = torch.zeros(self.V)
        self.e = torch.zeros(self.V); self.l = torch.zeros(self.V)
        for d_idx, d in enumerate(self.depots):
            self.e[d_idx] = float(d.get("e_0", 0))
            self.l[d_idx] = float(d.get("l_0", 120))
        for c_idx, c in enumerate(self.customers):
            v = self.D + c_idx
            self.q[v] = float(c["q_i"])
            self.e[v] = float(c["e_i"])
            self.l[v] = float(c["l_i"])

        self.rider_depot: Dict[int, int] = {}
        for d_idx, d in enumerate(self.depots):
            for rk in d.get("riders", []):
                self.rider_depot[rk] = d_idx
        assert set(self.rider_depot) == set(range(self.K))

        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.unvisited = set(range(self.D, self.V))    # customer nodes only
        self.rider_order = list(range(self.K))
        self.current_rider_idx = 0
        self.curr_node = self.rider_depot[self.rider_order[0]]
        self.curr_load = 0.0
        self.curr_clock = float(self.e[self.curr_node])
        self.routes: Dict[int, List[int]] = {k: [self.rider_depot[k]] for k in range(self.K)}
        self.total_t = 0.0
        self.total_r = 0.0
        self.total_c = 0.0
        self.total_late = 0.0
        self.done = False
        return self._state()

    # ------------------------------------------------------------------ state
    def _state(self) -> torch.Tensor:
        # [one-hot curr_node | unvisited mask | remaining_cap | norm_clock]
        s = torch.zeros(self.V * 2 + 2)
        s[self.curr_node] = 1.0
        for v in self.unvisited:
            s[self.V + v] = 1.0
        s[-2] = (self.Q - self.curr_load) / max(self.Q, 1)
        s[-1] = self.curr_clock / max(float(self.l.max().item()), 1.0)
        return s

    # ------------------------------------------------------------------- mask
    def action_mask(self) -> torch.Tensor:
        """
        Return a boolean mask over V actions.
          * Customer v feasible iff v unvisited AND curr_load + q[v] <= Q.
          * Only the current rider's depot d(k) is a valid return action.
          * If the current node is a depot and there are unvisited customers,
            at least one customer must be chosen (no-op depot->depot forbidden).
        """
        mask = torch.zeros(self.V, dtype=torch.bool)
        k = self.rider_order[self.current_rider_idx]
        dk = self.rider_depot[k]

        for v in self.unvisited:
            if self.curr_load + float(self.q[v].item()) <= self.Q + 1e-9:
                mask[v] = True

        # Allow return to own depot only (never to another depot).
        if self.curr_node != dk:
            mask[dk] = True
        else:
            # At depot: if nothing feasible, allow staying (ends rider cleanly).
            if mask.sum().item() == 0:
                mask[dk] = True
        return mask

    # ------------------------------------------------------------------- step
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        prev = self.curr_node
        k = self.rider_order[self.current_rider_idx]
        dk = self.rider_depot[k]

        # Transition costs
        dt = float(self.T[prev, action].item())
        dr = float(self.R[prev, action].item())
        dc = float(self.C[prev, action].item())

        self.curr_clock += dt
        late = 0.0
        if action >= self.D:                         # customer
            if self.curr_clock < float(self.e[action].item()):
                self.curr_clock = float(self.e[action].item())
            late = max(0.0, self.curr_clock - float(self.l[action].item()))
            self.curr_load += float(self.q[action].item())
            assert self.curr_load <= self.Q + 1e-9, "mask failed"
            self.unvisited.discard(action)

        self.total_t += dt; self.total_r += dr; self.total_c += dc; self.total_late += late
        self.routes[k].append(action)
        self.curr_node = action

        # Reward: negative multi-objective cost
        reward = -(self.L1 * dt + self.L2 * dr + self.L3 * (dc + late))

        # Rider completion: returned to own depot AND has served >=1 customer
        rider_finished = (action == dk and len(self.routes[k]) >= 3) \
                      or (action == dk and prev == dk)  # no unvisited feasible
        if rider_finished:
            self.current_rider_idx += 1
            if self.current_rider_idx >= self.K:
                self.done = True
            else:
                next_k = self.rider_order[self.current_rider_idx]
                self.curr_node = self.rider_depot[next_k]
                self.curr_load = 0.0
                self.curr_clock = float(self.e[self.curr_node].item())

        # Unserved penalty applied as terminal shaping
        if self.done and self.unvisited:
            reward -= 50.0 * len(self.unvisited)

        return self._state(), reward, self.done

    def summary(self) -> dict:
        return {
            "travel_time": self.total_t,
            "risk":        self.total_r,
            "lateness":    self.total_late,
            "unserved":    len(self.unvisited),
            "routes":      self.routes,
            "objective":   self.L1 * self.total_t + self.L2 * self.total_r
                           + self.L3 * (self.total_c + self.total_late)
                           + 50.0 * len(self.unvisited),
        }


# ---------------------------------------------------------------------------
# Pointer-style policy
# ---------------------------------------------------------------------------
class PointerPolicy(nn.Module):
    def __init__(self, V: int, hidden: int = 128):
        super().__init__()
        self.V = V
        self.fc1 = nn.Linear(V * 2 + 2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, V)

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.head(x)
        # Hard mask - any mask==False -> -inf logit
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
        return F.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Training & inference
# ---------------------------------------------------------------------------
def train_agent(env: MDVRPTW_Env, epochs: int = 400, lr: float = 5e-3,
                seed: int = 42, verbose: bool = False) -> PointerPolicy:
    torch.manual_seed(seed)
    policy = PointerPolicy(env.V)
    opt = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(epochs):
        state = env.reset()
        log_probs, rewards = [], []
        while not env.done:
            mask = env.action_mask()
            logp = policy(state, mask)
            dist = torch.distributions.Categorical(logits=logp)
            a = dist.sample()
            state, r, _ = env.step(int(a.item()))
            log_probs.append(dist.log_prob(a))
            rewards.append(r)

        # Discounted returns, normalized
        R = 0.0; returns = []
        for r in rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if returns.std() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -torch.stack([lp * R for lp, R in zip(log_probs, returns)]).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ep % 50 == 0:
            print(f"  [DRL ep {ep}] reward_sum={sum(rewards):.2f}")
    return policy


def run_drl_inference(policy: PointerPolicy, env: MDVRPTW_Env) -> dict:
    t0 = time.time()
    state = env.reset()
    with torch.no_grad():
        while not env.done:
            mask = env.action_mask()
            logp = policy(state, mask)
            a = int(torch.argmax(logp).item())
            state, _, _ = env.step(a)
    res = env.summary()
    res["runtime"] = time.time() - t0
    res["algorithm"] = "DRL"
    res["K"] = env.K; res["Q"] = env.Q
    return res


def solve_drl(instance: dict, T_mat: np.ndarray, R_mat: np.ndarray,
              C_mat: np.ndarray | None = None, epochs: int = 400,
              seed: int = 42, verbose: bool = False) -> dict:
    env = MDVRPTW_Env(instance, T_mat, R_mat, C_mat)
    policy = train_agent(env, epochs=epochs, seed=seed, verbose=verbose)
    return run_drl_inference(policy, env)
