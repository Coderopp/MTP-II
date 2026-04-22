# SA-VRPTW — Canonical Mathematical Formulation

**Status:** authoritative specification. Every solver (MILP, GA, ALNS), instance generator, and evaluator in `savrptw/` MUST implement this exactly. Deviations require an update to this document first.

**Revision:** v1.0 — 2026-04-22. Target journal: Wiley OR / transportation family (Networks / JAT / ITOR).

---

## 1. Problem Statement

A quick-commerce platform operates a cluster of dark stores (depots) in an urban area. Customers place orders with a **10-minute delivery promise**. A homogeneous fleet of riders dispatches out of each depot in micro-batches (typically 1–3 orders per route). Every road segment carries a *per-traversal crash probability* `r_ij`, a *residential-street indicator* `h_ij`, and a *congestion-inflated travel time* `t_ij`.

Given one dispatch batch, find routes that (primarily) minimise deviation between promised and actual arrival times plus an exponential lateness penalty, while respecting:

- vehicle capacity, time windows, and maximum route duration,
- a per-route **risk budget** `R̄` (cumulative log-survival penalty),
- a fleet-wide **residential-road budget** `H̄` (protect vulnerable neighbourhoods),
- a per-route **max residential hops** `H̄_k` (anti-funnel protection).

The Pareto frontier over `(R̄, H̄)` is traced by ε-constraint sweeping.

---

## 2. Sets and Indices

| Symbol | Description |
|---|---|
| `G = (V, E)` | directed street graph (OSMnx-extracted for the city) |
| `D ⊂ V` | depot nodes (k-means cluster centroids of real Blinkit stores, snapped to OSM nodes) |
| `C ⊂ V` | customer nodes (sampled with residential-density weighting; see §11) |
| `N_d = \|C_d\|` | customers assigned to depot `d`; each customer is pre-assigned to its nearest depot by free-flow time |
| `K` | set of riders, partitioned `K = ⋃_{d∈D} K_d` |
| `i, j ∈ V` | node indices |
| `k ∈ K` | rider index |
| `(i,j) ∈ E` | directed edge |

### Super-graph note

MILP and ALNS operate on a **complete customer+depot super-graph** in which arc `(i,j)` is the shortest-time path from `i` to `j` on the underlying street graph. All arc-level attributes (`t_ij`, `R_ij`, `C_ij`, `H_ij`) are computed by summing over the street-graph edges of that shortest path. The GA operates directly on permutations of `C ∪ D` and reconstructs the same super-graph at evaluation time. **All three solvers evaluate solutions via identical functions** in `savrptw.eval`.

---

## 3. Parameters

### 3.1 Edge parameters (street graph)

| Symbol | Units | Source |
|---|---|---|
| `len_ij` | metres | OSM `length` |
| `v_ij^free` | km/h | OSM `speed_kph` (imputed via `osmnx.add_edge_speeds`) |
| `t_ij^free = 60·len_ij / (1000·v_ij^free)` | minutes | derived |
| `ρ_ij ∈ [0, ρ_max]` | dimensionless | BPR congestion factor: `ρ_ij = α(V_ij/C_ij)^β`, `α=0.15`, `β=4`; `V/C` imputed from highway class × lane count × time-of-day profile |
| `t_ij = t_ij^free · (1 + ρ_ij)` | minutes | congestion-inflated travel time — **this is the travel time used everywhere downstream** |
| `r_ij ∈ [0, 0.99]` | probability | per-traversal crash probability (BASM; §10.2) |
| `h_ij ∈ {0,1}` | binary | 1 iff OSM `highway ∈ {residential, living_street}` |

### 3.2 Super-graph arc parameters (for customer/depot super-graph)

For each `(i,j)` where `i, j ∈ C ∪ D`, let `P_ij` be the free-flow shortest path on the street graph. Then:

| Symbol | Definition |
|---|---|
| `T_ij = Σ_{e ∈ P_ij} t_e` | total travel time (min) |
| `R_ij = Σ_{e ∈ P_ij} −ln(1 − r_e)` | cumulative log-survival (dimensionless; additive by survival independence) |
| `H_ij = Σ_{e ∈ P_ij} h_e` | count of residential edges used |

**Why shortest-time paths for the super-graph**: fixes `P_ij` at instance-generation time so all three solvers see the same arcs. The paper states this clearly (§3 Methodology).

### 3.3 Customer parameters

| Symbol | Description |
|---|---|
| `q_i ∈ {1,2}` | demand (items) — drawn from observed Blinkit basket size |
| `e_i` | order placement time (min since batch start) |
| `ETA_i = e_i + 10` | promised arrival time (q-commerce 10-min promise) |
| `l_i = ETA_i` | soft upper bound of time window |
| `s_i ∈ {2, 4}` | service time — 2 min default, 4 min if customer node lies within OSM `building=apartments`, `residential=tower`, or any polygon ≥ 4 floors (see §10.5) |

Time windows are **soft**. Early arrival incurs a **waiting** penalty (riders' time is burnt idling); late arrival incurs the **exponential STW penalty** `P_L`.

### 3.4 Fleet and instance constants

| Symbol | Value | Meaning |
|---|---|---|
| `Q` | 2 | vehicle capacity (items) |
| `K_d` | ⌈1.2 · N_d / Q⌉ | riders per depot (slight over-provision) |
| `T_max` | 35 min | maximum route duration (dark-store→customers→dark-store) |
| `Service hours` | shift is ~240 min, but one *route* ≤ `T_max` |
| `R̄` | ε-parameter | per-route risk budget (swept) |
| `H̄` | ε-parameter | fleet residential-edge budget (swept) |
| `H̄_k` | 8 | per-route residential-edge cap (anti-funnel) |
| `β_stw` | 0.12 | lateness-penalty exponent (from §4.2) |
| `w_early` | 1.0 | idle-wait weight in objective |
| `M` | 10^4 | big-M for linking constraints (MILP) |

---

## 4. Decision Variables

| Symbol | Domain | Meaning |
|---|---|---|
| `x_{ij}^k ∈ {0,1}` | binary | rider `k` traverses super-arc `(i,j)` |
| `a_i^k ≥ 0` | continuous, min | arrival time at node `i` by rider `k` (defined only if rider visits `i`) |
| `w_i^k ≥ 0` | continuous, min | idle waiting time at `i` (early arrival) |
| `τ_i^k ≥ 0` | continuous, min | lateness at `i` (`= max(0, a_i^k − ETA_i)`) |
| `u_i^k ∈ {1, …, \|C\|}` | integer | MTZ sequence variable for sub-tour elimination |

### Effective arrival linkage

The service-start time at customer `i` is `max(a_i^k, e_i)`. In the linear model, enforce:

```
w_i^k ≥ e_i − a_i^k
τ_i^k ≥ a_i^k − ETA_i
```

Both with domain `≥ 0`; the solver will drive whichever side is zero.

---

## 5. Objective — F₁ (primary)

**Minimise ETA deviation + soft-time-window penalty**, summed across the fleet:

```
F₁ = Σ_{k ∈ K} Σ_{i ∈ C} [ w_early · w_i^k  +  ( exp(β_stw · τ_i^k) − 1 ) ]
```

- `w_i^k` = idle wait when rider arrives before order is ready (early deviation).
- `exp(β_stw · τ_i^k) − 1` = convex SLA penalty when rider arrives after promised ETA (late deviation).
- No raw travel-time term: travel time is *implicit* through the arrival-time constraints.
- No congestion term: congestion is already baked into `t_ij`.
- No risk term: risk is handled as a constraint (ε-method, §6).

### 5.1 MILP linearisation of the exponential lateness penalty

`exp(β·τ) − 1` is convex. In the MILP, approximate via piecewise-linear outer-approximation with breakpoints `τ ∈ {0, 2, 5, 10, 15, 20, 30}` minutes. Use SOS2 or a simple linear-segment lower envelope. GA/ALNS evaluate the exponential directly.

---

## 6. ε-Constraints (swept)

### 6.1 Per-route crash-survival budget

For every rider `k`, cumulative log-survival ≤ `R̄`:

```
Σ_{(i,j) ∈ E_super} R_ij · x_{ij}^k  ≤  R̄,   ∀ k ∈ K
```

Interpretation: the survival probability along rider `k`'s full tour is ≥ `exp(−R̄)`. With `R̄ = 0.1`, survival ≥ 0.905.

### 6.2 Fleet-wide residential-road budget

```
Σ_{k ∈ K} Σ_{(i,j) ∈ E_super} H_ij · x_{ij}^k  ≤  H̄
```

### 6.3 Per-route residential cap

```
Σ_{(i,j) ∈ E_super} H_ij · x_{ij}^k  ≤  H̄_k,  ∀ k ∈ K
```

### 6.4 ε-sweep protocol

Sweep `R̄ ∈ {0.05, 0.10, 0.20, 0.40, 0.80, 1.60, ∞}` and `H̄ ∈ {0, 4, 8, 16, 32, ∞}` over a 7×6 grid; for each cell record `F₁*`. Apply dominance filter (§13) before plotting. The resulting non-dominated frontier is Figure “Pareto Surface”.

---

## 7. Core VRPTW Constraints

All summations are over feasible indices only (skip self-loops).

### 7.1 Each customer visited exactly once

```
Σ_{k ∈ K} Σ_{j ∈ C ∪ D, j≠i} x_{ji}^k = 1,   ∀ i ∈ C     (1)
```

### 7.2 Flow conservation

```
Σ_{i ∈ C ∪ D, i≠j} x_{ij}^k = Σ_{i ∈ C ∪ D, i≠j} x_{ji}^k,   ∀ j ∈ C, ∀ k ∈ K     (2)
```

### 7.3 Capacity

```
Σ_{i ∈ C} q_i · Σ_{j ∈ C ∪ D, j≠i} x_{ij}^k  ≤  Q,   ∀ k ∈ K     (3)
```

### 7.4 Multi-depot coupling (each rider is tied to one depot)

For each rider `k ∈ K_d`:

```
Σ_{j ∈ C} x_{d,j}^k  ≤  1,    (leaves depot ≤ once)         (4)
Σ_{j ∈ C} x_{j,d}^k  = Σ_{j ∈ C} x_{d,j}^k   (returns if left)  (5)
x_{ij}^k = 0    for all i ∈ D \ {d} or j ∈ D \ {d}          (6)
```

(Rider `k` cannot touch any depot other than its home depot `d`.)

### 7.5 Sub-tour elimination (MTZ)

For `i, j ∈ C`, `i ≠ j`:

```
u_i^k − u_j^k + |C| · x_{ij}^k  ≤  |C| − 1,    ∀ k ∈ K     (7)
1 ≤ u_i^k ≤ |C|,    ∀ i ∈ C, ∀ k ∈ K                       (8)
```

### 7.6 Arrival-time linking

For `i ∈ C ∪ D`, `j ∈ C`, `i ≠ j`, `∀ k`:

```
a_j^k  ≥  a_i^k + s_i + T_ij  −  M · (1 − x_{ij}^k)          (9)
```

At depot: `a_d^k = 0` if rider `k` dispatches from `d` (initial).

### 7.7 Max route duration

```
Σ_{i ∈ C} Σ_{j ∈ C ∪ D, j≠i} T_ij · x_{ij}^k  +  Σ_{i ∈ C} s_i · Σ_{j} x_{ji}^k  ≤  T_max,  ∀ k ∈ K   (10)
```

### 7.8 Soft time windows

```
w_i^k  ≥  e_i − a_i^k,    w_i^k ≥ 0         (11)
τ_i^k  ≥  a_i^k − ETA_i,  τ_i^k ≥ 0         (12)
```

### 7.9 Domain

```
x_{ij}^k ∈ {0,1},   a_i^k, w_i^k, τ_i^k ≥ 0,   u_i^k ∈ ℤ_{≥1}
```

---

## 8. Complete Compact Formulation

```
min  F₁ = Σ_{k, i} [ w_early · w_i^k + ( exp(β_stw · τ_i^k) − 1 ) ]
s.t. (1)–(12),
     per-route risk:        Σ_j R_ij x_{ij}^k ≤ R̄                    ∀k
     fleet residential:     Σ_{k,i,j} H_ij x_{ij}^k ≤ H̄
     per-route residential: Σ_{i,j} H_ij x_{ij}^k ≤ H̄_k              ∀k
```

`R̄` and `H̄` are swept to trace the Pareto frontier.

---

## 9. Instance-Generation Logic

Pseudocode (implemented in `savrptw.instance`):

```
load_osm(city)                           # OSM drive network, bboxed
attach_edge_attrs(G, BASM, BPR)          # §10
depot_points = kmeans(blinkit_kml[city], k ∈ {3,5,8})
depots = [nearest_osm_node(G, p) for p in depot_points]
customers = []
for _ in range(N):                       # N ∈ {20,50,100,200}
    node = sample_weighted(G, weights=residential_density(G))
    customers.append(node)
for c in customers:
    c.home_depot = argmin_d T_shortest(d, c, weight='t_ij')
    c.q_i = random([1,2])
    c.s_i = 4 if is_highrise(c) else 2
    c.e_i = uniform(0, 60)               # order placed in first hour
    c.ETA_i = c.e_i + 10
```

---

## 10. Data Calibration — Every Parameter Has a Source

### 10.1 Road network

OSMnx `graph_from_place(city, network_type='drive')`. Clipped to **city-specific bounding box** stored in `conf/cities/<city>.yaml`. **Delete `utils.BENGALURU_BBOX`.**

### 10.2 BASM — per-traversal crash probability `r_ij`

**Status: resolved in Task #16.** The calibrated source for this branch is `morth_mohan_osm_proxy_v1`, documented in `docs/BASM_CALIBRATION.md`.

Functional form (will hold regardless of which source calibrates the constants):

```
r_ij = clip(λ_class · (len_ij / 1000) · severity_multiplier · proxy_edge_weight, 0, 0.99)
```

where:
- `λ_class` (per-km base rate) is derived from MoRTH 2022 road-category fatality totals and road-length shares, uplifted from fatal-only to fatal+grievous using the national 2022 fatal/grievous ratio, then converted to per-traversal priors using the project's class-level exposure table,
- `severity_multiplier` encodes the arterial > collector > local ordering supported by Mohan et al. (2017) and is computed through a reproducible bounded transform of MoRTH road-category fatality intensity,
- `proxy_edge_weight` redistributes risk within each class using approximate edge betweenness, local signal density, and local crossing density, clipped to a bounded range so it cannot overwhelm the sourced class prior.

Full derivation and citations are in `docs/BASM_CALIBRATION.md`. **Every constant in `conf/risk/basm_v1.yaml` must trace back to that note.**

Cross-validation rule: expected annual events from the instantiated `r_ij` field must match the chosen public city aggregate within ±25% once the city-level targets are assembled.

### 10.3 BPR congestion `ρ_ij`

```
ρ_ij = 0.15 · ( V_ij / C_ij )^4
```

- `C_ij` (capacity, veh/hour): `hcm_capacity_table[highway_class, lane_count]` (Highway Capacity Manual 2010 Exhibit 10-3).
- `V_ij` (flow, veh/hour): `aadt_multiplier[hour_of_day] · AADT_class`; AADT-class defaults drawn from Central Road Research Institute (CRRI) New Delhi urban-flow surveys.
- Peak-hour profile: `8-11 AM` and `6-9 PM` use peak multipliers; other hours off-peak.
- No paid API calls. Cross-validate on a 50-edge sample of Bengaluru against HERE Free Tier (optional Task #10 extension).

### 10.4 Dark stores (depots)

Source: `data/raw/darkstoremap_in_2026-04-09.kml` (committed snapshot from `https://darkstoremap.in/dark_store.kml`, 2026-04-09). Primary single-brand experiments retain **Bengaluru, Delhi, Gurugram, Mumbai, and Pune**. Hyderabad is excluded because the committed snapshot yields zero Blinkit placemarks inside the Hyderabad bounding box; the regression test documenting this gap remains in the suite. For retained cities, filter the committed KML by allowed folders and city bounding box, then apply k-means to 3/5/8 depots by instance size.

### 10.5 Service time `s_i`

Query OSM for building tags within 30 m of each customer node. If any polygon has `building=apartments` and `building:levels ≥ 4`, or `building=residential` with levels ≥ 5, then `s_i = 4`; else `s_i = 2`. Document the rule in `docs/SERVICE_TIME.md`.

### 10.6 Customer placement

Weight node sampling by residential-land-use density: for each graph node, compute the area of OSM polygons with `landuse=residential` within 200 m, normalise to a sampling weight. Deliveries thus land in residential zones, not on highways or industrial parks.

### 10.7 Behavioural compliance (for evaluation, not for optimisation)

Rider-behaviour Monte Carlo (§12.3) is an **evaluation** step only; it does not enter `F₁`. Beta(α, β) prior stays, but is calibrated with citations:

- `α = 5, β = 2` — central estimate; cites e.g. Das et al. (2020) on Indian gig-delivery rider compliance.
- Sensitivity sweep: `α ∈ {3, 5, 8}`, `β ∈ {1, 2, 3}` — reported as IQR bands in the compliance-vs-safety figure.

---

## 11. Evaluation Metrics (reported in paper tables)

Per solution, compute:

| Metric | Definition |
|---|---|
| `F₁*` | objective value (primary) |
| `Σ_k τ̄_k` | mean per-route lateness (min) |
| `R_total` | Σ of route-level `Σ R_ij x` (informational, should match active `R̄`) |
| `E[fatalities]` | `1 − exp(−R_total)` converted to expected-crashes per dispatch |
| `Residential share` | `H_total / total_edges_used` |
| `Run time` | wall-clock seconds |
| `MIP gap` | for MILP only |

Statistical comparison across solvers: **Wilcoxon signed-rank** on paired runs (same seed, same instance), **Bonferroni-corrected** for multi-way (MILP/GA/ALNS) comparisons.

---

## 12. Reproducibility

- Fixed seeds: `{42, 101, 202, 303, 404, 505, 606, 707, 808, 909}` — 10 replications per configuration.
- All code deterministic given `(seed, city, N, R̄, H̄, solver)`.
- Every run logs: git SHA, full Hydra config dump, solver versions, hardware, wall-clock.

### 12.1 Crash-survival Monte Carlo (replaces SUMO)

For each rider's realised tour `P_k`:

```
for n in range(10_000):
    traversed = False
    for edge in P_k:
        if bernoulli(r_edge) samples 1:
            crash_n = True; break
    counts.append(crash_n)
p_crash_route = mean(counts)
CI_95 = bootstrap CI
```

Reported as "expected crashes per 10 k dispatches, 95 % CI" per solver.

### 12.2 Behavioural compliance MC

Same structure, compliance-modified `r_edge`:

```
r_effective = 1 − (1 − r_edge) ^ (1 + k_c·(1 − c))
```

with `k_c = 2` (non-compliant rider triples expected risk at `c=0`). Citation TBD in Task #12.

---

## 13. Pareto Dominance Filter

Before plotting any "Pareto" frontier, apply standard dominance:

```
for each point p in sweep:
    p is non-dominated iff no other q satisfies
        q.F₁ ≤ p.F₁ AND q.R_total ≤ p.R_total AND at least one strict
```

Only non-dominated points are plotted. Documented procedure; the current `12_all_cities_figures.py` does NOT do this — fixing it is a Task #14 requirement.

---

## 14. Deviations from the Existing LaTeX Proposal

This formulation is the **corrected** version. Differences from `SafetyAwareRoutingProposal.tex`:

1. `F₁` is ETA-deviation + STW, **not** `Σ t_ij x + Σ P_L`. Rationale: the paper's original `F₁` double-counts travel time (it's already in arrival time that drives lateness).
2. Risk moves from objective `F₂` to **ε-constraint** (§6.1).
3. Congestion is absorbed into `t_ij` via BPR, **not** a third objective.
4. **Service time `s_i`** is added (absent in current paper).
5. **Max route duration `T_max`** is added (absent in current paper).
6. **Multi-depot coupling (4–6)** enforced — current GA silently violates it.
7. `r_ij` is **per-traversal crash probability** in `[0, 0.99]`, calibrated from MoRTH, not an ad-hoc severity index.
8. MTZ uses coefficient `|C|`, not `Q` (the current MILP erroneously ties sub-tour elimination to capacity).
9. Capacity constraint operates on `q_i` (respects variable demand); current MILP ignores `q_i`.
10. DRL objective is **out of scope** for this paper.
11. Pareto frontier is produced via dominance-filtered ε-sweep; current code plots raw samples.
12. Behavioural MC is an **evaluation** step, not part of the cost.

The LaTeX will be rewritten to match this (Task #15).

---

## 15. Contract: All Solvers Must

- Read the same Hydra-materialised instance object from `savrptw.instance.Instance`.
- Call `savrptw.eval.objective(solution, instance)` — the **only** scoring function in the codebase.
- Return `savrptw.solvers.Solution` — a dataclass containing `routes: List[List[int]]`, `arrivals: dict[(k,i) -> float]`, and `run_meta: dict`.
- Reject any solution that violates constraints (1)–(12) or the active ε-bounds. Return `Infeasible` status rather than silently emit invalid routes.

If a solver needs to relax a constraint (e.g., GA uses soft penalties in fitness), the evaluator still validates the final incumbent against the full formulation before accepting it.

---

*End of canonical formulation.*
