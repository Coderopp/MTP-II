# SA-VRPTW Codebase Audit Report
*A line-by-line review of the Safety-Aware VRPTW pipeline — data sources, parameters, shortcuts, and mock/simulated data.*

---

## 0. Executive Summary

The repository implements a **Safety-Aware Vehicle Routing Problem with Time Windows (SA-VRPTW)** pipeline for quick-commerce delivery in Bengaluru (and 5 other Indian cities for figure generation). The codebase is split across two **parallel pipelines** that are only loosely coupled:

| Pipeline | Purpose | Data Source | Scripts |
|---|---|---|---|
| **A — "Production" pipeline** | End-to-end solve on real Bengaluru OSM graph, synthetic accidents, speed-proxy congestion, 30-customer GA solution | `data/03_graph_final.graphml` (99,572 nodes, 256,887 edges) | `01→02→03→04→05→07_solver_ga→07_verify→08_behavioral_model→09_dashboard` |
| **B — "Publication figures" pipeline** | Generate 9 figure types × 6 cities for the LaTeX proposal | *Fresh* OSM pulls with **fabricated r_ij / c_ij / h_ij** per edge | `08_run_real_benchmarks.py`, `10_spatial_overlays.py`, `12_all_cities_figures.py` |

**Headline findings**
1. **Every "risk" and "congestion" value used in the published figures is synthetic.** No real iRAD data is loaded anywhere (`data/02_irad_metadata.json` → `"data_source": "synthetic"`).
2. **The DRL "O(1) inference" figure is faked** — plotted as `ga_time × 0.05` (line 198 of `12_all_cities_figures.py`), not measured from the actual `09_drl_agent.py` PyTorch network.
3. **SUMO integration (`11_sumo_integration.py`) is a stub** that swallows exceptions and prints "Bypassing for theoretical workflow tracking".
4. **"MILP" benchmarks use `PULP_CBC_CMD(timeLimit=12–15s)`** — timeouts are frequent (returns `NaN`), and the scalability plot only tests N ∈ {8, 12, 16}.
5. **Multiple parameter inconsistencies** between scripts (Q=2 in instance, Q=3 in GA solver, Q=3 in DRL; K=30 in instance, K=3 in MILP, K=5 in figures).
6. **Hard-coded Bengaluru-centric bounding box** is reused across the 5 other cities via `filter_bbox` (would clip every non-Bengaluru accident to zero if real data were supplied).
7. **DRL training data is pure `np.random.uniform`** (`09_drl_agent.py` line 196-197) — the network never sees real city topology.

---

## 1. Repository Layout

```
MTP-Final/
├── code/
│   ├── utils.py                     # Graph I/O, bbox, normalization
│   ├── run_pipeline.py              # Master runner (steps 1–5)
│   ├── 01_osm_graph.py              # Download OSM drive network
│   ├── 02_irad_risk.py              # Compute r_ij (REAL or SYNTHETIC)
│   ├── 03_congestion.py             # Compute c_ij (SPEED_PROXY or GOOGLE_MAPS)
│   ├── 04_instance_generator.py     # 30-customer multi-depot VRPTW
│   ├── 05_validate_instance.py      # 9-check validation suite
│   ├── 06_export_geojson.py         # Web export (references obsolete schema)
│   ├── 07_solver_ga.py              # Single-objective weighted-sum GA
│   ├── 07_verify_solution.py        # Lightweight solution check
│   ├── 08_behavioral_model.py       # Rider compliance Monte Carlo
│   ├── 08_run_real_benchmarks.py    # Figures 6-8 (Phase 6 plots)
│   ├── 09_dashboard_visualization.py# Folium map + behavioral chart
│   ├── 09_drl_agent.py              # PyTorch REINFORCE pointer (stand-alone)
│   ├── 10_spatial_overlays.py       # Fast-vs-safe route divergence, heatmap
│   ├── 11_sumo_integration.py       # SUMO *placeholder* (no actual sim)
│   ├── 12_all_cities_figures.py     # 9 figures × 6 cities (main driver)
│   ├── rewrite_simplified.py        # Overwrites the LaTeX proposal
│   └── rewrite_proposal_v3.py       # Patches the LaTeX proposal
├── data/ …                          # GraphML + instance JSON/PKL
├── figures/                         # Generated plots
│   ├── Bengaluru/… Mumbai/          # 9 figures × 6 cities
└── cache/                           # OSMnx HTTP cache (51 MB)
```

Total: **3,668 LOC** across 19 Python scripts.

---

## 2. High-Level Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     PIPELINE A — "PRODUCTION" SOLVE                      │
└──────────────────────────────────────────────────────────────────────────┘

   [OSMnx HTTP]                    [iRAD CSV if present]          [Google Maps API if key set]
        │                                   │                             │
        ▼                                   ▼                             ▼
 ┌─────────────┐  99.6k nodes     ┌─────────────────┐        ┌──────────────────────┐
 │ 01_osm      │  256.9k edges    │ 02_irad_risk    │        │ 03_congestion        │
 │ 10 km @     │ ───────────────► │ REAL  or        │ ─────► │ SPEED_PROXY   or     │
 │ (12.97,77.6)│  t_ij            │ SYNTHETIC (150) │ r_ij   │ GOOGLE_MAPS (≤200)   │ c_ij
 └─────────────┘                  └─────────────────┘        └──────────────────────┘
        │                                   │                             │
        ▼                                   ▼                             ▼
                                    data/03_graph_final.graphml  (enriched MultiDiGraph)
                                                   │
                                                   ▼
                                    ┌──────────────────────────┐
                                    │ 04_instance_generator    │  → 04_vrptw_instance.json
                                    │ 30 cust, 3 depots,       │    (rolling horizon,
                                    │ K=30, Q=2, TW=15min,     │     tt_from_depot matrix)
                                    │ λ=(0.4,0.4,0.2)          │
                                    └──────────────────────────┘
                                                   │
                                                   ▼
                                    ┌──────────────────────────┐
                                    │ 05_validate_instance     │  (9 checks: edge attrs,
                                    │                          │   reachability, λ sum=1)
                                    └──────────────────────────┘
                                                   │
                                                   ▼
                                    ┌──────────────────────────┐
                                    │ 07_solver_ga             │  GA (pop=50, gens=100)
                                    │ weighted-sum fitness     │  → 05_solution.json
                                    │ CAPACITY=3 (≠ instance!) │    (17 routes, score 394.28)
                                    └──────────────────────────┘
                                                   │
                                                   ▼
                                    ┌──────────────────────────┐
                                    │ 08_behavioral_model      │  Beta(5,2) compliance draws,
                                    │ 10 riders × 17 routes    │  tt multiplier=0.8+0.4(1-c)
                                    │                          │  risk multiplier=1+2(1-c)
                                    └──────────────────────────┘
                                                   │
                                                   ▼
                                    ┌──────────────────────────┐
                                    │ 09_dashboard_visualizat. │  folium map + PNG chart
                                    └──────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│             PIPELINE B — "PUBLICATION FIGURES" (6 cities × 9 plots)      │
└──────────────────────────────────────────────────────────────────────────┘
    For each city in {Bengaluru, Delhi, Gurugram, Hyderabad, Pune, Mumbai}:

        [OSMnx graph_from_point(center, dist=1500m)]
                      │
                      ▼
         [Fabricate r_ij, c_ij, h_ij from highway-type + Gaussian noise]
                      │
                      ▼
         [matrices(G, n)] — sample n ∈ {12,15,20} nodes, run all-pairs Dijkstra
                      │
                      ├─► fig_scalability  (MILP ≤12s, GA, **DRL = GA × 0.05 FAKE**)
                      ├─► fig_pareto       (30-point GA sweep over λ)
                      ├─► fig_convergence  (3 GA runs, λ ∈ {0.9, 0.5, 0.1})
                      ├─► fig_smax         (cumulative risk along GA route)
                      ├─► fig_hierarchy    (stacked bar by highway class)
                      ├─► fig_fleet_risk   (random boxplot)
                      ├─► fig_stw          (lateness=t-25+N(0,4); P = exp(0.12·L)-1)
                      └─► fig_folium_maps  (fastest vs safest via neg-log risk)
```

---

## 3. Line-by-Line Walk-Through

### 3.1 `code/utils.py` (138 LOC)

| Line | Content |
|---|---|
| 17-21 | Path constants. `DATA_DIR = repo/data`. |
| 37-49 | `load_graph`: reads GraphML, re-casts `length, t_ij, r_ij, c_ij, speed_kph, travel_time, maxspeed` back to float (GraphML stores everything as string). |
| 52-89 | `save_graph`: **deep-copies** the graph, strips geometry/lists/non-primitives so GraphML can serialize. Note: deep-copy of a 99 k-node graph is memory-hungry. |
| 97-105 | `normalize`: min-max to [0,1], clipped. |
| 113-118 | **`BENGALURU_BBOX`**: hard-coded lat [12.834, 13.144], lon [77.460, 77.784]. Used as the **universal filter** for accidents even though the figure-generation pipeline covers 5 other cities. |

### 3.2 `code/01_osm_graph.py` (114 LOC)

**Parameters**
- `NETWORK_TYPE = "drive"`
- `CENTRE_LAT, CENTRE_LON = 12.9716, 77.5946` (Bengaluru GPO)
- `RADIUS_M = 10_000` (10 km)

**Flow**
1. `_try_point()` → `osmnx.graph_from_point(center, dist=10 km)`; on failure falls back to `_try_place("Bengaluru, Karnataka, India")`.
2. `ox.add_edge_speeds(G)` → fills `speed_kph` from OSM `maxspeed` tags; defaults to *class median* when tag missing.
3. `ox.add_edge_travel_times(G)` → `travel_time` (seconds) = length / speed.
4. Lines 96-97: **`t_ij` = travel_time / 60** (minutes).
5. Lines 100-102: placeholders `r_ij=0.0`, `c_ij=1.0` on every edge.

**Real/mock**: **REAL** OSM topology. Speed values are OSM-tag-imputed (no real traffic).

**Shortcut**: 10 km radius around MG Road; anything outside central Bengaluru is ignored.

### 3.3 `code/02_irad_risk.py` (286 LOC) — **synthetic by default**

**Parameters**
- `IRAD_CSV_PATH = DATA_DIR / "irad_accidents.csv"` → **file absent** (see `.gitignore`/tree); triggers SYNTHETIC mode.
- `SEVERITY_WEIGHTS = {fatal:4, grievous:3, minor:2, damage:1}`.
- `SYNTHETIC_N = 150`, `YEARS_SPAN = 5`, `RANDOM_SEED = 42`.

**Synthetic data generator** (lines 59-100)
```python
cluster_centres = [(12.97,77.59), (12.93,77.62), (12.98,77.64), (13.03,77.59)]
weights        = [0.35, 0.25, 0.25, 0.15]
severities     = [fatal, grievous, minor, damage only]
sev_probs      = [0.10, 0.20, 0.45, 0.25]
```
→ Draws 150 lat/lon points from a 4-component Gaussian mixture with σ=0.008° (~880 m).

**Risk computation** (lines 155-246)
1. `ox.distance.nearest_edges(G, lons, lats)` snaps each accident to the closest OSM edge.
2. `edge_risk[(u,v,k)] += severity_weight` (sum of severities per edge).
3. **Normalization**: divide by `5 years × length_km`, apply `log1p`, then min-max to [0,1] (lines 222-225). Log-scaling is explicitly introduced to damp "a single massive pileup" from flattening variance.

**Result observed in `data/02_irad_metadata.json`:**
```
{ "data_source": "synthetic", "accident_count": 150, "year_range": [2018, 2023],
  "severity_counts": {"minor":77,"damage only":34,"grievous":30,"fatal":9},
  "edges_with_risk": 137, "total_edges": 256887 }
```
→ **Only 0.053 % of edges have a non-zero r_ij.** This is an enormous sparsity issue that cascades into the solver (most inter-node Dijkstra paths on `r_ij` weight will return 0).

### 3.4 `code/03_congestion.py` (319 LOC)

**Parameters**
- `CONGESTION_MODE = "SPEED_PROXY"` (default, from env).
- `MAX_API_EDGES = 200`, `API_SLEEP_SEC = 0.05` (if `GOOGLE_MAPS` mode).

**Mode A — SPEED_PROXY** (no API, 256,887 edges processed)
1. For each edge: `speed_kph` (default 30 if missing, floor 5).
2. `speed_norm = minmax(speeds)` across the whole graph.
3. **Logic** (lines 93-100):
   - If `highway ∈ {residential, living_street, service, pedestrian}`: `c_ij = 0.1 + 0.1·speed_norm` (low, because inherently slow but empty).
   - Else: `c_ij = 1 − speed_norm` (arterials with lower limit = more congested).
4. Metadata observed: `c_ij_mean = 0.2115`, `min = 0.0`, `max = 1.0`.

**Mode B — GOOGLE_MAPS** (only 200 edges sampled)
1. Sample 200 random edges; query Google Distance Matrix for each midpoint at next-Monday 8 AM IST peak.
2. `raw_c = (live − ff) / ff`, capped at 2.0.
3. **Propagation shortcut** (lines 232-261): adds a "SUPER_SOURCE" node connected to every sampled endpoint with length 0, runs single-source Dijkstra, and assigns each graph node the c_ij of its nearest sampled node along the *road* graph. Edges between two nodes get the **mean** of their inherited c values.

**Real/mock**: SPEED_PROXY = deterministic from OSM tags (not traffic). GOOGLE_MAPS = real but sampled at only 0.08 % coverage and smeared by graph-distance nearest-neighbour.

**Shortcut**: Mode B caps API calls aggressively and interpolates the remaining 99.92 % of edges.

### 3.5 `code/04_instance_generator.py` (221 LOC)

**Parameters**
| Name | Value | Meaning |
|---|---|---|
| `N_CUSTOMERS` | 30 | Orders per dispatch window |
| `K_RIDERS` | 30 | Fleet size (**matches N** — each rider could take one order) |
| `VEHICLE_CAPACITY` | 2 | Q (units per rider) |
| `DEMAND_MIN / MAX` | 1 / 2 | q_i range |
| `TW_WINDOW_LENGTH` | 15 | `l_i − e_i` (min) |
| `LAMBDA` | `[0.40, 0.40, 0.20]` | λ₁ time, λ₂ risk, λ₃ congestion |
| `RANDOM_SEED` | 42 | |

**Three depots** (hard-coded lat/lon)
- `DarkStore_Majestic` (12.9766, 77.5713)
- `DarkStore_Indiranagar` (12.9784, 77.6408)
- `DarkStore_Koramangala` (12.9279, 77.6271)

**Flow**
1. Snap each depot to nearest OSM node (`ox.distance.nearest_nodes`).
2. Sample 30 customer nodes uniformly at random from all 99,572 graph nodes.
3. For each customer, compute shortest-path travel time to **each depot** using `nx.shortest_path(weight="t_ij")` and assign to the nearest.
4. **Rolling-horizon time windows**: `e_i = current_tick`, `l_i = e_i + 15`, then `current_tick += rng.randint(1, 3)` (1-3 min between drops).
5. Custom `get_edge_attr` (lines 71-91) explicitly picks the **minimum-`t_ij` parallel edge** (vs. default index-0) — a quiet bug fix over vanilla `MultiDiGraph` handling.

**Real/mock**: Topology real, customers randomly sampled (synthetic). Demand uniform in {1,2}.

**Shortcut**: 30 Dijkstra calls per depot-customer pair (3 × 30 = 90 SPs on a 99.6 k-node graph). Works, but there's no caching; each run recomputes.

### 3.6 `code/05_validate_instance.py` (235 LOC)

Runs 9 sanity checks. **All passed** at last run (see `run_02.log`). Nothing synthetic.

### 3.7 `code/06_export_geojson.py` (177 LOC)

Exports to `web/static/data/`. **Bug**: references `instance["depot"]` (singular) at line 77 but the instance now produces `instance["depots"]` (plural, list). This script will crash if run — not part of the live pipeline.

### 3.8 `code/07_solver_ga.py` (279 LOC) — "Production" solver

**GA hyper-parameters**
| Name | Value |
|---|---|
| `POP_SIZE` | 50 |
| `GENS` | 100 |
| `CX_PROB` | 0.8 |
| `MUT_PROB` | 0.2 |
| `ELITE_SIZE` | 2 |
| `CAPACITY` | **3** (overrides the instance's Q=2 — *inconsistency*) |
| Tournament size | 3 |

**Cost matrix construction** (`_prepare_matrices`, lines 54-82)
- For the 33 relevant nodes (3 depots + 30 customers), runs **three separate** `nx.single_source_dijkstra_path_length` calls per node: one each for `t_ij`, `r_ij`, `c_ij` weights.
- **99 Dijkstras total** on a 99.6 k-node graph — dominates runtime.
- Honest comment at line 60: *"In a real large-scale scenario, we'd use a contraction hierarchy or pre-distilled graph."*

**Fitness** (lines 156-179)
```
score = λ1·Σt_ij + λ2·Σr_ij + λ3·Σc_ij + 100 · Σ TW_violation_minutes
```
- Soft time-window violation with a **constant** penalty multiplier of 100.
- Late arrival triggers `tw_violation += (t − l_i)`; early arrival causes the solver to *wait* (`current_time = e_i`) — no penalty.

**Split procedure** (lines 125-154)
- Greedy capacity-based splitting of the giant tour. Author acknowledges optimal Bellman-Ford split exists but is "faster for GA iterations".

**Observed output** (`data/05_solution.json`)
- 17 routes across 3 depots, score = **394.28**.
- Routes have 1-3 customers each (consistent with Q=2 demand, Q=3 capacity).
- `total_risk = 0.0` on every route in `08_behavioral_analysis.csv` — because `r_ij > 0` exists on only 137 edges and shortest-path on `r_ij` finds a zero-risk path through side streets for every depot→customer pair.

**Real/mock**: Uses real graph data; GA is genuine but single-objective (weighted sum, not Pareto).

**Shortcut**: Skips exact split, skips local search, uses the same fitness evaluation both for selection and elitism. No early stopping.

### 3.9 `code/07_verify_solution.py` (63 LOC)
Lightweight: checks all customers visited, depot book-ending, no double visits. **Does not re-evaluate TW feasibility** (says "Conceptual" on line 57).

### 3.10 `code/08_behavioral_model.py` (98 LOC) — **fully synthetic**

**Parameters**
- `compliance ∼ Beta(5, 2)` (mean ≈ 0.71, biased toward compliance).
- `realized_tt = path_tt · (0.8 + 0.4·(1-c))` → non-compliant rider is *faster*.
- `realized_risk = path_risk · (1.0 + 2.0·(1-c))` → non-compliant rider incurs 3× risk at c=0.

**Flow**
- For each of the 17 routes × 10 simulated riders = **170 rows** in `08_behavioral_analysis.csv`.
- Between two consecutive route nodes, uses `nx.shortest_path_length(weight="t_ij"|"r_ij"|"c_ij")`.
- **`total_risk = 0.0` everywhere in the output CSV** — a downstream symptom of the r_ij sparsity issue.

**Real/mock**: Behavioral model is a **pure Monte Carlo** with no empirical calibration. The multipliers (0.8+0.4x, 1+2x) are hand-picked constants.

### 3.11 `code/08_run_real_benchmarks.py` (270 LOC) — **largely fabricated**

Builds figures `6_3d_pareto.png`, `7_smax_fatigue.png`, `8_stw_scatter.png`.

**Fabricated edge attributes** (lines 42-58)
```python
speed      = 40.0 if hw in ('primary','secondary') else 20.0
base_risk  = 0.8 if hw in ('primary','trunk')      else 0.2
data['r_ij'] = clip(base_risk + N(0, 0.15), 0.05, 1.0)
base_cong  = 0.9 if hw in ('primary','motorway')   else 0.3
data['c_ij'] = clip(base_cong + N(0, 0.2),  0.1, 1.0)
data['h_ij'] = 1 if hw in ('residential','living_street') else 0
```
→ **Risk is a function of highway class + Gaussian noise**, not of any accident data.

**MILP solver** (lines 99-134)
- Uses `pulp.PULP_CBC_CMD(timeLimit=15, msg=0)`. Returns `(elapsed, NaN)` whenever the solver can't prove optimality in 15 s (very common for N ≥ 12 with MTZ sub-tour constraints).
- **K=3, Q=2, S_max=5.0, H_cap=8** (different from instance!).
- Adds the novel constraints from the proposal's rewrite: `Σ R·x ≤ 5` and `Σ H·x ≤ 8` per route.

**STW penalty** (line 242)
```python
lateness = max(0, t - 30.0 + random.normalvariate(0, 5))
pt_tau   = math.exp(0.12 * lateness) - 1.0
```
→ The "lateness" is synthesized from noise; the `math.exp(0.12 · L)` factor is a **hand-chosen curve**, not derived from any SLA schedule.

**Note**: line 242 uses `math.exp` but `import math` only happens at line 263 (after `run_3d_pareto`); the import order works because `math` is needed inside `matrices()` which is called before the scatter — but `run_stw_penalty_scatter` would fail if called before `generate_routing_matrices`. Fragile.

### 3.12 `code/09_dashboard_visualization.py` (121 LOC)

**Simulated heatmap** (lines 70-77) — hardcoded hotspots:
```python
hotspots = [
    [12.9716, 77.5946, 0.8],  # MG Road
    [12.9279, 77.6271, 0.9],  # Koramangala
    [12.9141, 77.5891, 0.7],  # Jayanagar
    [13.0285, 77.5896, 0.6],  # Hebbal
]
```
Developer's own comment (line 69): *"Since our iRAD data conversion was sparse in the snippet, we simulate hotspots near central areas."* — **explicit admission**.

### 3.13 `code/09_drl_agent.py` (203 LOC) — **toy synthetic demo**

**Architecture**
- `SimplePointerNetwork`: `fc1(N*3→128) → fc2(128→128) → head(128→N)`, softmax over a feasibility mask. Not a true attention-based Pointer Network.
- `SAVRPTW_Env`: State = `[curr_node_onehot | unvisited_mask | remaining_cap]`.
- Reward: `-(0.6·t_cost + 0.4·r_cost)`; penalty `-100` if capacity violated.
- Trainer: REINFORCE, `lr=0.005`, `γ=0.99`, `epochs=500`-1000, baseline = (returns − mean) / std.

**Data** (lines 195-199)
```python
T_mat = np.random.uniform(1, 10,  (20,20))
R_mat = np.random.uniform(0.01, 1.5, (20,20))
```
→ **DRL is trained on pure uniform noise**, never touched by any real graph.

**It is not used by `12_all_cities_figures.py`.** Its "DRL O(1)" scalability line in figures is synthesized as `ga_time × 0.05` (see 3.15).

### 3.14 `code/10_spatial_overlays.py` (101 LOC)

- Downloads OSM graph for Bengaluru (2 km radius), imputes `time` and `risk` per edge using the same highway-type + Gaussian rule as 3.11.
- Picks random start/end > 10 min apart; plots fastest path (by `time`) in red vs safest path (by `−log(1 − risk)`) in green.
- **Heatmap** (lines 76-90): filters for major roads and generates `danger = np.random.uniform(0.7, 1.0)` — comment line 79: *"fake up the iRAD empirical limits"*.

### 3.15 `code/12_all_cities_figures.py` (423 LOC) — main figure driver

**Global parameters**
| Name | Value |
|---|---|
| `RADIUS` | 1,500 m per city |
| `N_SMALL / N_CONV / N_LARGE` | 12 / 15 / 20 |
| `S_MAX_LIMIT` | 5.0 |
| `H_CAP` | 8 |
| `Q_CAPACITY` | 2 |
| `K_FLEET` | 5 |
| MILP time limit | 12 s |
| GA generations | 30-50 per figure |

**Faked DRL scalability** (line 198)
```python
rows.append({"N": n, "Time (s)": dt*0.05, "Algorithm": "DRL Agent (O(1))"})
```
→ The "DRL O(1)" line in `1_scalability.png` is literally **GA time × 0.05**. No neural network is instantiated in this file.

**Pareto frontier** (lines 207-221)
- 30-point λ sweep from 0.05 to 0.95, 40 GA generations per point. Points plotted directly (not a genuine non-dominated front — it's the per-run best, with no filtering).

**STW penalty** (lines 306-322)
- Same formula as 3.11: `lateness = max(0, t − 25 + N(0, 4))`, `penalty = exp(0.12·L) − 1`. Fully synthetic.

**Hierarchy plot** (lines 259-285)
- Walks the GA route and counts highway categories of the underlying shortest path. The only plot that uses *real* per-edge data (OSM `highway` tags).

**Fleet risk boxplot** (lines 288-303)
- Draws **random sub-sequences** of 5 nodes per simulated rider and sums their pairwise R values. Not a real routing outcome.

**Folium maps** (lines 325-366)
- Real OSM topology, but with fabricated `r_ij` from 3.11.

### 3.16 `code/11_sumo_integration.py` (75 LOC) — placeholder

- Attempts to `import traci, sumolib`; swallows `ImportError`.
- `generate_sumo_network` is **empty (`pass`)**.
- `simulate_routes` wraps everything in `try/except`; the except branch prints *"Bypassing for theoretical workflow tracking validation."* — meaning if SUMO isn't installed with a prepared `.sumocfg`, it silently returns success-ish output.
- Not called from anywhere.

### 3.17 `code/rewrite_simplified.py`, `code/rewrite_proposal_v3.py`

Not part of the data/solver pipeline. They programmatically rewrite `SafetyAwareRoutingProposal.tex` (the LaTeX document).

---

## 4. Mock / Simulated / Fabricated Data Inventory

| # | Location | What is fabricated | Evidence |
|---|---|---|---|
| 1 | `02_irad_risk.py` lines 59-100 | **All 150 accidents** (Gaussian mixture around 4 Bengaluru centres) | `data/02_irad_metadata.json` → `"data_source": "synthetic"` |
| 2 | `03_congestion.py` lines 63-117 (default mode) | `c_ij` derived from OSM speed limits (not real traffic) | `data/03_congestion_metadata.json` → `"mode": "SPEED_PROXY"` |
| 3 | `04_instance_generator.py` lines 142-170 | Customer locations (random sample), demands (uniform 1-2), time windows (rolling arrival ticks) | `data/04_vrptw_instance.json` |
| 4 | `08_behavioral_model.py` lines 40-65 | Compliance ∼ Beta(5,2), tt/risk multipliers = hand-picked constants | `data/08_behavioral_analysis.csv` |
| 5 | `08_run_real_benchmarks.py` lines 42-58 | `r_ij`, `c_ij`, `h_ij` = **f(highway class) + Gaussian noise** | n/a (applied in-memory only) |
| 6 | `08_run_real_benchmarks.py` line 241 | STW `lateness = max(0, t − 30 + N(0, 5))` | synthetic noise distribution |
| 7 | `09_dashboard_visualization.py` lines 70-77 | 4 hardcoded Bengaluru heat-map points (author's own comment: *"we simulate hotspots"*) | `data/09_final_dashboard.html` |
| 8 | `09_drl_agent.py` lines 195-199 | **DRL training matrices = `np.random.uniform`** | main block at EOF |
| 9 | `10_spatial_overlays.py` lines 30-35 | Per-edge time & risk synthesized from highway class + N(0, 0.15) | inline only |
| 10 | `10_spatial_overlays.py` lines 76-90 (comment: *"fake up the iRAD empirical limits"*) | Heatmap `danger = uniform(0.7, 1.0)` on major roads | `figures/10_danger_density_heatmap.html` |
| 11 | `11_sumo_integration.py` | Entire SUMO simulation is a stub with a no-op generator | — |
| 12 | `12_all_cities_figures.py` lines 67-83 | Same fabricated `r_ij`, `c_ij`, `h_ij` **for all 6 cities** | `figures/<City>/*.png` |
| 13 | `12_all_cities_figures.py` line 198 | DRL scalability point = `ga_time × 0.05` | `figures/*/1_scalability.png` |
| 14 | `12_all_cities_figures.py` lines 288-303 | Fleet risk = random sub-sequences of 5 nodes | `figures/*/6_fleet_risk.png` |
| 15 | `12_all_cities_figures.py` lines 306-322 | STW lateness synthesized as noise; penalty = exp(0.12·L)−1 | `figures/*/7_stw_penalty.png` |

**Net**: The two experimental artifacts that use **any real-world safety data** are the Bengaluru synthetic iRAD (150 points) and the OSM topology itself. Everything else downstream of those is derived or fabricated.

---

## 5. Shortcuts Taken for Processing Time

| # | Where | Shortcut | Implication |
|---|---|---|---|
| 1 | `01_osm_graph.py` | 10 km radius around MG Road (vs. full Bengaluru metro) | Outer suburbs absent |
| 2 | `02_irad_risk.py` | Only 150 synthetic accidents | r_ij ≠ 0 on only 137 / 256,887 edges (0.053 %) |
| 3 | `03_congestion.py` GOOGLE_MAPS | Samples at most `MAX_API_EDGES = 200` of 256,887 edges; interpolates rest via super-source Dijkstra | 99.92 % of c_ij values come from nearest-neighbour extrapolation |
| 4 | `04_instance_generator.py` | 30 customers × 3 depots (90 depot-customer SPs only) | Tiny instance vs. real q-commerce dispatch windows |
| 5 | `07_solver_ga.py` lines 54-82 | 99 `single_source_dijkstra_path_length` calls on full 99 k-node graph (no contraction hierarchy, no A*). Author comment acknowledges this | Slowest step of the "production" pipeline |
| 6 | `07_solver_ga.py` line 126 | Greedy split instead of optimal Bellman-Ford split | Can miss globally better route splits |
| 7 | `07_solver_ga.py` | Fitness re-computed from scratch on every individual, every generation, no caching | O(pop × gens × routes × nodes) = 50·100·17·3 = 255 k evaluations |
| 8 | `08_run_real_benchmarks.py` | MILP `timeLimit=15s` → typically returns NaN, falls back to "freezes past N=20" narrative | "Exact" baseline is actually *best-effort-in-15s* |
| 9 | `08_run_real_benchmarks.py` | Radius = 1,200-1,500 m per city (vs. full metro) | Figures compare tiny subgraphs, not actual cities |
| 10 | `08_run_real_benchmarks.py` lines 42-58 | Risk values *computed from OSM highway tags* instead of downloaded iRAD files | Eliminates any true per-city variation in the safety signal |
| 11 | `09_drl_agent.py` | Not invoked by the figures pipeline; its DRL numbers in plots are **scaled GA time** | DRL "O(1) inference" claim unsupported by measurement |
| 12 | `12_all_cities_figures.py` | `N ∈ {12, 15, 20}` for all figures, GA gens ∈ {30, 40, 50} | All-pairs matrix is a 20×20 at most |
| 13 | `12_all_cities_figures.py` Pareto | 30 λ-sweep points plotted directly; no dominance filtering | Displayed "Pareto" curve can contain dominated points |
| 14 | `11_sumo_integration.py` | SUMO never actually runs | "Empirical validation" claim in proposal is aspirational |

---

## 6. Consolidated Parameter Table

### Physical / geographic
| Symbol | Value | Where | Meaning |
|---|---|---|---|
| centre | (12.9716, 77.5946) | `01_osm_graph.py` | Bengaluru MG Road |
| radius | 10 km | `01_osm_graph.py` | Pipeline A graph extent |
| radius | 1,500 m | `12_all_cities_figures.py` | Per-city figure extent |
| BBox | lat [12.834, 13.144], lon [77.460, 77.784] | `utils.py` | Bengaluru-only filter |

### Risk model
| Symbol | Value | Meaning |
|---|---|---|
| `SYNTHETIC_N` | 150 | # synthetic accidents |
| `YEARS_SPAN` | 5 | accidents/km-year normaliser |
| Severity weights | fatal=4, grievous=3, minor=2, damage=1 | linear encoding |
| Cluster σ | 0.008° ≈ 880 m | Gaussian spread of synthetic points |
| Normalization | log1p → min-max [0,1] | anti-outlier |
| r_ij (fig pipeline) | 0.8·1{primary,trunk} + 0.2·1{else} + N(0, 0.15) clipped [0.05,1] | fabricated per-edge risk |

### Congestion
| Symbol | Value | Meaning |
|---|---|---|
| `CONGESTION_MODE` | SPEED_PROXY | default; no API needed |
| `MAX_API_EDGES` | 200 | GOOGLE_MAPS sample cap |
| `API_SLEEP_SEC` | 0.05 | rate limit |
| c_ij (speed proxy, arterial) | 1 − speed_norm | ≥ 0.5 for slow majors |
| c_ij (speed proxy, residential) | 0.1 + 0.1·speed_norm | ≤ 0.2 always |
| c_ij (fig pipeline) | 0.85·1{primary,motorway} + 0.3·1{else} + N(0, 0.2) | fabricated |

### Instance (production, `04_instance_generator.py`)
| Symbol | Value | Meaning |
|---|---|---|
| N_CUSTOMERS | 30 | Orders in dispatch window |
| K_RIDERS | 30 | Fleet |
| Q (vehicle capacity) | 2 | Units per rider |
| demand q_i | U{1, 2} | |
| TW length (l_i − e_i) | 15 min | strict SLA |
| Inter-arrival between drops | U{1, 2, 3} min | rolling-horizon ticks |
| Depot TW `[e_0, l_0]` | [0, 120] min | 2-hour shift |
| λ₁ (time) | 0.40 | |
| λ₂ (risk) | 0.40 | |
| λ₃ (congestion) | 0.20 | Σλ = 1 |
| Seed | 42 | |

### GA solver (`07_solver_ga.py`)
| Symbol | Value |
|---|---|
| POP_SIZE | 50 |
| GENS | 100 |
| CX_PROB | 0.8 |
| MUT_PROB | 0.2 |
| ELITE_SIZE | 2 |
| Tournament k | 3 |
| CAPACITY | **3** (inconsistent with instance Q = 2) |
| TW penalty multiplier | 100 (per minute late) |
| Crossover | Ordered crossover (OX1) |
| Mutation | Swap |

### Figure-pipeline MILP (`08_run_real_benchmarks.py`, `12_all_cities_figures.py`)
| Symbol | Value | Meaning |
|---|---|---|
| K | 3 | fleet |
| Q | 2 | capacity |
| S_max | 5.0 | cumulative risk budget per route |
| H_cap | 8 | residential-edge budget |
| Sub-tour elim | MTZ (big-M) | |
| Solver | `PULP_CBC_CMD` | CBC open-source |
| Time limit | 12-15 s | triggers NaN often |

### GA (figure pipeline, `12_all_cities_figures.py`)
| Symbol | Value |
|---|---|
| pop_size | `max(20, N)` |
| gens | 30-50 (per figure) |
| mutation | 0.1 |

### Behavioral model (`08_behavioral_model.py`)
| Symbol | Value |
|---|---|
| Compliance prior | Beta(5, 2) — mean ≈ 0.71 |
| Realized travel time | `path_tt · (0.8 + 0.4·(1−c))` |
| Realized risk | `path_risk · (1.0 + 2.0·(1−c))` |
| Riders per route | 10 |
| Total rows | 17 × 10 = 170 |

### DRL (`09_drl_agent.py`) — **unused by figures**
| Symbol | Value |
|---|---|
| hidden_dim | 128 |
| lr | 0.005 |
| γ (discount) | 0.99 |
| Optimizer | Adam |
| Baseline | (returns − mean) / std |
| Reward weights | L1 = 0.6, L2 = 0.4 |
| Epochs | 500 (smoke) / 1000 (train_agent default) |
| N | 20 (hard-coded) |
| Q_cap | 3 (**inconsistent** again) |
| Training T_mat | U(1, 10) random |
| Training R_mat | U(0.01, 1.5) random |

### STW penalty (figure pipeline)
`lateness = max(0, t − 25_or_30 + N(0, 4_or_5))`
`P_L(τ) = exp(0.12 · lateness) − 1`

---

## 7. Inconsistencies & Bugs Found During Line-by-Line Review

1. **Capacity Q mismatch**: instance (Q=2) vs. production GA (CAPACITY=3) vs. DRL (Q_cap=3). The GA will always over-pack.
2. **K mismatch**: instance K=30 vs. figure MILP K=3 vs. figure GA "fleet" K_FLEET=5.
3. **Bbox hard-coded to Bengaluru in `utils.py`** but used as a filter inside `02_irad_risk.py` — if a real iRAD CSV covered Delhi or Mumbai, every record would be dropped silently.
4. **`06_export_geojson.py` line 77** references `instance["depot"]` (singular, pre-multi-depot API); `04_instance_generator.py` writes `instance["depots"]` (plural). Script is broken.
5. **`r_ij` sparsity** (137 / 256,887 edges = 0.053 %) → Dijkstra on `r_ij` almost always returns 0 → `08_behavioral_analysis.csv` shows `total_risk = 0.0` for every one of the 170 rows.
6. **`08_run_real_benchmarks.py` line 263** (`import math`) comes *after* its use at line 242 inside `run_stw_penalty_scatter`. Works only because `__main__` calls `run_3d_pareto` first (which doesn't use `math`) and the top-level `import math` executes before `run_stw_penalty_scatter`.
7. **`11_sumo_integration.py`** silently "succeeds" on failure — it catches `Exception` and prints *"Bypassing…"*. Calling code can't distinguish.
8. **"DRL O(1)" plot is `ga_time × 0.05`** (`12_all_cities_figures.py` line 198). This is a hard-coded hack, not a measurement.
9. **Pareto plot plots all samples**, not the non-dominated front. Can include dominated points.
10. **Greedy split in GA** ignores depot assignment: `current_route` is built until capacity exceeded, but the same route can contain customers from different depots; only the first customer's `assigned_depot` is used (line 169).

---

## 8. What Is Real vs. What Is Fake — At a Glance

```
                REAL                                 FABRICATED
 ┌──────────────────────────────┐     ┌────────────────────────────────────┐
 │ OSM road topology (OSMnx)    │     │ iRAD accident locations & sev.      │
 │ OSM speed/length tags        │     │ Congestion c_ij (speed proxy OR     │
 │ Bengaluru 99,572-node graph  │     │   200-edge GM sample interpolated)  │
 │ Customer node snap (nearest) │     │ Customer lat/lon, q_i, TW           │
 │ Pipeline A GA (runs end-end) │     │ Figure-pipeline r_ij, c_ij, h_ij    │
 │ CBC MILP (runs, may time out)│     │ DRL "O(1)" scalability point        │
 │ Pipeline B Folium maps       │     │ Behavioral compliance distribution  │
 │ GraphML I/O, normalization   │     │ STW lateness Gaussian noise         │
 │                              │     │ Dashboard heatmap (4 hardcoded pts) │
 │                              │     │ SUMO simulation (stub only)         │
 │                              │     │ Fleet risk boxplot (random subseqs) │
 └──────────────────────────────┘     └────────────────────────────────────┘
```

---

## 9. Recommendations (for the thesis reviewer)

1. **Label every figure** with the data source used. If `r_ij = f(highway)+noise`, state it.
2. **Unify Q, K, and solver parameters** across all modules — a single `config.yaml` or `parameters.py`.
3. **Either integrate or remove** `11_sumo_integration.py` and `09_drl_agent.py` — their headline claims (SUMO validation, DRL O(1)) currently don't run inside the figure pipeline.
4. **Replace the "DRL O(1)" proxy** (`ga_time × 0.05`) with either real inference timings from `09_drl_agent.py` or drop the line from `1_scalability.png`.
5. **Expand iRAD coverage**: the real CSV path (`data/irad_accidents.csv`) is referenced but not populated. Without it, every claim about empirical road safety rests on 150 Gaussian-mixture points.
6. **Generalize `BENGALURU_BBOX`** to the active city or remove the bbox filter from `02_irad_risk.py`.
7. **Fix the Pareto plot** to return the non-dominated front only.
8. **Plot solver metrics from the actual production GA** (`07_solver_ga.py`) instead of the synthetic matrices in `12_all_cities_figures.py` — the production solver's 394.28 score on 30 real customers is more defensible than thousands of λ-swept toy runs.
