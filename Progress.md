# Implementation Progress: Safety-Aware Vehicle Routing in Quick Commerce

This document tracks the execution of the research proposal. The implementation is divided into five phases, following a socio-technical approach to urban logistics.

## 📈 Research Roadmap & Progress

| Phase | Task | Status | Deliverable |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Foundations & Data Acquisition** | 🟢 Complete | Literature Review, OSM Graph, iRAD Risk, Congestion, VRPTW Instance |
| **Phase 2** | **Graph Construction & Safety Enrichment** | ⚪ Pending | Safety-Enriched GraphML File |
| **Phase 3** | **Mathematical Modeling & Solver Design** | ⚪ Pending | Python VRP Solver (GA/ACO) |
| **Phase 4** | **Behavioral Incentive Integration** | ⚪ Pending | Behavioral Weighting Module |
| **Phase 5** | **Simulation, Validation & Analysis** | ⚪ Pending | SUMO Simulation Results & Thesis |

---

## 🛠️ Detailed Implementation Plan

### Phase 1: Foundations & Data Acquisition (Month 1-2)
*   [x] **Literature Review**: Synthesize VRPTW, safety-aware routing, and behavioral logistics.
*   [x] **Mathematical Formulation**: Define the multi-objective function $\min (\lambda_1 T + \lambda_2 R + \lambda_3 C)$.
*   [x] **Data Sourcing**:
    *   [x] `code/01_osm_graph.py` — OSM road network for Kharagpur → `t_ij`, `d_ij`.
    *   [x] `code/02_irad_risk.py` — iRAD accident data → `r_ij` (dual-mode: real CSV or synthetic).
    *   [x] `code/03_congestion.py` — congestion index `c_ij` (SPEED_PROXY or GOOGLE_MAPS API).
    *   [x] `code/04_instance_generator.py` — 20-customer VRPTW instance with time windows.
    *   [x] `code/05_validate_instance.py` — 9-check validation suite.
    *   [x] `code/run_pipeline.py` — master runner: `python run_pipeline.py`.
    *   [x] `code/requirements.txt` — pinned Python dependencies.
    *   [ ] Download real iRAD CSV (place at `data/irad_accidents.csv` when available).
    *   [ ] Add `GOOGLE_MAPS_API_KEY` to `.env` when available.

### Phase 2: Graph Construction & Safety Enrichment (Month 3)
*   [ ] **Map Extraction**: Use `osmnx` to download the `drive` network for the target urban area.
*   [ ] **Safety Layering**: 
    *   Clean iRAD coordinates and perform a spatial join with OSM edges.
    *   Compute $r_{ij}$ (Risk Probability) for every road segment based on accident severity and frequency.
*   [ ] **Static Verification**: Visualize the road network highlighting "High-Risk Zones" in Red.

### Phase 3: Mathematical Modeling & Solver Design (Month 4)
*   [ ] **Instance Generation**: Create synthetic delivery orders with strict 10-30 min time windows.
*   [ ] **Algorithm Implementation**:
    *   **Baseline**: Standard Dijkstra for "Fastest Path."
    *   **Proposed**: Genetic Algorithm (GA) or Ant Colony Optimization (ACO) to solve the Multi-Objective VRPTW.
*   [ ] **Objective Tuning**: Test various $\lambda$ weights to see how routes change when safety is prioritized over speed.

### Phase 4: Behavioral Incentive Integration (Month 5)
*   [ ] **Incentive Modeling**: Define a "Rider Compliance" factor based on estimated earnings vs. safety-route length.
*   [ ] **Decision Layer**: Implement a "Rider Acceptance" probability module where riders might choose to ignore the safe route if the time penalty is too high (>20%).

### Phase 5: Simulation, Validation & Analysis (Month 6)
*   [ ] **SUMO Integration**: Convert the graph to a `.net.xml` file for SUMO.
*   [ ] **Conflict Analysis**: Run simulations of 100 delivery trips. Compare:
    *   **Fastest Routing**: Number of near-misses, average speed, delivery time.
    *   **Safety-Aware Routing**: Reduction in risk exposure vs. increase in delivery time.
*   [ ] **Final Thesis/Paper**: Document findings and policy recommendations for q-commerce platforms.

---

## 🚀 Technical Stack
*   **Language**: Python 3.10+
*   **Graph Processing**: OSMnx, NetworkX
*   **Geospatial**: GeoPandas, Shapely
*   **Optimization**: PyVRP or custom Meta-heuristics
*   **Simulation**: SUMO (Simulation of Urban MObility)
*   **Visualization**: Matplotlib, Folium (for interactive maps)
