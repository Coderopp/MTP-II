# SA-VRPTW — Remaining Tasks

**Updated:** 2026-04-22 (end of Task #13).
**Branch:** `refactor/modularize` (uncommitted).
**State:** pipeline is end-to-end runnable on synthetic instances; 54/54 tests green. To produce paper-ready results, the blockers below must close in order.

---

## Dependency graph

```
        ┌────────── #18 Hyderabad data gap (USER DECISION)
        │
        ▼
   #17 Blinkit coord cross-validation        ┐
                                             ├─►  full 6-city experiment grid
   #16 Alternative risk calibration source   │
        │                                    │
        ▼                                    │
   #9  BASM implementation  ─────────────────┘
        │
        ▼
   #14 Paper figures
        │
        ▼
   #15 LaTeX rewrite

   #19 Legacy parity harness  — optional, parallel
```

---

## Task #16 — Alternative risk-calibration source  *(BLOCKER)*

### Why this is first

`r_ij` per edge is the one quantity we still don't have a defensible source for.  Without it, `savrptw.risk.basm.attach_risk` intentionally raises `NotImplementedError` — the generator can produce synthetic test Instances but not real-city Instances.  Every downstream task (#9, #14, #15) is blocked on closing this.

### Goal

Choose one calibration route, document it in `docs/BASM_CALIBRATION.md`, and populate `conf/risk/basm_v1.yaml` with sourced constants (`lambda_class`, `severity_multiplier`).  Must be fully reproducible — every number in the YAML traces back to a named public source.

### Candidate sources (to evaluate)

| # | Source | Cost | Granularity | Risk |
|---|---|---|---|---|
| A | MoRTH "Road Accidents in India 2022" **PDF aggregates** | free PDF | per-state / per-road-class / per-severity | tables must be digitised; per-city numbers only via state apportionment |
| B | WHO Global Status Report on Road Safety 2023 | free PDF | country-level only | too coarse → needs a spatial-distribution proxy |
| C | Delhi Traffic Police / Bengaluru Traffic Police crash dashboards | free, public | per-city point data, sometimes polygon | uneven coverage across our 6 cities |
| D | **OSM-based black-spot proxy**: signal density, pedestrian-crossing density, intersection-betweenness | free, already in graph | per-edge surrogates | not "real" crash data — but *is* defensible if calibrated to an aggregate |
| E | Published academic datasets (Mohan et al., Tiwari et al.) | free PDF tables | per-road-class multipliers | use as multiplier only, needs a base rate from elsewhere |

### Recommended hybrid (to propose to reviewers)

`base rate from (A) MoRTH 2022 city aggregates (fatal+grievous / vehicle-km)`
`× class multiplier from (E) Mohan 2017`
`× edge-local proxy from (D) OSM features (signals, crossings, betweenness)`

→ Each edge gets `r_ij = base · class_mult · proxy_edge_weight`, calibrated so the city-total expected fatalities/year matches the MoRTH aggregate within ±25%.  The proxy rescales *within* a class without changing the class total — defensible even if reviewers challenge the proxy.

### Work breakdown

1. Dig out the MoRTH 2022 tables we need (per-city fatal+grievous counts; per-road-class split).  One day.
2. Read Mohan 2017 (and 1–2 similar Indian urban-safety studies) and extract class multipliers with citation anchors.  ½ day.
3. Write `docs/BASM_CALIBRATION.md` — section per source, table per number, exact equation for `r_ij`.  1 day.
4. Fill `conf/risk/basm_v1.yaml` with `lambda_class`, `severity_multiplier`, `source: "morth_mohan_osm_proxy_v1"`.  ½ day.
5. Implement the calibration-computation function (see Task #9).  (In Task #9.)

### Inputs needed from you

- Confirmation you're OK with the hybrid recommendation, OR pick a different source.
- Any Indian-urban-safety references you already have — saves literature hunt.

### Acceptance criteria

- [ ] Every constant in `conf/risk/basm_v1.yaml` has a `source:` string pointing to a specific table/equation in `docs/BASM_CALIBRATION.md`.
- [ ] `docs/BASM_CALIBRATION.md` cites the source PDFs by URL and page number.
- [ ] Cross-validation test asserts `|computed_fatalities_per_year − MoRTH_city_total| / MoRTH_city_total ≤ 0.25`.

### Estimated effort: **3 days**

---

## Task #18 — Hyderabad Blinkit data gap  *(USER DECISION)*

### Context

The 2026-04-09 KML snapshot has **zero Blinkit placemarks in the Hyderabad bbox**.  Confirmed by a regression test (`test_hyderabad_has_no_blinkit_in_snapshot`).

### Options

1. **Drop Hyderabad from primary experiments.**  Paper headline becomes "5 major Indian cities".  Minimal methodology change.  Simplest, most defensible.
2. **Use Zepto for Hyderabad only.**  The snapshot has 165 Zepto-tagged Hyderabad placemarks; clustering works the same way.  Paper must justify the per-city brand selection.  Slight methodology complication.
3. **Fetch a fresher KML** and hope Blinkit Hyderabad has been added upstream.  Gamble; breaks reproducibility unless we freeze *both* snapshots.
4. **Substitute Chennai** (136 Zepto placemarks, 9 Blinkit) — this still changes the brand-uniformity rule; flips the sentence "Bengaluru + Delhi + Gurugram + Mumbai + Pune + Chennai".

### Recommended: option 1

The uniformity gain is worth one city.  A paragraph in the methodology says "Blinkit's KML snapshot does not cover Hyderabad as of 2026-04-09; we confine single-brand experiments to 5 cities and leave multi-brand extension to future work".

### Work if option 1 chosen

1. Delete `conf/city/hyderabad.yaml`, or keep it with a `disabled: true` flag (preferred for future extension).
2. Update `conf/experiment/default.yaml` city list.
3. Update regression test: `test_hyderabad_has_no_blinkit_in_snapshot` stays — documents the rationale.

### Inputs needed from you

Pick option 1/2/3/4.

### Acceptance criteria

- [ ] `conf/experiment/default.yaml` explicitly lists the final city set.
- [ ] `docs/FORMULATION.md` §10.4 updated to match.
- [ ] Paper methodology section names the excluded/substituted city and why.

### Estimated effort: **½ day once decided**

---

## Task #17 — Cross-validate Blinkit snapshot (N=50 coords)

### Goal

Per FORMULATION.md §10.4: confirm a random 50-store sample resolves to a real Blinkit storefront within 50m using OSM Nominatim (free, no API key).

### Work breakdown

1. `tools/crossval_darkstores.py`:
   - Read committed KML.
   - Sample 50 random placemarks with a fixed seed.
   - For each, query OSM Nominatim `reverse` endpoint (respect 1-req/s rate limit via `time.sleep(1)`).
   - Also query Nominatim `search` for "Blinkit, <city>" and pick the closest result.
   - Compute great-circle distance between claimed coord and Nominatim result.
2. Emit `data/validation/darkstore_crossval.json` with per-store `{claimed_lat, claimed_lon, nominatim_lat, nominatim_lon, distance_m, within_50m: bool}`.
3. Summary row: `n=50, within_50m_count, within_50m_rate`.
4. New test: `tests/test_crossval.py::test_acceptance_rate` asserts `within_50m_rate >= 0.90`.

### Edge cases

- Nominatim may not find a match for a store (especially freshly opened ones). Treat "no match" as a miss but log separately; ≥ 90% match rate is the threshold.
- Rate-limit respectfully; whole run will take ~1 minute.
- Cache results in the JSON so re-runs are free.

### Inputs needed

Nothing — Nominatim is free.

### Acceptance criteria

- [ ] `data/validation/darkstore_crossval.json` exists and is committed.
- [ ] ≥ 90% of sampled stores match within 50m.
- [ ] Test asserts the rate automatically so future KML refreshes auto-validate.

### Estimated effort: **1 day**

---

## Task #9 — BASM implementation

### Blocked by

Task #16 (must close first — that's where the calibration constants come from).

### Goal

Replace the `NotImplementedError` in `savrptw.risk.basm.attach_risk` with a real, calibrated implementation that populates `r_ij ∈ [0, 0.99]` on every edge of an enriched OSM graph.

### Concrete implementation sketch

```python
def attach_risk(G, cfg):
    if cfg.source == "uncalibrated":
        raise RuntimeError("uncalibrated — pick a source in conf/risk/basm_v1.yaml")
    lambda_class = cfg.lambda_class            # dict[highway -> per-km base]
    sev_mult = cfg.severity_multiplier          # dict[highway -> class mult]
    for u, v, k, data in G.edges(keys=True, data=True):
        hw = data["highway"]
        length_km = data["length"] / 1000.0
        proxy = _osm_proxy_weight(G, u, v, data)   # betweenness, signals, etc.
        base = lambda_class.get(hw, lambda_class["unclassified"])
        mult = sev_mult.get(hw, 1.0)
        r = min(cfg.r_max_clip, base * length_km * mult * proxy)
        data["r_ij"] = max(0.0, r)
    return G
```

### Work breakdown

1. Write `_osm_proxy_weight()` — combines intersection betweenness (precomputed), signal density (nodes tagged `highway=traffic_signals` within 100m), pedestrian crossings (tag `highway=crossing`).  Normalised to multiply base rate by ~[0.5, 2.0].
2. Write the city-aggregate cross-validation function:
   - Simulate annual vehicle-km per class (AADT × 365 × length-per-class).
   - Sum `r_ij × traversals_per_year` across the graph.
   - Assert within ±25% of MoRTH-reported city total.
3. Tests:
   - `test_r_ij_in_range` — every edge in [0, 0.99].
   - `test_calibration_matches_city_total` — uses a tiny synthetic "city" graph with known-input rates.
   - `test_arterial_riskier_than_residential` — sanity check on class multipliers.
4. Remove the regression guard that refuses uncalibrated runs (keep the `source: "uncalibrated"` guard; just add `source: "morth_mohan_osm_proxy_v1"` as a real option).

### Acceptance criteria

- [ ] `attach_risk` produces `r_ij ∈ [0, 0.99]` on every edge.
- [ ] Cross-validation passes within ±25% of MoRTH city total for all 5 (or 6) cities.
- [ ] Instance generator produces a real `Instance` end-to-end on Bengaluru in <60s.

### Estimated effort: **2 days after #16 closes**

---

## Task #14 — Paper figures

### Goal

Produce 8 figure families for the Wiley submission, each driven by the runner's JSON output.  No fabricated lines, no raw-sample Pareto plots (dominance filter mandatory).

### Figure inventory

| # | Name | What it plots | Data source |
|---|---|---|---|
| F1 | Scalability | MILP / GA / ALNS wall-clock vs N ∈ {20, 50, 100, 200}, per city; IQR bands over 10 seeds | runner JSON: `solve_wallclock_s` |
| F2 | Pareto frontier | ε-swept `(R̄, F₁)` with dominance filter; per city | runner JSON: `R_bar`, `F1`; sweep grid from `conf/experiment/default.yaml` |
| F3 | Convergence | Incumbent F₁ over iterations (GA: per-generation; ALNS: per-segment); shaded IQR over 10 seeds | solver.run_meta (requires per-iter logging — add) |
| F4 | Crash-survival Mann-Whitney | Box-and-whisker of E[crashes per 10k dispatches] across solvers × ε budgets; Wilcoxon + Bonferroni p-values annotated | `crash_mc.fleet_expected_crashes` |
| F5 | Behavioural sensitivity | 3×3 grid of (α, β); E[crashes] with 95% Wilson CI; colour-coded by compliance mean | `behavioral_mc.points` |
| F6 | S_max vs F₁ tradeoff | Per-route `R_route` accumulation; dashed line at R̄ | per-route analytics |
| F7 | Residential-hop heatmap | Per-city folium overlay: edge thickness ∝ how often it's traversed across 10 seeds' solutions | street graph + `solution.routes` |
| F8 | Cross-city comparison | Relative F₁ / E[crashes] across 5 cities; normalised | aggregate over runs |

### Work breakdown

1. `src/savrptw/viz/aggregate.py` — reads `data/results/*.json`, emits a tidy pandas DataFrame.
2. `src/savrptw/viz/plots.py` — eight plotting functions, each consumes the DataFrame and emits a PNG + a PDF.  No matplotlib rcParams hacks; everything driven by `conf/viz.yaml` (colour palette, font size).
3. `scripts/make_figures.py` — one-shot: regenerate every figure from `data/results/`.
4. Solver per-iteration logging — add optional `log_trace=True` to GA/ALNS that records `(it, best_F1)`; emitted to `solution.run_meta["trace"]`.
5. Tests (viz is notoriously under-tested — keep it honest):
   - `test_aggregate_handles_missing_keys` — tolerant to partial runs.
   - `test_pareto_dominance_filter` — fixed input produces the known non-dominated set.
   - Snapshot tests on figure metadata (title, axis labels, legend entries) — not pixels.

### Acceptance criteria

- [ ] 8 figure families emit PNG + PDF per city where city-specific.
- [ ] Every figure's underlying CSV is committed alongside the figure (reviewer reproducibility).
- [ ] No figure plots a line that isn't measured (`ga_time × 0.05`-style hacks are blocked by the code review gate).

### Estimated effort: **4 days after #9 closes**

---

## Task #15 — LaTeX rewrite

### Goal

Update `SafetyAwareRoutingProposal.tex` so the mathematics and narrative match the canonical formulation and the actual implementation.

### Scope of change (from FORMULATION.md §14)

12 distinct deviations from the current proposal — list reproduced here for the rewrite:

1. F₁ becomes **ETA deviation + STW penalty**, not `Σ t_ij x + Σ P_L`.
2. Risk moves from objective F₂ to **ε-constraint** (per-route R̄).
3. Congestion absorbed into `t_ij` via BPR, not a third objective.
4. Service time `s_i` added (2 min default, 4 min high-rise).
5. Max route duration `T_max = 35 min` added.
6. Multi-depot coupling (4–6) enforced — the GA now splits per-depot.
7. `r_ij` redefined as per-traversal crash probability in [0, 0.99), calibrated from MoRTH + Mohan + OSM proxy.
8. MTZ coefficient |C|, not Q (no longer conflated with capacity).
9. Capacity respects `q_i` (previously fixed at 1).
10. DRL removed — Pipeline B's faked `ga_time × 0.05` line is retracted; MILP vs GA vs ALNS is the comparison.
11. Pareto frontier produced via dominance-filtered ε-sweep.
12. Behavioural Monte Carlo is a post-hoc evaluator (Beta sensitivity sweep), not part of the cost function.

### Work breakdown

1. Replace §4 (Formulation) wholesale.  Keep the set-and-sets layout; rewrite equations from `docs/FORMULATION.md §§2–8`.
2. Add §4.5 "Data calibration" summarising BASM + BPR + darkstoremap.in provenance with a one-paragraph explanation of why each source.
3. Rewrite §5 (Solution Approaches): MILP exact (PWL outer-approx), GA (giant-tour + Bellman-Ford split, OX1, swap/insert/reverse, SA not used), ALNS (Ropke-Pisinger with 4 destroy / 3 repair ops).  Drop DRL.
4. §6 Experiments: 5 (or 6) cities × N ∈ {20, 50, 100, 200} × 10 seeds × ε-grid.  Statistical tests: Wilcoxon signed-rank + Bonferroni.
5. §7 Results: swap every figure reference to the Task #14 outputs.
6. §8 Conclusion: discuss the trade-off observed; limitations (compliance MC priors, calibration uncertainty).
7. Abstract + introduction: prune DRL claim, acknowledge SUMO replacement (crash-survival MC).

### Acceptance criteria

- [ ] `SafetyAwareRoutingProposal.tex` compiles on the CI LaTeX container.
- [ ] Every equation in the paper matches a named equation in `docs/FORMULATION.md`.
- [ ] Every figure reference (`\ref{fig:…}`) points to an artefact produced by `scripts/make_figures.py`.
- [ ] No claim that isn't measured appears in the results section (grep for "DRL", "SUMO", "O(1)" — all must be absent or explicitly scoped as future work).

### Estimated effort: **3 days after #14 closes**

---

## Task #19 — Legacy parity harness  *(optional, low-priority)*

### Purpose

Cross-check: given the same (seed, N), does the new pipeline reproduce the old `07_solver_ga.py`'s score within tolerance?  This is diagnostics only — the formulation has changed, so exact parity is NOT expected.  The value is *regression-tracking* while we migrate.

### Work breakdown

1. Write `tests/test_parity.py`:
   - Load `data/05_solution.json` (old pipeline output, score 394.28).
   - Build a matching `Instance` using the legacy 30-customer instance JSON.
   - Run the new GA with a calibrated risk config.
   - Report both scores side-by-side.
   - No hard assertion — just emit a diff with a 50% tolerance warning for large deltas.
2. A short note in `docs/LEGACY_PARITY.md` explaining *why* the numbers will differ (objective function changed, risk semantics changed).

### Acceptance criteria

- [ ] Test runs green (no hard assertion, just a printable diff).
- [ ] `docs/LEGACY_PARITY.md` documents the expected divergence.

### Estimated effort: **½ day, any time**

---

## Suggested execution order and calendar

Assuming one engineer working sequentially:

| Day | Task | Output |
|---|---|---|
| 1–3 | #16 — calibration source research & doc | `docs/BASM_CALIBRATION.md`, populated `conf/risk/basm_v1.yaml` |
| 3 | #18 — your decision; config/doc update | final city list locked in |
| 4 | #17 — Blinkit cross-val | `data/validation/darkstore_crossval.json` |
| 5–6 | #9 — BASM implementation + cross-val test | `attach_risk` landed, end-to-end real run works |
| 7 | full 5-city experiment grid run (1 seed first) | `data/results/*.json` |
| 8–11 | #14 — paper figures | 8 figure families + per-figure CSVs |
| 12–14 | #15 — LaTeX rewrite | updated `SafetyAwareRoutingProposal.tex` |
| 15 | full 10-seed grid run on GPU cluster | `data/results/` populated |
| 16 | regenerate figures from the full grid, final paper check | submission-ready |

**Total: ~16 working days from calibration decision to submission-ready.**  #19 (parity) can slot in any time.

---

## What I need from you to unblock work

1. **Option pick for #18** (Hyderabad: drop / Zepto-sub / refresh / Chennai-sub).
2. **Sign-off on the #16 hybrid approach** (MoRTH + Mohan + OSM proxy) or an alternative.
3. **Any Indian-urban-safety references** you already have in hand (saves search time).
4. **GPU cluster access details** (for the full 10-seed grid) — when it's convenient.
5. **Permission to commit the work so far** on `refactor/modularize` (the branch has ~4.6k LOC uncommitted).

Once (1)–(3) are answered I can fully self-drive through #9 → #14 → #15 without further blocks.
