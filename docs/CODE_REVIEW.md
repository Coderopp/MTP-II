# PR Code Review: Legacy Solver Branches vs. Modular Package

## Overview

Two PR branches (`fix-formulation-11540870885496433647` and `claude/vrptw-optimization-e68yt`) were authored before the SA-VRPTW refactor (commit eb7fe9a) and commit raw solver files as monolithic scripts. The modular package on `main` already encapsulates MILP, GA, ALNS solvers; instance generation; and validation within typed dataclass contracts (`Instance`, `Solution`) and pluggable configurations (Hydra). Both PRs use dict-based instance structures, weighted-sum multi-objective (λ-weights), and lack the ε-constraint multi-objective reformulation. **Verdict: Both are superseded.**

---

## Per-File Verdict Table

| Filename | Branch | LOC | Verdict | Reason |
|----------|--------|-----|---------|--------|
| `fix_milp.py` | `fix-formulation` | 208 | SUPERSEDED | PuLP MILP solver; soft-TW via linear lateness. Modular `src/savrptw/solvers/milp.py` (483 LOC) uses PWL outer-approximation for convex exp(β·τ)−1, matching FORMULATION.md §8. Dict-based instance vs. typed. |
| `fix_meta.py` | `fix-formulation` | 153 | SUPERSEDED | GA/ALNS framework; 2-opt + relocate operators. Modular `ga.py` (208 LOC) + `alns.py` (559 LOC) offer richer local-search (Shaw removal, regret insertion). Uses typed `Instance`/`Solution` and adaptive σ-weighting. |
| `fix_drl.py` | `fix-formulation` | 252 | REJECT | DRL baseline; user explicitly dropped DRL per requirements. Do not port. |
| `fix_runexp.py` | `fix-formulation` | 73 | SUPERSEDED | Experimental harness with weighted-sum λ-sweep. Modular `src/savrptw/runner.py` (112 LOC) uses ε-constraint multi-objective. Different design. |
| `fix_instgen.py` | `fix-formulation` | 242 | SUPERSEDED | OSM pipeline + instance generation. Modular `src/savrptw/instance/generator.py` (192 LOC) + graph/risk/congestion modules. Handles calibrated BASM risk; fix version uses synthetic uncalibrated risk. |
| `fix_validate.py` | `fix-formulation` | 234 | SUPERSEDED | Instance validation checker. Modular `src/savrptw/eval/feasibility.py` (324 LOC) validates against typed `Instance`/`Solution`. Same spirit, typed dataclasses. |
| `vrptw_milp.py` | `claude/vrptw-optimization` | 298 | SUPERSEDED | PuLP solver with lambda-weighted objective. Modular version is stricter (PWL vs. linear lateness), uses ε-constraints. Routes extracted identically. |
| `vrptw_meta.py` | `claude/vrptw-optimization` | 426 | SUPERSEDED | GA + ALNS with mutation & repair. Modular versions are more complete (Shaw removal, regret-2/3, sigma weighting). Weighted-sum λ inconsistent with ε-constraint design. |
| `vrptw_drl.py` | `claude/vrptw-optimization` | 266 | REJECT | REINFORCE Pointer network; DRL dropped. Do not port. |
| `vrptw_runexp.py` | `claude/vrptw-optimization` | 256 | SUPERSEDED | Benchmark harness with weighted-sum multi-objective. Modular runner.py uses ε-constraint approach. Different strategy. |
| `vrptw_instgen.py` | `claude/vrptw-optimization` | 230 | SUPERSEDED | Instance generator without calibrated risk (uncalibrated synthetic). Modular generator enforces BASM calibration. |

---

## Cherry-Pick Candidates

**None.** Both PRs use structural decisions (weighted-sum λ, dict-based instances, uncalibrated/synthetic risk) that contradict the modular package's design. The modular solvers are feature-complete (PWL for STW, Shaw/regret operators, sigma adaptation) and supersede the legacy implementations.

### Rejected DRL Ports

Both branches include DRL solvers (fix_drl.py, vrptw_drl.py). Per requirements, DRL has been **explicitly dropped** — do not port either version. The environment masking and training loops are sound but unnecessary for the current scope.

### Rejected __pycache__ Files

The `claude/vrptw-optimization` branch includes `__pycache__` directories (Python bytecode). These are build artifacts and should never be committed. Flag as REJECT if present in any PR.

---

## Comments to Post on the PRs

### For `fix-formulation-11540870885496433647`

**Comment 1 (Formulation):**
> The MILP solver uses a linear lateness penalty (`late_jk >= tau_jk - l_j`), while the current package implements the convex STW penalty (exp(β·τ)−1) via PWL outer-approximation per FORMULATION.md §8. The difference impacts optimality and F₁ reporting. The modular solver is the canonical one; this PR's approach is subsumed.

**Comment 2 (Instance Design):**
> This PR uses dict-based instances with hand-rolled λ-weighting. The modular package (commit eb7fe9a) uses typed `Instance`/`Solution` dataclasses and ε-constraint multi-objective reformulation (FORMULATION.md §6). The two designs are incompatible. All features are now in the modular package.

**Comment 3 (DRL):**
> The DRL baseline in fix_drl.py will not be ported — per user guidance, DRL was dropped from the scope. The MILP and metaheuristic solvers are the official baselines.

---

### For `claude/vrptw-optimization-e68yt`

**Comment 1 (Multi-Objective Design):**
> This branch uses weighted-sum multi-objective (λ = [0.4, 0.4, 0.2]), while the canonical formulation (commit eb7fe9a) switched to ε-constraint multi-objective with dynamic budgets (R_bar, H_bar) per FORMULATION.md §6. The weighted-sum approach is not compatible with the current package design. The modular solvers already implement the correct objective.

**Comment 2 (Instance Format):**
> The instance generator and solvers in this branch expect dict-based instances with λ fields. The modular package uses `Instance`/`Solution` dataclasses (savrptw/types.py) and Hydra configs. Integration would require a full rewrite of the data contract.

**Comment 3 (Local Search Operators):**
> The GA/ALNS solvers here use 2-opt and simple relocate. The modular ALNS (alns.py, 559 LOC) includes Shaw removal, regret-2 and regret-3 insertion, and adaptive σ-weighting (Ropke & Pisinger, 2006). All improvements are already incorporated.

**Comment 4 (DRL & __pycache__):**
> DRL will not be ported per user guidance. Also note: the branch includes __pycache__ directories (Python bytecode) which should never be committed; use .gitignore to exclude them.

---

## Summary of Findings

- **Weighted-sum λ vs. ε-constraint:** All PR code uses λ-weighting; modular package uses dynamic ε-constraints. Incompatible designs.
- **Dict vs. dataclass instances:** PRs use raw dicts; modular uses typed `Instance`/`Solution`.
- **Risk model:** PRs use synthetic uncalibrated risk; modular enforces BASM v1 calibration.
- **Soft TW:** Both use linear lateness or simple penalties; modular uses PWL exp(β·τ)−1 per FORMULATION.md §8.
- **Local search:** PRs have basic 2-opt/relocate; modular has Shaw removal, regret insertion, sigma weighting.
- **DRL:** Both include it; user explicitly dropped DRL from scope.
- **No porting recommended.** The modular package is more complete, type-safe, and follows the canonical formulation.

