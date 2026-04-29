# Self-Review — IEEE Access Manuscript (pre-results pass)

**Paper:** `ACCESS_latex_template_20240429/access.pdf` — 7 pages, 17 cited
references, 3 tables, 1 algorithm float, 2 figure placeholders.
**Reviewed at:** commit on `main` after BibTeX externalization +
ALNS pseudocode addition.
**Scope:** structural / formulation / calibration / writing review,
explicitly skipping the known empirical-numbers gap (\TBD markers).

## Review Summary

| Dimension          | R1 (harsh-fair) | R2 (harsh-critical) | R3 (open-minded) | Avg |
|--------------------|----------------:|--------------------:|-----------------:|----:|
| Formulation        | 7 | 6 | 8 | 7 |
| Calibration story  | 8 | 7 | 9 | 8 |
| Methods clarity    | 6 | 5 | 7 | 6 |
| Writing quality    | 6 | 6 | 7 | 6 |
| Structural         | 7 | 6 | 8 | 7 |
| **Overall**        | **6.5** | **5.5** | **7.5** | **6.5** |

Decision (assuming TBD numbers land cleanly): **Accept with minor revisions.**
Decision today (pre-results): **Major revisions required** — but driven
by the known gap, not by formulation issues.

## Consensus strengths

1. **Formulation is internally consistent and complete.** Equations
   (4)–(15) form a clean MILP-ready statement. The ε-constraint move
   (5)–(7) is correctly orthogonal to the F₁ objective.
2. **Calibration honesty is unusual and refreshing.** The MoRTH 2022
   + Mohan 2017 + OSM-proxy chain is reproducible, every constant in
   `basm_v1.yaml` traces to a public source, and the iRAD-microdata
   gap is openly acknowledged.
3. **Replication kit is high-quality.** Hydra configs, MIT licence,
   `scripts/run_experiment.py` and `scripts/run_grid.py` invocations
   are quoted verbatim. This is rare for the venue.
4. **Algorithm 1 (ALNS) gives reviewers a concrete object** to verify
   against the implementation in `src/savrptw/solvers/alns.py`.

## Consensus weaknesses (fix before the iDRAC numbers land)

### W1. §III.F — multi-depot coupling equations elided
Current text: *"omitted for brevity; full equations in the open-source
repository."* For a methods-heavy paper this is the wrong choice — at
minimum state the three coupling constraints (rider leaves home depot
≤ once, returns iff left, cannot touch other depots) as displayed
equations. Reviewers should not have to leave the PDF to verify the
formulation is sound.

### W2. Super-graph design imposes an unstated risk-optimization limit
The super-graph fixes the underlying street-graph path between every
customer/depot pair to the shortest-time path at instance-generation
time (§III.A, eqs. 1–3). This means R_{ij} and H_{ij} are computed on
that fixed path. **Therefore the optimisation is over customer
*orderings*, not over which physical streets to drive on.** This is a
non-trivial modelling restriction that should be flagged explicitly in
§III.A, not just buried in the discussion. A risk-aware planner
might prefer a slightly slower path through safer streets — the
current formulation cannot select that. Acknowledge it; defer to
future work if you want to lift it.

### W3. BASM proxy weights not stated in the paper
§IV.A says π_{ij} ∈ [0.5, 2.0] is *"a bounded local proxy combining
approximate edge betweenness, traffic-signal density, and crossing
density"* — but the actual weights (`w_betweenness=0.40`,
`w_signal_density=0.30`, `w_crossing_density=0.30`) live only in
`conf/risk/basm_v1.yaml`. Add a one-sentence equation:

> π_{ij} = clip(1 + 0.40·(2β_{ij} − 1) + 0.30·s_{ij} + 0.30·c_{ij}, 0.5, 2.0)

with β, s, c the normalised betweenness/signal/crossing densities.
Otherwise reviewers cannot replicate the proxy from the paper alone.

### W4. Cross-validation rule mentioned in FORMULATION but absent from paper
The internal note specifies that the instantiated r_{ij} field must
match the chosen public city aggregate within ±25%. This is exactly
the kind of falsifiable claim reviewers like — promote it from
internal note to a paragraph in §IV.A.

### W5. GA has prose but no pseudocode (symmetry with ALNS)
ALNS gets Algorithm 1; the GA gets a paragraph. Add Algorithm 2 (GA
with giant-tour + Bellman–Ford split) for symmetry. Without it the
methods section feels lopsided.

### W6. Related Work is a list of stubs, not a synthesis
The four subsections (VRPTW with soft TW, safety-aware VR,
quick-commerce, calibration) are 2–3 sentences each. For an IEEE
Access submission expand each to a short paragraph with at least one
contrast sentence ("X did Y; we differ in Z"). Currently the section
reads like an outline.

### W7. No notation table
With 9 distinct symbols in the formulation (q, e, ETA, s, T, R, H,
H̄, β_{stw}, …) a single table summarising symbols, units, and
sources would substantially help reviewers. Slot it after §III.B.

### W8. PWL breakpoint choice unjustified
§III.G states breakpoints {0, 2, 5, 10, 15, 20, 30} for the
exp(βτ)−1 outer envelope but does not say why these. One sentence
("breakpoints chosen by curvature-density heuristic to keep relative
approximation error below 5% over τ ∈ [0, 30]") would close this gap.

## Lower-priority suggestions

- **S1.** §VI.A: replace `[TBD: remote iDRAC compute node specification]`
  with a short table once the run completes.
- **S2.** §I (introduction): the 6-item contributions list runs ~14
  lines. Trim to 4 items (merge 4+5, drop 6 or absorb into 1).
- **S3.** §II.D claims that Sinchaisri et al. 2023 documents
  "operational intensity of q-commerce dispatch in the United
  States." Verify the citation actually says that (the entry is
  marked "to be published"); if it shifts, soften the claim.
- **S4.** Add a `\noindent` after the algorithm float so the next
  paragraph aligns visually.
- **S5.** Two figure placeholders currently render as text-in-frame.
  Once `figures/` populates from the iDRAC run, replace with
  `\includegraphics[width=\columnwidth]{<path>}`.

## Pre-results checklist (what to fix now, before pushing the iDRAC button)

- [ ] W1: spell out multi-depot coupling equations.
- [ ] W2: add one paragraph in §III.A noting the super-graph
      shortest-time-path fixing.
- [ ] W3: add the explicit π_{ij} formula with weights.
- [ ] W4: add the ±25% cross-validation paragraph in §IV.A.
- [ ] W5: add Algorithm 2 (GA + Bellman–Ford split).
- [ ] W6: expand Related Work subsections to short paragraphs.
- [ ] W7: insert a notation table in §III.
- [ ] W8: justify the PWL breakpoint choice in one sentence.

## Per-reviewer notes

**R1 (harsh-fair):** "I'd accept this with minor revisions if the
empirical study lands. The formulation is sound; the calibration
chain is the kind of careful work I want to see more of. But the
multi-depot equations being omitted is unprofessional and the GA
asymmetry with ALNS is jarring."

**R2 (harsh-critical):** "The super-graph design quietly limits what
'safety-aware routing' means in this paper — it's safety-aware
*ordering*, not safety-aware *path selection*. That's defensible but
must be acknowledged in the formulation, not in the discussion. Also,
the proxy weights for the BASM are not in the paper. Without them I
cannot evaluate whether the calibration is principled or
hand-tuned."

**R3 (open-minded):** "Genuinely interesting problem framing.
ε-constraints with a separate residential-edge budget is the kind of
operator-facing dial that gets adopted in practice. The calibration
honesty is rare and worth highlighting in the abstract. Don't
under-sell it."
