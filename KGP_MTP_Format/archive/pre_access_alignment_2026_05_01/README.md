# Archived MTP draft — pre-ACCESS-alignment

This folder is a snapshot of `KGP_MTP_Format/Chapters/`, `FrontMatter/abstract.tex`,
and `references.bib` taken on 2026-05-01 *before* the rewrite that aligned the
thesis to the ACCESS manuscript (`ACCESS_latex_template_20240429/access.tex`)
and the actual codebase under `src/savrptw/`.

**Why archived (and not used):**
- Per-route residential cap was `H̄_k = 8`; the codebase + ACCESS use `100`.
- ε-grid `H̄ ∈ {0,4,8,16,32,∞}` and `R̄ ∈ {0.05,…,1.6,∞}` did not match the
  experiment YAML.
- Literature-review chapter listed PRISMA-style "450 records → 25 included"
  numbers that were not actually executed.
- Cited keys for DRL baselines (`vinyals2015`, `kool2019`, `mnih2015`,
  `williams1992`, `nagata2010`) and iRAD/iRAP/Fairwork (`irad2021`,
  `irap2020`, `fairwork2024`) which the rewritten scope no longer claims.
- Missing the explicit `π_{ij}` formula, the multi-depot coupling equations,
  pseudocode for the heuristics, and the PWL-breakpoint justification that
  the ACCESS self-review flagged as required.

The live thesis content is the rewritten `KGP_MTP_Format/Chapters/*.tex`
and `FrontMatter/abstract.tex` produced on 2026-05-01.
