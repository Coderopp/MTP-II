# Legacy Port Notes

Source files:
- `data/04_vrptw_instance.json`
- `data/05_solution.json`
- `data/03_graph_final.graphml`

## Preserved
- 3 depots (Blinkit brand assumed)
- 30 customers with (e_i, demand, home_depot)
- 17 routes with their node sequences

## Reconstructed (best-effort)
- `eta_i = e_i + 10` (10-min q-commerce promise; legacy stored only `e_i`/`l_i`)
- `service_time = 2.0` minutes (BASM default; legacy did not record this)
- `arrivals = [0.0, ...]` placeholder; legacy did not record per-node arrival times

## Dropped
- Synthetic crash rates (legacy used iRAD-Bengaluru placeholder; not portable to BASM v1)
- Lambda objective weights `[0.4, 0.4, 0.2]` (modular schema uses ε-constraint, not weighted sum)

## Provenance flag
`Instance.city == 'legacy_bengaluru'` — this is a marker, not a real city slug.
