# Dark-Store Snapshot Validation

## Purpose

The Blinkit dark-store coordinates used in this project come from a single
public source: `https://darkstoremap.in/dark_store.kml`. The committed
snapshot is `data/raw/darkstoremap_in_2026-04-09.kml`. This note records
the cross-validation we attempted against OpenStreetMap (OSM) and the
conclusion we reached.

## Method

`tools/validate_blinkit_snapshot.py` samples N stores per primary city
from the KML, queries the Overpass API for OSM features tagged
`brand=Blinkit` or `name~"Blinkit"` within a configurable radius, and
reports per-city and pooled match rates with Wilson 95% CI.

## Results

### Run A — 2026-04-25, radius = 250 m, sample = 3/city (15 total)

| City      | Matched | Rate | 95% CI |
|-----------|--------:|-----:|--------|
| Bengaluru | 0/3 | 0.00 | [0.00, 0.56] |
| Delhi     | 0/3 | 0.00 | [0.00, 0.56] |
| Gurugram  | 0/3 | 0.00 | [0.00, 0.56] |
| Mumbai    | 0/3 | 0.00 | [0.00, 0.56] |
| Pune      | 0/3 | 0.00 | [0.00, 0.56] |
| **Pooled** | **0/15** | **0.00** | **[0.00, 0.20]** |

### Run B — 2026-04-25, radius = 1000 m, sample = 5/city (24 responded)

| City      | Matched | Rate | 95% CI |
|-----------|--------:|-----:|--------|
| Bengaluru | 0/4 | 0.00 | [0.00, 0.49] |
| Delhi     | 1/5 | 0.20 | [0.04, 0.62] |
| Gurugram  | 1/5 | 0.20 | [0.04, 0.62] |
| Mumbai    | 0/5 | 0.00 | [0.00, 0.43] |
| Pune      | 0/5 | 0.00 | [0.00, 0.43] |
| **Pooled** | **2/24** | **0.08** | **[0.02, 0.26]** |

Run B raw results in `data/validation/blinkit_validation_wide.json`.

## Conclusion

OSM coverage of Indian dark stores is essentially zero. Across two
sampling rounds and two radii, the pooled 95% CI upper bound on the
match rate is 0.26. This is **not** evidence about KML coordinate
quality; it is evidence that OSM is not a viable ground-truth source
for Blinkit dark-store coordinates in India in 2026.

We retain the validator script for reproducibility and as a regression
tripwire (if OSM coverage ever closes, the same command will detect it).
For the paper we report:

- Source: `darkstoremap.in` snapshot 2026-04-09, 461 Blinkit placemarks.
- Per-city in-bbox counts: Bengaluru 113, Delhi 111, Gurugram 38, Mumbai 96, Pune 39.
- OSM cross-validation deferred — OSM brand tagging coverage too sparse
  (≤ 26% upper bound at 1 km radius).

## Future work

- A 50–100 sample manual check on Google Maps would lift this to a real
  accuracy figure. We did not attempt this — out of scope for the
  modular refactor.
- If a richer reproducible source becomes available (e.g. RBI store
  registry, GST registration data), wire it through
  `tools/validate_blinkit_snapshot.py` by adding a new query backend.
