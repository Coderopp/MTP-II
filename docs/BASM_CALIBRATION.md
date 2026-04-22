# BASM Calibration

## Scope

This note documents the reproducible calibration used by `conf/risk/basm_v1.yaml` and implemented in `src/savrptw/risk/basm.py`. The goal is not to claim access to protected iRAD microdata. Instead, the model converts publicly documented aggregate road-safety rates into a transparent edge-level prior and then redistributes that prior within each OSM class using graph-local proxy structure.

The calibrated source string is:

`morth_mohan_osm_proxy_v1`

The retained primary cities are:

- Bengaluru
- Delhi
- Gurugram
- Mumbai
- Pune

Hyderabad is explicitly excluded from the primary single-brand experiment set because the committed Blinkit snapshot dated 2026-04-09 has zero in-bounding-box placemarks there.

## Sources

### S1. MoRTH 2022 annual report

Source PDF:

- Ministry of Road Transport and Highways. *Road Accidents in India 2022*. https://www.morth.gov.in/sites/default/files/RA_2022_30_Oct.pdf

Tables used:

- Page 8: national counts of fatal accidents and grievous-injury accidents in 2022.
- Page 25, Table 2.1: fatalities by road category in 2022 and the corresponding road-length shares discussed in Section 2.
- Page 89, Table 6.5: top-10 million-plus-city fatalities in 2022. This is used only as a city-anchor note for Delhi and Bengaluru, not to derive the class constants in `basm_v1.yaml`.

Numbers extracted from the report:

| Quantity | Value | MoRTH location |
|---|---:|---|
| Fatal accidents, India 2022 | 155,781 | p. 8 |
| Grievous-injury accidents, India 2022 | 143,374 | p. 8 |
| National Highway fatalities | 61,038 | p. 25, Table 2.1 |
| State Highway fatalities | 41,012 | p. 25, Table 2.1 |
| Other Roads fatalities | 66,441 | p. 25, Table 2.1 |
| National Highway road length share | 1.32 lakh km | p. 25, Section 2.1 |
| State Highway road length share | 1.80 lakh km | p. 25, Section 2.1 |
| Other Roads length share | 60.59 lakh km | p. 25, Section 2.1 |

### S2. Mohan, Bangdiwala, and Villaveces (2017)

Source:

- Mohan, D., Bangdiwala, S. I., and Villaveces, A. (2017). *Urban street structure and traffic safety*. *Journal of Safety Research*, 62, 63-71. DOI: https://doi.org/10.1016/j.jsr.2017.06.003

Supporting public abstracts:

- PubMed: https://pubmed.ncbi.nlm.nih.gov/28882278/
- ScienceDirect article page: https://www.sciencedirect.com/science/article/pii/S0022437516305357

Directional findings used:

- Higher junction density is associated with lower fatality rates.
- A greater share of highways and main arterial roads is associated with higher fatality rates than non-arterial roads.

This paper does not provide an India-specific numeric multiplier table by OSM highway tag. Therefore, Mohan (2017) is used as a directional constraint on the ordering of multipliers, not as a direct source of raw coefficients.

### S3. Existing congestion exposure table

Source config:

- `conf/congestion/bpr.yaml`

This file already encodes the project’s class-level capacity, lane, and V/C assumptions used for BPR travel-time inflation. BASM reuses those values only to convert annual crash totals per kilometre into per-traversal priors. That keeps risk and travel-time exposure models internally consistent.

## Derivation

### Step 1. Convert fatal-only category totals to fatal-plus-grievous totals

The national severity uplift factor is:

`severity_uplift = (fatal + grievous) / fatal = (155,781 + 143,374) / 155,781 = 1.92037`

Applying that factor to the MoRTH road-category fatalities on page 25 gives fatal-plus-grievous events per road category:

| Road category | Fatalities | Fatal + grievous proxy |
|---|---:|---:|
| National Highways | 61,038 | 117,211.88 |
| State Highways | 41,012 | 78,757.56 |
| Other Roads | 66,441 | 127,593.30 |

### Step 2. Convert category totals into annual events per kilometre

Using the road lengths from page 25:

| Road category | Length (km) | Fatal + grievous per km-year |
|---|---:|---:|
| National Highways | 132,000 | 0.8880 |
| State Highways | 180,000 | 0.4375 |
| Other Roads | 6,059,000 | 0.0211 |

### Step 3. Convert annual per-km rates into per-traversal priors

For each OSM class, BASM uses the existing BPR exposure assumptions:

`annual_edge_traversals_reference = capacity_per_lane × default_lanes × vc_class_peak × (6 × 1.0 + 18 × 0.35) × 365`

The corresponding values are copied into `conf/risk/basm_v1.yaml` under `cross_validation.annual_edge_traversals_reference`.

The base per-traversal, per-km rates are:

`lambda_class = (fatal+grievous per km-year for mapped category) / annual_edge_traversals_reference`

Mapping from MoRTH road category to OSM class:

- National Highways: `motorway`, `trunk`, `primary`
- State Highways: `secondary`
- Other Roads: `tertiary`, `residential`, `living_street`, `service`, `unclassified`

This yields the `lambda_class` values in `basm_v1.yaml`:

| OSM class | `lambda_class` |
|---|---:|
| motorway | 3.88e-08 |
| trunk | 6.13e-08 |
| primary | 6.11e-08 |
| secondary | 8.65e-08 |
| tertiary | 6.53e-09 |
| residential | 2.61e-08 |
| living_street | 1.04e-07 |
| unclassified | 9.41e-09 |
| service | 5.87e-08 |

### Step 4. Build bounded class multipliers with MoRTH intensity and Mohan ordering

From page 25, compute fatality intensity by road category as:

`intensity = fatality_share / road_length_share`

Using the 2022 shares:

- National Highways: `36.23 / 2.10 = 17.252`
- State Highways: `24.34 / 2.80 = 8.693`
- Other Roads: `39.43 / 95.00 = 0.415`

Let `g` be the geometric mean of these three intensities:

`g = (17.252 × 8.693 × 0.415)^(1/3) = 3.963`

To keep class multipliers bounded while preserving the arterial > collector > local ordering supported by Mohan (2017), BASM applies the transform:

`severity_multiplier(category) = sqrt(intensity / g)`

This gives:

| Category tier | Multiplier |
|---|---:|
| Arterial tier (NH-like) | 2.086 |
| Collector tier (SH-like) | 1.481 |
| Local tier (Other-roads-like) | 0.324 |

Mapped to OSM classes:

- `motorway`, `trunk`, `primary` → `2.086`
- `secondary`, `tertiary` → `1.481`
- `residential`, `living_street`, `service`, `unclassified` → `0.324`

The transform is explicit, reproducible, and audit-friendly: the rank comes from Mohan (2017), while the magnitude comes from MoRTH’s road-category fatality intensity.

### Step 5. OSM proxy redistribution

Within each class, BASM multiplies the class prior by a bounded local proxy:

`proxy_edge ∈ [0.5, 2.0]`

The proxy is computed from:

- approximate edge betweenness on the city’s time-weighted simple graph,
- local traffic-signal density near the edge endpoints,
- local crossing density near the edge endpoints.

This does not alter the source provenance of `lambda_class` or `severity_multiplier`; it only redistributes risk locally inside the class using graph structure already present in the OSM network.

## Final BASM equation

For each edge `(i, j)`:

`r_ij = clip(lambda_class(hw_ij) × len_km × severity_multiplier(hw_ij) × proxy_edge(i,j), 0, 0.99)`

where `hw_ij` is the canonicalized OSM highway class and `len_km = length_m / 1000`.

## Notes on city-level validation

The code exposes `expected_annual_events()` and `relative_error()` so a city-specific validation can be written as:

`relative_error(expected_annual_events(G), target_city_total) <= 0.25`

For Delhi and Bengaluru, the report’s page-89 table provides direct fatality anchors:

- Delhi 2022 fatalities: 1,461
- Bengaluru 2022 fatalities: 772

The current repository does not yet contain direct 2022 MoRTH fatality anchors for Mumbai, Pune, and Gurugram in the same table set, so the class constants are documented and implemented now, while full five-city target validation remains a separate data-assembly step.
