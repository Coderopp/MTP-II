"""Cross-validate the committed Blinkit KML snapshot against OpenStreetMap.

Methodology
-----------
1. Sample N stores uniformly at random from the committed KML snapshot
   (per-city or pooled).
2. For each sample, query Overpass for OSM nodes/ways with `brand=Blinkit`
   or `name~"Blinkit"` within `--radius` metres of the KML coordinate.
3. A sample is considered "matched" if at least one OSM hit is found within
   the radius.
4. Report per-city and pooled match rates with Wilson 95% CI.

Caveats
-------
OSM coverage of dark stores is incomplete and inconsistent. This is a
recall lower bound, not a coordinate-quality measurement. A low match
rate is more evidence about OSM coverage than about KML accuracy. Treat
the output as a sanity check, not as ground truth.

Usage
-----
    PYTHONPATH=src python3 tools/validate_blinkit_snapshot.py \\
        --kml data/raw/darkstoremap_in_2026-04-09.kml \\
        --sample-per-city 10 --radius 250 \\
        --out data/validation/blinkit_validation.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from hydra import compose, initialize_config_dir

from savrptw.data.darkstores import filter_by_bbox, parse_blinkit_placemarks


OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "savrptw-blinkit-validator/0.1 (research; contact: refactor/modularize)"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def overpass_query(lat: float, lon: float, radius_m: int) -> str:
    return f"""[out:json][timeout:25];
(
  node(around:{radius_m},{lat},{lon})["brand"="Blinkit"];
  node(around:{radius_m},{lat},{lon})["name"~"Blinkit",i];
  way(around:{radius_m},{lat},{lon})["brand"="Blinkit"];
  way(around:{radius_m},{lat},{lon})["name"~"Blinkit",i];
);
out tags center;"""


def overpass_hits(lat: float, lon: float, radius_m: int, timeout: int = 30) -> int:
    body = f"data={quote(overpass_query(lat, lon, radius_m))}"
    req = Request(
        OVERPASS,
        data=body.encode(),
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"  WARN: overpass error at ({lat:.4f}, {lon:.4f}): {e}", file=sys.stderr)
        return -1
    return len(payload.get("elements", []))


def sample_stores(kml: Path, conf_dir: Path, n_per_city: int, seed: int) -> dict[str, list]:
    rng = random.Random(seed)
    all_recs = parse_blinkit_placemarks(kml)
    out: dict[str, list] = {}
    with initialize_config_dir(config_dir=str(conf_dir.resolve()), version_base="1.3"):
        cfg = compose(config_name="config")
        cities = list(cfg.experiment.cities)
        for city in cities:
            ccfg = compose(config_name="config", overrides=[f"city={city}"])
            allow = set(ccfg.city.get("kml_folders") or [])
            pool = all_recs if not allow else [r for r in all_recs if r.folder in allow]
            pool = filter_by_bbox(pool, ccfg.city.bbox)
            if len(pool) <= n_per_city:
                out[city] = list(pool)
            else:
                out[city] = rng.sample(list(pool), n_per_city)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kml", default="data/raw/darkstoremap_in_2026-04-09.kml")
    ap.add_argument("--conf", default="conf")
    ap.add_argument("--sample-per-city", type=int, default=10)
    ap.add_argument("--radius", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--throttle", type=float, default=2.0,
                    help="seconds between Overpass requests")
    ap.add_argument("--out", default="data/validation/blinkit_validation.json")
    args = ap.parse_args()

    samples = sample_stores(Path(args.kml), Path(args.conf), args.sample_per_city, args.seed)

    per_city = {}
    all_results = []
    for city, recs in samples.items():
        hits = []
        for r in recs:
            n_hits = overpass_hits(r.lat, r.lon, args.radius)
            hits.append({
                "store_name": r.store_name,
                "lat": r.lat,
                "lon": r.lon,
                "osm_hits": n_hits,
                "matched": (n_hits is not None and n_hits > 0),
            })
            all_results.append({"city": city, **hits[-1]})
            time.sleep(args.throttle)
        ok = sum(1 for h in hits if h["matched"])
        n = len([h for h in hits if h["osm_hits"] >= 0])  # exclude errors
        p, lo, hi = wilson_ci(ok, n)
        per_city[city] = {
            "n_sampled": len(hits),
            "n_responded": n,
            "n_matched": ok,
            "rate": p,
            "ci95": [lo, hi],
        }
        print(f"  {city:11s} {ok}/{n} matched (rate={p:.2f}, 95% CI [{lo:.2f}, {hi:.2f}])")

    pooled_ok = sum(1 for r in all_results if r["matched"])
    pooled_n = sum(1 for r in all_results if r["osm_hits"] >= 0)
    p, lo, hi = wilson_ci(pooled_ok, pooled_n)

    out = {
        "snapshot": str(args.kml),
        "radius_m": args.radius,
        "seed": args.seed,
        "per_city": per_city,
        "pooled": {
            "n_sampled": len(all_results),
            "n_responded": pooled_n,
            "n_matched": pooled_ok,
            "rate": p,
            "ci95": [lo, hi],
        },
        "details": all_results,
        "caveats": [
            "OSM coverage of dark stores is incomplete and inconsistent.",
            "A low rate is evidence about OSM coverage, not KML accuracy.",
            "Overpass-API errors (-1) are excluded from the denominator.",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nPooled: {pooled_ok}/{pooled_n} matched "
          f"(rate={p:.2f}, 95% CI [{lo:.2f}, {hi:.2f}]) -> {args.out}")


if __name__ == "__main__":
    main()
