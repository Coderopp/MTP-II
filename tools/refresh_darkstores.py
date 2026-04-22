"""Refresh the committed Blinkit KML snapshot.

Paper results are tied to `data/raw/darkstoremap_in_2026-04-09.kml`.  Running
this script downloads a newer snapshot to a dated filename; it will NOT
overwrite the committed artefact.  Extending the paper to a fresher dataset
requires updating `conf/experiment/default.yaml` and re-running the pipeline.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://darkstoremap.in/dark_store.kml"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    today = dt.date.today().isoformat()
    out = OUT_DIR / f"darkstoremap_in_{today}.kml"
    if out.exists():
        print(f"[refresh] {out} already exists — leaving in place", file=sys.stderr)
        return 0
    print(f"[refresh] downloading {URL} → {out}")
    with urllib.request.urlopen(URL) as resp:  # noqa: S310
        body = resp.read()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    print(f"[refresh] saved {len(body):,} bytes — sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
