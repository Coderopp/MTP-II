"""Port legacy `data/04_vrptw_instance.json` and `data/05_solution.json`
into the modular `savrptw.types.Instance` / `Solution` schema.

The legacy instance was generated with synthetic risk and bbox-only depots,
so the ported objects are *archaeological*: they let the modular code
ingest, validate, and re-evaluate the legacy results, but they should not
be confused with a real-pipeline instance.

Usage:

    python tools/port_legacy_instance.py \\
        --instance data/04_vrptw_instance.json \\
        --solution data/05_solution.json \\
        --graph    data/03_graph_final.graphml \\
        --out      data/legacy_ported/

The script writes:

    data/legacy_ported/instance.pkl   — pickled savrptw.types.Instance
    data/legacy_ported/solution.json  — savrptw.types.Solution as JSON
    data/legacy_ported/PORT_NOTES.md  — what was preserved / what was dropped
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx

from savrptw.types import Customer, Depot, Instance, Route, Solution


def _node_id_to_int(raw) -> int:
    """Legacy stored OSM ids as strings; the modular schema uses ints."""
    return int(str(raw))


def port_instance(legacy: dict, graph: nx.MultiDiGraph) -> Instance:
    depots: list[Depot] = []
    for idx, d in enumerate(legacy["depots"]):
        depots.append(
            Depot(
                depot_id=-(idx + 1),
                osm_node=_node_id_to_int(d["node_id"]),
                lat=float(d["lat"]),
                lon=float(d["lon"]),
                brand="Blinkit",
            )
        )

    legacy_node_to_depot_id = {
        _node_id_to_int(d["node_id"]): depots[i].depot_id
        for i, d in enumerate(legacy["depots"])
    }

    customers: list[Customer] = []
    for i, c in enumerate(legacy["customers"]):
        home = legacy_node_to_depot_id[_node_id_to_int(c["assigned_depot"])]
        customers.append(
            Customer(
                customer_id=i,
                osm_node=_node_id_to_int(c["node_id"]),
                lat=float(c["lat"]),
                lon=float(c["lon"]),
                demand=int(c["q_i"]),
                e_i=float(c["e_i"]),
                eta_i=float(c["e_i"]) + 10.0,
                service_time=2.0,
                home_depot=home,
            )
        )

    params = legacy.get("parameters", {})
    meta = legacy.get("metadata", {})
    return Instance(
        city="legacy_bengaluru",
        depots=depots,
        customers=customers,
        street_graph=graph,
        super_arcs={},
        Q=int(params.get("Q", meta.get("vehicle_capacity", 2))),
        T_max=float(params.get("T_max", 35.0)),
        H_cap_route=int(params.get("H_cap_route", 8)),
        R_bar=float("inf"),
        H_bar=10**6,
        seed=int(meta.get("seed", 42)),
        meta={"ported_from": "data/04_vrptw_instance.json", "n_riders_legacy": int(meta.get("k_riders", 0))},
    )


def port_solution(legacy_sol: dict, instance: Instance) -> Solution:
    osm_to_customer_id = {c.osm_node: c.customer_id for c in instance.customers}
    osm_to_depot_id = {d.osm_node: d.depot_id for d in instance.depots}

    routes: list[Route] = []
    for k, r in enumerate(legacy_sol["routes"]):
        nodes_int: list[int] = []
        for n in r["nodes"]:
            n_int = _node_id_to_int(n)
            if n_int in osm_to_depot_id:
                nodes_int.append(osm_to_depot_id[n_int])
            elif n_int in osm_to_customer_id:
                nodes_int.append(osm_to_customer_id[n_int])
            else:
                # Unknown intermediate OSM node — keep as raw int but it
                # will fail validation; legacy paths sometimes include
                # transit nodes that the new schema does not allow.
                nodes_int.append(n_int)
        depot_int = _node_id_to_int(r["depot"])
        routes.append(
            Route(
                rider_id=k,
                depot_id=osm_to_depot_id.get(depot_int, depot_int),
                nodes=nodes_int,
                arrivals=[0.0] * len(nodes_int),
            )
        )
    return Solution(
        routes=routes,
        objective=float(legacy_sol.get("summary", {}).get("score", 0.0)),
        constraint_summary={"ported": True},
        solver="legacy",
    )


def write_port_notes(out_dir: Path, instance: Instance, solution: Solution) -> None:
    note = (
        "# Legacy Port Notes\n\n"
        "Source files:\n"
        "- `data/04_vrptw_instance.json`\n"
        "- `data/05_solution.json`\n"
        "- `data/03_graph_final.graphml`\n\n"
        "## Preserved\n"
        f"- {len(instance.depots)} depots (Blinkit brand assumed)\n"
        f"- {len(instance.customers)} customers with (e_i, demand, home_depot)\n"
        f"- {len(solution.routes)} routes with their node sequences\n\n"
        "## Reconstructed (best-effort)\n"
        "- `eta_i = e_i + 10` (10-min q-commerce promise; legacy stored only `e_i`/`l_i`)\n"
        "- `service_time = 2.0` minutes (BASM default; legacy did not record this)\n"
        "- `arrivals = [0.0, ...]` placeholder; legacy did not record per-node arrival times\n\n"
        "## Dropped\n"
        "- Synthetic crash rates (legacy used iRAD-Bengaluru placeholder; not portable to BASM v1)\n"
        "- Lambda objective weights `[0.4, 0.4, 0.2]` (modular schema uses ε-constraint, not weighted sum)\n\n"
        "## Provenance flag\n"
        "`Instance.city == 'legacy_bengaluru'` — this is a marker, not a real city slug.\n"
    )
    (out_dir / "PORT_NOTES.md").write_text(note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="data/04_vrptw_instance.json")
    ap.add_argument("--solution", default="data/05_solution.json")
    ap.add_argument("--graph", default="data/03_graph_final.graphml")
    ap.add_argument("--out", default="data/legacy_ported")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    legacy_inst = json.loads(Path(args.instance).read_text())
    legacy_sol = json.loads(Path(args.solution).read_text())
    G = nx.read_graphml(args.graph) if Path(args.graph).exists() else nx.MultiDiGraph()

    instance = port_instance(legacy_inst, G)
    solution = port_solution(legacy_sol, instance)

    with (out / "instance.pkl").open("wb") as f:
        pickle.dump(instance, f)
    (out / "solution.json").write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "rider_id": r.rider_id,
                        "depot_id": r.depot_id,
                        "nodes": r.nodes,
                        "arrivals": r.arrivals,
                    }
                    for r in solution.routes
                ],
                "objective": solution.objective,
                "constraint_summary": solution.constraint_summary,
                "solver": solution.solver,
            },
            indent=2,
        )
    )
    write_port_notes(out, instance, solution)
    print(f"Ported: {len(instance.depots)} depots, {len(instance.customers)} customers, "
          f"{len(solution.routes)} routes -> {out}")


if __name__ == "__main__":
    main()
