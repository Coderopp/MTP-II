"""
web/server.py — Flask backend for the SA-VRPTW Web Control Center.

Start with:  python3 web/server.py
Open:        http://localhost:5050
"""

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEB_DIR     = Path(__file__).parent
REPO_ROOT   = WEB_DIR.parent
CODE_DIR    = REPO_ROOT / "code"
DATA_DIR    = REPO_ROOT / "data"
WEB_DATA    = WEB_DIR / "static" / "data"
CONFIG_PATH = WEB_DIR / "pipeline_config.json"

sys.path.insert(0, str(CODE_DIR))

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------------
# SSE log queue (one global queue per server lifetime)
# ---------------------------------------------------------------------------
_log_queue: queue.Queue = queue.Queue(maxsize=500)


def _enqueue(line: str) -> None:
    try:
        _log_queue.put_nowait(line)
    except queue.Full:
        pass


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "irad_mode": "synthetic",
    "irad_csv_path": "data/irad_accidents.csv",
    "congestion_mode": "SPEED_PROXY",
    "google_maps_api_key": "",
    "n_customers": 20,
    "k_riders": 5,
    "vehicle_capacity": 10,
    "lambda1": 0.40,
    "lambda2": 0.40,
    "lambda3": 0.20,
    "algorithm": "GA",
    "steps": [1, 2, 3, 4, 5],
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        # merge with defaults for any missing keys
        return {**DEFAULT_CONFIG, **cfg}
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def apply_config_to_scripts(cfg: dict) -> None:
    """Patch pipeline script config values from the web UI config."""
    # Patch 04_instance_generator.py constants via env vars (passed to subprocess)
    pass  # env vars handled in _run_pipeline


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
SCRIPT_MAP = {
    1: CODE_DIR / "01_osm_graph.py",
    2: CODE_DIR / "02_irad_risk.py",
    3: CODE_DIR / "03_congestion.py",
    4: CODE_DIR / "04_instance_generator.py",
    5: CODE_DIR / "05_validate_instance.py",
    6: CODE_DIR / "06_export_geojson.py",
}

_pipeline_lock = threading.Lock()
_pipeline_running = False


def _run_pipeline(cfg: dict, steps: list[int]) -> None:
    global _pipeline_running
    with _pipeline_lock:
        _pipeline_running = True

    # Build environment overrides from config
    env = os.environ.copy()
    env["CONGESTION_MODE"] = cfg.get("congestion_mode", "SPEED_PROXY")
    gmaps_key = cfg.get("google_maps_api_key", "")
    if gmaps_key:
        env["GOOGLE_MAPS_API_KEY"] = gmaps_key

    # Patch n_customers / k_riders / vehicle_capacity / lambdas via env
    env["SAVRPTW_N_CUSTOMERS"]  = str(cfg.get("n_customers", 20))
    env["SAVRPTW_K_RIDERS"]     = str(cfg.get("k_riders", 5))
    env["SAVRPTW_CAPACITY"]     = str(cfg.get("vehicle_capacity", 10))
    env["SAVRPTW_LAMBDA1"]      = str(cfg.get("lambda1", 0.4))
    env["SAVRPTW_LAMBDA2"]      = str(cfg.get("lambda2", 0.4))
    env["SAVRPTW_LAMBDA3"]      = str(cfg.get("lambda3", 0.2))
    env["SAVRPTW_IRAD_MODE"]    = cfg.get("irad_mode", "synthetic")

    # Always export to web after running steps
    run_steps = list(steps) + [6]

    try:
        for step in sorted(set(run_steps)):
            script = SCRIPT_MAP.get(step)
            if not script or not script.exists():
                _enqueue(f"[SKIP] Step {step}: script not found\n")
                continue

            _enqueue(f"\n{'─'*48}\n")
            _enqueue(f"▶  STEP {step}: {script.name}\n")
            _enqueue(f"{'─'*48}\n")

            proc = subprocess.Popen(
                [sys.executable, str(script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(REPO_ROOT),
                env=env,
            )
            for line in proc.stdout:
                _enqueue(line)
            proc.wait()

            if proc.returncode != 0:
                _enqueue(f"\n✗ Step {step} failed (exit {proc.returncode})\n")
                break
            else:
                _enqueue(f"\n✓ Step {step} completed\n")

        _enqueue("\n\n[DONE] Pipeline finished.\n")
    finally:
        _pipeline_running = False


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.get("/api/config")
def get_config():
    return jsonify(load_config())


@app.post("/api/config")
def set_config():
    data = request.get_json(force=True)
    cfg = load_config()
    cfg.update(data)
    # Enforce lambda sum = 1
    total = cfg["lambda1"] + cfg["lambda2"] + cfg["lambda3"]
    if abs(total - 1.0) > 0.01:
        cfg["lambda3"] = round(1.0 - cfg["lambda1"] - cfg["lambda2"], 3)
    save_config(cfg)
    return jsonify({"status": "ok", "config": cfg})


@app.post("/api/run")
def run_pipeline():
    global _pipeline_running
    if _pipeline_running:
        return jsonify({"status": "busy", "message": "Pipeline already running"}), 409

    data = request.get_json(force=True) or {}
    cfg = load_config()
    steps = data.get("steps", cfg.get("steps", list(range(1, 6))))

    # Clear old log queue
    while not _log_queue.empty():
        try:
            _log_queue.get_nowait()
        except queue.Empty:
            break

    threading.Thread(
        target=_run_pipeline, args=(cfg, steps), daemon=True
    ).start()
    return jsonify({"status": "started", "steps": steps})


@app.get("/api/status")
def status_stream():
    """Server-Sent Events stream of pipeline log lines."""
    def generate():
        yield "data: [Pipeline log stream connected]\n\n"
        while True:
            try:
                line = _log_queue.get(timeout=30)
                escaped = line.replace("\n", "<br>")
                yield f"data: {escaped}\n\n"
            except queue.Empty:
                yield "data: [ping]\n\n"  # keep-alive
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.get("/api/network")
def get_network():
    path = WEB_DATA / "network.geojson"
    if not path.exists():
        return jsonify({"error": "network.geojson not found — run pipeline first"}), 404
    return send_from_directory(str(WEB_DATA), "network.geojson",
                               mimetype="application/geo+json")


@app.get("/api/instance")
def get_instance():
    path = WEB_DATA / "instance.json"
    if not path.exists():
        return jsonify({"error": "instance.json not found — run pipeline first"}), 404
    return send_from_directory(str(WEB_DATA), "instance.json")


@app.get("/api/graph")
def get_graph():
    path = WEB_DATA / "graph_vis.json"
    if not path.exists():
        return jsonify({"error": "graph_vis.json not found — run pipeline first"}), 404
    return send_from_directory(str(WEB_DATA), "graph_vis.json")


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Export web data on startup if available
    if (DATA_DIR / "03_graph_final.graphml").exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "export", str(CODE_DIR / "06_export_geojson.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.main()
        except Exception as e:
            print(f"  [warn] Could not auto-export: {e}")

    print("\n  SA-VRPTW Control Center")
    print("  http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
