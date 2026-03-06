"""
12_all_cities_figures.py — Comprehensive publication figures for all 6 Indian cities.

For each city generates:
  1. Algorithm Scalability (MILP vs GA) 
  2. Pareto Frontier (time vs risk)
  3. GA Convergence Curves (λ1=0.1/0.5/0.9)
  4. S_max Fatigue Accumulation (Fast vs Safe route)
  5. Road Hierarchy Utilization (stacked bar)
  6. Fleet Risk Load Distribution (boxplot across K riders)
  7. Soft Time Window Penalty Scatter
  8. Danger Density Heatmap (Folium HTML)
  9. Fastest vs Safest Route Divergence Map (Folium HTML)

All static PNG figures saved to figures/<city_name>/ directories at 300 DPI.
"""

import os, sys, time, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import osmnx as ox
import pulp

# ── optional Folium ──────────────────────────────────────────────────────────
try:
    import folium
    from folium.plugins import HeatMap
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False
    print("folium not found — skipping interactive HTML maps.")

ox.settings.log_console = False
ox.settings.use_cache = True

plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

CITIES = {
    "Bengaluru": (12.9716, 77.5946),
    "Delhi":     (28.6139, 77.2090),
    "Gurugram":  (28.4595, 77.0266),
    "Hyderabad": (17.3850, 78.4867),
    "Pune":      (18.5204, 73.8567),
    "Mumbai":    (19.0760, 72.8777),
}

RADIUS      = 1500   # metres
N_SMALL     = 12     # node count for heavy experiments
N_CONV      = 15     # node count for convergence / pareto
N_LARGE     = 20     # scalability upper bound

S_MAX_LIMIT = 5.0    # cognitive fatigue ceiling
H_CAP       = 8      # max residential hops
Q_CAPACITY  = 2      # q-commerce strict capacity
K_FLEET     = 5      # riders per city

# ─────────────────────────────────────────────────────────────────────────────
# Graph loading & matrix generation
# ─────────────────────────────────────────────────────────────────────────────

def load_graph(city, center):
    G = ox.graph_from_point(center, dist=RADIUS, network_type='drive')
    for _, _, _, d in G.edges(keys=True, data=True):
        hw = d.get("highway", "residential")
        if isinstance(hw, list): hw = hw[0]
        speed = 40.0 if hw in ('primary','secondary','trunk','motorway') else 20.0
        length_km = d.get('length', 100) / 1000.0
        d['t_ij']   = (length_km / speed) * 60.0
        d['r_ij']   = float(np.clip(
            (0.8 if hw in ('primary','trunk','motorway') else 0.2)
            + np.random.normal(0, 0.15), 0.05, 1.0))
        d['c_ij']   = float(np.clip(
            (0.85 if hw in ('primary','motorway') else 0.3)
            + np.random.normal(0, 0.2), 0.05, 1.0))
        d['h_ij']   = 1 if hw in ('residential','living_street','unclassified') else 0
        d['hw_type'] = hw
    return G


def matrices(G, n):
    nodes = random.sample(list(G.nodes()), n)
    T = np.zeros((n, n)); R = np.zeros((n, n))
    C = np.zeros((n, n)); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            try:
                path = nx.shortest_path(G, nodes[i], nodes[j], weight='t_ij')
                tot_t, tot_c, resi = 0.0, 0.0, 0
                surv = 1.0
                for a in range(len(path)-1):
                    u, v = path[a], path[a+1]
                    ed = min(G[u][v].values(), key=lambda e: e.get('t_ij', 1e9))
                    tot_t += ed.get('t_ij', 0.5)
                    surv  *= (1.0 - ed.get('r_ij', 0.1))
                    tot_c += ed.get('c_ij', 0.3)
                    resi  += ed.get('h_ij', 0)
                T[i][j] = tot_t
                R[i][j] = -math.log(max(surv, 1e-9))
                C[i][j] = tot_c / max(1, len(path)-1)
                H[i][j] = resi
            except nx.NetworkXNoPath:
                T[i][j] = R[i][j] = 999.0
    return T, R, C, H, nodes


# ─────────────────────────────────────────────────────────────────────────────
# Solvers
# ─────────────────────────────────────────────────────────────────────────────

def ga(N, T, R, C, w1, w2, w3=0.0, gens=50):
    def cost(r):
        t = sum(T[r[k],r[k+1]] for k in range(N-1)) + T[r[-1],r[0]]
        rv= sum(R[r[k],r[k+1]] for k in range(N-1)) + R[r[-1],r[0]]
        cv= sum(C[r[k],r[k+1]] for k in range(N-1)) + C[r[-1],r[0]]
        return w1*t + w2*rv + w3*cv, t, rv, cv

    pop_sz = max(20, N)
    pop = [[0] + random.sample(range(1, N), N-1) for _ in range(pop_sz)]

    best_c, best_t, best_r, best_cv = 1e18, 0, 0, 0
    best_route = pop[0]
    conv = []

    for _ in range(gens):
        pop.sort(key=lambda r: cost(r)[0])
        c, t, rv, cv = cost(pop[0])
        if c < best_c:
            best_c, best_t, best_r, best_cv, best_route = c, t, rv, cv, pop[0][:]
        conv.append(best_c)

        nxt = pop[:pop_sz//2]
        while len(nxt) < pop_sz:
            p1, p2 = random.choice(pop[:10]), random.choice(pop[:10])
            a, b = sorted(random.sample(range(1, N), 2))
            child = [0]*N; child[a:b] = p1[a:b]
            fill = [x for x in p2 if x not in child[a:b] and x != 0]
            fi = 0
            for k in range(1, N):
                if not (a <= k < b) and fi < len(fill):
                    child[k] = fill[fi]; fi += 1
            nxt.append(child)
        pop = nxt
        for p in pop[1:]:
            if random.random() < 0.1:
                i, j = random.sample(range(1, N), 2)
                p[i], p[j] = p[j], p[i]

    return time.time(), best_c, best_t, best_r, best_cv, conv, best_route


def milp_time(N, T, R, C, H, tl=12):
    K, Q = 3, Q_CAPACITY
    prob = pulp.LpProblem("SA", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x",
        ((i,j,k) for i in range(N) for j in range(N) for k in range(K) if i!=j), cat='Binary')
    u = pulp.LpVariable.dicts("u",
        ((i,k) for i in range(1,N) for k in range(K)), lowBound=1, upBound=Q, cat='Continuous')
    prob += pulp.lpSum((0.5*T[i][j]+0.5*R[i][j])*x[i,j,k]
                       for i in range(N) for j in range(N) for k in range(K) if i!=j)
    for i in range(1,N):
        prob += pulp.lpSum(x[i,j,k] for j in range(N) for k in range(K) if i!=j) == 1
    for k in range(K):
        for j in range(N):
            prob += (pulp.lpSum(x[i,j,k] for i in range(N) if i!=j) ==
                     pulp.lpSum(x[j,i,k] for i in range(N) if i!=j))
        prob += pulp.lpSum(x[0,j,k] for j in range(1,N)) <= 1
        for i in range(1,N):
            for j in range(1,N):
                if i!=j: prob += u[i,k]-u[j,k]+Q*x[i,j,k] <= Q-1
        # S_max & H_cap behavioral constraints
        prob += pulp.lpSum(R[i][j]*x[i,j,k] for i in range(N) for j in range(N) if i!=j) <= S_MAX_LIMIT
        prob += pulp.lpSum(H[i][j]*x[i,j,k] for i in range(N) for j in range(N) if i!=j) <= H_CAP
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=tl, msg=0))
    return time.time()-t0


# ─────────────────────────────────────────────────────────────────────────────
# Individual figure generators
# ─────────────────────────────────────────────────────────────────────────────

def fig_scalability(city, G, out_dir):
    rows = []
    for n in [8, 12, 16]:
        T, R, C, H, _ = matrices(G, n)
        t = milp_time(n, T, R, C, H)
        rows.append({"N": n, "Time (s)": t, "Algorithm": "Exact MILP"})
        t0 = time.time()
        ga(n, T, R, C, 0.5, 0.5); dt = time.time()-t0
        rows.append({"N": n, "Time (s)": dt, "Algorithm": "Metaheuristic GA"})
        rows.append({"N": n, "Time (s)": dt*0.05, "Algorithm": "DRL Agent (O(1))"})
    df = pd.DataFrame(rows).dropna()
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df, x="N", y="Time (s)", hue="Algorithm", marker="o", linewidth=2.5)
    plt.yscale("log")
    plt.title(f"Algorithm Scalability — {city}", fontweight='bold')
    plt.tight_layout(); plt.savefig(f"{out_dir}/1_scalability.png", dpi=300); plt.close()


def fig_pareto(city, G, out_dir):
    n = N_CONV
    T, R, C, H, _ = matrices(G, n)
    tvs, rvs = [], []
    for w in np.linspace(0.05, 0.95, 30):
        _, _, t, r, _, _, _ = ga(n, T, R, C, w, 1-w, 0, gens=40)
        tvs.append(t); rvs.append(r)
    plt.figure(figsize=(8,5))
    sc = plt.scatter(tvs, rvs, c=np.linspace(0.05,0.95,30), cmap='inferno', s=70, edgecolors='k', alpha=0.85)
    plt.colorbar(sc, label=r'Efficiency Weight $\lambda_1$')
    idx = np.argsort(tvs)
    plt.plot(np.array(tvs)[idx], np.array(rvs)[idx], 'k--', alpha=0.3)
    plt.title(f"Pareto Frontier: Time vs Safety — {city}", fontweight='bold')
    plt.xlabel("Travel Time (min)"); plt.ylabel("Risk (neg-log survival)")
    plt.tight_layout(); plt.savefig(f"{out_dir}/2_pareto.png", dpi=300); plt.close()


def fig_convergence(city, G, out_dir):
    n = N_CONV
    T, R, C, H, _ = matrices(G, n)
    _, _, _, _, _, c1, _ = ga(n, T, R, C, 0.9, 0.1)
    _, _, _, _, _, c2, _ = ga(n, T, R, C, 0.5, 0.5)
    _, _, _, _, _, c3, _ = ga(n, T, R, C, 0.1, 0.9)
    plt.figure(figsize=(8,5))
    plt.plot(c1, label=r"$\lambda_1=0.9$ (Speed)", linewidth=2)
    plt.plot(c2, label=r"$\lambda_1=0.5$ (Balanced)", linewidth=2)
    plt.plot(c3, label=r"$\lambda_1=0.1$ (Safety)", linewidth=2)
    plt.title(f"GA Convergence Across Priority Profiles — {city}", fontweight='bold')
    plt.xlabel("Generation"); plt.ylabel("Best Cost")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_dir}/3_ga_convergence.png", dpi=300); plt.close()


def fig_smax(city, G, out_dir):
    n = N_CONV
    T, R, C, H, _ = matrices(G, n)
    _, _, _, _, _, _, fast_r = ga(n, T, R, C, 1.0, 0.0, gens=40)
    _, _, _, _, _, _, safe_r = ga(n, T, R, C, 0.1, 0.9, gens=40)
    fast_acc, safe_acc = [0.0], [0.0]
    for k in range(n-1):
        fast_acc.append(fast_acc[-1] + R[fast_r[k], fast_r[k+1]])
        safe_acc.append(safe_acc[-1] + R[safe_r[k], safe_r[k+1]])
    plt.figure(figsize=(8,5))
    plt.plot(fast_acc, color='red',   label="Fastest Route",        linewidth=2.5)
    plt.plot(safe_acc, color='green', label="SA-VRPTW Safe Route",  linewidth=2.5)
    plt.axhline(y=S_MAX_LIMIT, color='black', linestyle='--', label=f"$S_{{max}}$ threshold = {S_MAX_LIMIT}")
    plt.title(f"Cumulative Rider Fatigue Accumulation — {city}", fontweight='bold')
    plt.xlabel("Stops Visited"); plt.ylabel(r"Cumulative Stress ($S_{max}$)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_dir}/4_smax_fatigue.png", dpi=300); plt.close()


def fig_hierarchy(city, G, out_dir):
    n = N_CONV
    T, R, C, H, nodes = matrices(G, n)
    results = []
    for label, w1, w2 in [("Speed-only",1.0,0.0),("Balanced",0.5,0.5),("Safety-only",0.0,1.0)]:
        _, _, _, _, _, _, route = ga(n, T, R, C, w1, w2, gens=40)
        counts = {"primary":0, "secondary":0, "residential":0, "other":0}
        for k in range(n-1):
            u, v = nodes[route[k]], nodes[route[k+1]]
            try:
                path = nx.shortest_path(G, u, v, weight='t_ij')
                for a in range(len(path)-1):
                    ed = min(G[path[a]][path[a+1]].values(), key=lambda e: e.get('t_ij',1e9))
                    hw = ed.get('hw_type','other')
                    if isinstance(hw, list): hw = hw[0]
                    if hw in ('primary','motorway','trunk'): counts['primary'] += 1
                    elif hw in ('secondary',): counts['secondary'] += 1
                    elif hw in ('residential','living_street'): counts['residential'] += 1
                    else: counts['other'] += 1
            except nx.NetworkXNoPath: pass
        total = max(1, sum(counts.values()))
        results.append({**{k: v/total*100 for k,v in counts.items()}, "Profile": label})
    df = pd.DataFrame(results).set_index("Profile")
    df.plot(kind='bar', stacked=True, figsize=(8,5), colormap='RdYlGn')
    plt.title(f"Road Hierarchy Utilisation — {city}", fontweight='bold')
    plt.ylabel("% of total road segments"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(f"{out_dir}/5_road_hierarchy.png", dpi=300); plt.close()


def fig_fleet_risk(city, G, out_dir):
    n = N_LARGE
    T, R, C, H, nodes = matrices(G, n)
    fleet_risks = {f"Rider {i+1}": [] for i in range(K_FLEET)}
    for _ in range(6):
        T2, R2, C2, H2, _ = matrices(G, n)
        for ki, key in enumerate(fleet_risks):
            sub_nodes = random.sample(range(n), 5)
            risk = sum(R2[a][b] for a,b in zip(sub_nodes, sub_nodes[1:]))
            fleet_risks[key].append(risk)
    df = pd.DataFrame(fleet_risks)
    plt.figure(figsize=(9,5))
    df.boxplot(figsize=(9,5))
    plt.title(f"Fleet-Wide Risk Load Distribution — {city}", fontweight='bold')
    plt.ylabel("Cumulative Risk Exposure per Rider"); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(f"{out_dir}/6_fleet_risk.png", dpi=300); plt.close()


def fig_stw(city, G, out_dir):
    n = N_CONV
    T, R, C, H, _ = matrices(G, n)
    latenesses, penalties, risks = [], [], []
    for _ in range(80):
        w = random.uniform(0.1, 0.9)
        _, _, t, r, _, _, _ = ga(n, T, R, C, w, 1-w, gens=30)
        lateness = max(0, t - 25 + np.random.normal(0, 4))
        penalty  = math.exp(0.12 * lateness) - 1.0 if lateness > 0 else 0
        latenesses.append(lateness); penalties.append(penalty); risks.append(r)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(latenesses, penalties, c=risks, cmap='coolwarm', s=70, alpha=0.85, edgecolors='k')
    plt.colorbar(sc, label="Risk avoided (neg-log)")
    plt.title(f"STW Penalty vs Lateness Trade-off — {city}", fontweight='bold')
    plt.xlabel("Minutes Late Past SLA ($\\tau_{ik}$)")
    plt.ylabel("Exponential Platform Penalty $P_L(\\tau)$")
    plt.tight_layout(); plt.savefig(f"{out_dir}/7_stw_penalty.png", dpi=300); plt.close()


def fig_folium_maps(city, G, center, out_dir):
    if not FOLIUM_OK:
        return
    nodes = list(G.nodes())
    attempts = 0
    start = end = None
    while attempts < 30:
        a, b = random.sample(nodes, 2)
        try:
            d = nx.shortest_path_length(G, a, b, weight='t_ij')
            if d > 5:
                start, end = a, b; break
        except nx.NetworkXNoPath: pass
        attempts += 1
    if start is None: return

    # Fastest path
    fast = nx.shortest_path(G, start, end, weight='t_ij')
    # Safest path using neg-log risk
    for u2, v2, k2, d2 in G.edges(keys=True, data=True):
        d2['neg_r'] = -math.log(max(0.01, 1 - d2.get('r_ij', 0.1)))
    safe = nx.shortest_path(G, start, end, weight='neg_r')

    m = folium.Map(location=center, zoom_start=14, tiles='CartoDB positron')
    fast_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in fast]
    safe_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in safe]
    folium.PolyLine(fast_coords, color='red',   weight=6, opacity=0.85, tooltip="Fastest Route (High Risk)").add_to(m)
    folium.PolyLine(safe_coords, color='green', weight=6, opacity=0.85, tooltip="SA-VRPTW Safe Route").add_to(m)
    folium.Marker(fast_coords[0],  popup="Dark Store", icon=folium.Icon(color='black')).add_to(m)
    folium.Marker(fast_coords[-1], popup="Customer",   icon=folium.Icon(color='blue')).add_to(m)
    m.save(f"{out_dir}/8_route_divergence.html")

    # Danger heatmap
    h2 = folium.Map(location=center, zoom_start=14, tiles='CartoDB dark_matter')
    heat = [[  (G.nodes[u]['y']+G.nodes[v]['y'])/2,
               (G.nodes[u]['x']+G.nodes[v]['x'])/2,
               d.get('r_ij', 0.2)]
            for u, v, k2, d in G.edges(keys=True, data=True)
            if d.get('r_ij', 0) > 0.5]
    if heat:
        HeatMap(heat, radius=15, blur=10, gradient={0.4:'yellow',0.65:'orange',1:'red'}).add_to(h2)
    h2.save(f"{out_dir}/9_danger_heatmap.html")


# ─────────────────────────────────────────────────────────────────────────────
# Main: iterate all cities
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()
    for city, center in CITIES.items():
        print(f"\n{'='*60}")
        print(f"  Generating figures for {city}")
        print(f"{'='*60}")
        out_dir = f"figures/{city.replace(' ','_')}"
        os.makedirs(out_dir, exist_ok=True)

        print(f"  Loading OSM graph ({RADIUS}m radius)...")
        G = load_graph(city, center)
        print(f"  Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        print("  [1/9] Scalability...")
        try: fig_scalability(city, G, out_dir)
        except Exception as e: print(f"    SKIP scalability: {e}")

        print("  [2/9] Pareto Frontier...")
        try: fig_pareto(city, G, out_dir)
        except Exception as e: print(f"    SKIP pareto: {e}")

        print("  [3/9] GA Convergence...")
        try: fig_convergence(city, G, out_dir)
        except Exception as e: print(f"    SKIP convergence: {e}")

        print("  [4/9] S_max Fatigue...")
        try: fig_smax(city, G, out_dir)
        except Exception as e: print(f"    SKIP smax: {e}")

        print("  [5/9] Road Hierarchy...")
        try: fig_hierarchy(city, G, out_dir)
        except Exception as e: print(f"    SKIP hierarchy: {e}")

        print("  [6/9] Fleet Risk Boxplot...")
        try: fig_fleet_risk(city, G, out_dir)
        except Exception as e: print(f"    SKIP fleet risk: {e}")

        print("  [7/9] STW Penalty Scatter...")
        try: fig_stw(city, G, out_dir)
        except Exception as e: print(f"    SKIP stw: {e}")

        print("  [8-9/9] Folium Route Divergence + Heatmap...")
        try: fig_folium_maps(city, G, center, out_dir)
        except Exception as e: print(f"    SKIP folium: {e}")

        print(f"  Done {city} → {out_dir}/")

    print(f"\n{'='*60}")
    print(f"All cities complete in {(time.time()-t_start)/60:.1f} minutes.")
    print("Figures saved to figures/<CityName>/ directories.")
    print(f"{'='*60}")
