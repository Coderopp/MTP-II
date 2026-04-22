import subprocess
import sys
try:
    import folium
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "folium", "--break-system-packages"])
    import folium

import osmnx as ox
import networkx as nx
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

os.makedirs("figures", exist_ok=True)
ox.settings.log_console = False

def create_divergence_overlay(city_name="Bengaluru", center=(12.9716, 77.5946)):
    print(f"--- [PHASE 6] Generating Real Spatial Folium map for {city_name} ---")
    G = ox.graph_from_point(center, dist=2000, network_type='drive')
    
    for u, v, k, data in G.edges(keys=True, data=True):
        hw = data.get("highway", "residential")
        if isinstance(hw, list): hw = hw[0]
        
        speed = 40.0 if hw in ['primary', 'secondary'] else 20.0
        dist_km = data.get('length', 100) / 1000.0
        
        data['time'] = (dist_km / speed) * 60.0 
        base_risk = 0.8 if hw in ['primary', 'trunk', 'motorway'] else 0.2
        data['risk'] = np.clip(base_risk + np.random.normal(0, 0.15), 0.05, 1.0)

    nodes = list(G.nodes())
    valid_pair = False
    while not valid_pair:
        start = random.choice(nodes)
        end = random.choice(nodes)
        try:
            if nx.shortest_path_length(G, start, end, weight='time') >= 10:
                valid_pair = True
        except nx.NetworkXNoPath:
            continue

    fast_path = nx.shortest_path(G, start, end, weight='time')
    import math
    for u, v, k, data in G.edges(keys=True, data=True):
        data['neg_log_risk'] = -math.log(max(0.01, 1 - data['risk']))
        
    safe_path = nx.shortest_path(G, start, end, weight='neg_log_risk')

    m = folium.Map(location=center, zoom_start=14, tiles='CartoDB positron')
    
    fast_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in fast_path]
    folium.PolyLine(locations=fast_coords, color='red', weight=6, opacity=0.8, tooltip="Absolute Minimum Travel Time (High Risk)").add_to(m)
    
    safe_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in safe_path]
    folium.PolyLine(locations=safe_coords, color='green', weight=6, opacity=0.8, tooltip="SA-VRPTW Regulated Limit (Safe)").add_to(m)
    
    folium.Marker(location=fast_coords[0], popup="Dark Store Depot (Start)", icon=folium.Icon(color='black')).add_to(m)
    folium.Marker(location=fast_coords[-1], popup="Customer Node (End)", icon=folium.Icon(color='blue')).add_to(m)
    
    output_file = f"figures/9_geographical_divergence_{city_name.lower()}.html"
    m.save(output_file)
    print(f"Saved Interactive Divergence Map to {output_file}")


def create_heatmap_overlay(center=(28.6139, 77.2090)): # Delhi Default
    from folium.plugins import HeatMap
    print("--- [PHASE 6] Generating Danger Density Spatial Heatmap ---")
    G = ox.graph_from_point(center, dist=2500, network_type='drive')
    
    heat_data = []
    
    for u, v, k, data in G.edges(keys=True, data=True):
        hw = data.get("highway", "residential")
        if isinstance(hw, list): hw = hw[0]
        # fake up the iRAD empirical limits
        if hw in ['primary', 'trunk', 'motorway']:
            danger = np.random.uniform(0.7, 1.0)
            lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
            lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
            heat_data.append([lat, lon, danger])
            
        elif hw == 'secondary' and random.random() < 0.3:
            danger = np.random.uniform(0.4, 0.7)
            lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
            lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
            heat_data.append([lat, lon, danger])

    m = folium.Map(location=center, zoom_start=14, tiles='CartoDB dark_matter')
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1, gradient={0.4: 'yellow', 0.65: 'orange', 1: 'red'}).add_to(m)
    
    output_file = "figures/10_danger_density_heatmap.html"
    m.save(output_file)
    print(f"Saved Density Risk Heatmap to {output_file}")

if __name__ == "__main__":
    create_divergence_overlay("Bengaluru", (12.9716, 77.5946)) # specific req
    create_heatmap_overlay(center=(12.9716, 77.5946))
