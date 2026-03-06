/* =========================================================
   SA-VRPTW Control Center — Frontend App
   ========================================================= */

// --- Global State ---
let currentConfig = {};
let map, networkLayer, textLayer, markerLayer;
let visNetwork;
let evtSource;

// Element refs
const el = (id) => document.getElementById(id);

// --- Initialization ---
document.addEventListener("DOMContentLoaded", async () => {
  await fetchConfig();
  initMap();
  initVisGraph();
  loadWebData();

  // Resize charts/graphs when tabs switch
  window.addEventListener("resize", () => {
    if (visNetwork) visNetwork.fit();
  });
});

// --- Tab Navigation ---
function showView(viewId, btn) {
  document.querySelectorAll(".view-tab").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".view-container").forEach(c => c.classList.remove("active"));
  btn.classList.add("active");
  el(`view-${viewId}`).classList.add("active");

  if (viewId === "map" && map) map.invalidateSize();
  if (viewId === "graph" && visNetwork) visNetwork.fit();
}

// --- Configuration Sync ---
async function fetchConfig() {
  try {
    const res = await fetch("/api/config");
    currentConfig = await res.json();
    syncUI();
  } catch (err) {
    console.error("Failed to load config", err);
  }
}

let saveTimer;
function schedSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveConfig, 400);
}

function syncUI() {
  const c = currentConfig;
  setToggleActive("irad-toggle", c.irad_mode);
  setToggleActive("cong-toggle", c.congestion_mode);
  el("irad-badge").className = `topbar-badge badge-${c.irad_mode}`;
  el("irad-badge").textContent = c.irad_mode === "synthetic" ? "Synthetic iRAD" : "Real iRAD";
  el("cong-badge").textContent = c.congestion_mode === "SPEED_PROXY" ? "Speed Proxy" : "Google Maps";
  
  el("gmaps-key").value = c.google_maps_api_key || "";
  
  el("l1").value = c.lambda1; el("l1v").innerText = c.lambda1.toFixed(2);
  el("l2").value = c.lambda2; el("l2v").innerText = c.lambda2.toFixed(2);
  el("l3").value = c.lambda3; el("l3v").innerText = c.lambda3.toFixed(2);
  validateLambdas();

  el("n-cust").value = c.n_customers;
  el("k-riders").value = c.k_riders;
  el("q-cap").value = c.vehicle_capacity;
  el("algorithm").value = c.algorithm;

  (c.steps || []).forEach(s => {
    const box = el(`s${s}`);
    if (box) box.checked = true;
  });
}

async function saveConfig() {
  const c = {
    irad_mode: document.querySelector("#irad-toggle .active").dataset.val,
    congestion_mode: document.querySelector("#cong-toggle .active").dataset.val,
    google_maps_api_key: el("gmaps-key").value.trim(),
    lambda1: parseFloat(el("l1").value),
    lambda2: parseFloat(el("l2").value),
    lambda3: parseFloat(el("l3").value),
    n_customers: parseInt(el("n-cust").value),
    k_riders: parseInt(el("k-riders").value),
    vehicle_capacity: parseInt(el("q-cap").value),
    algorithm: el("algorithm").value,
    steps: [1,2,3,4,5].filter(s => el(`s${s}`).checked)
  };
  
  try {
    const res = await fetch("/api/config", {
      method: "POST", body: JSON.stringify(c)
    });
    const data = await res.json();
    currentConfig = data.config;
    syncUI();
  } catch(e) { console.error("Save config failed", e); }
}

// --- UI Actions ---
function setToggleActive(groupId, val) {
  const container = el(groupId);
  if (!container) return;
  container.querySelectorAll(".toggle-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.val === val);
  });
}
function setIrad(val, btn) { setToggleActive("irad-toggle", val); saveConfig(); }
function setCong(val, btn) { setToggleActive("cong-toggle", val); saveConfig(); }
function toggleKeyVis() {
  const inp = el("gmaps-key");
  inp.type = inp.type === "password" ? "text" : "password";
}
function lambdaChanged(changedIdx) {
  const l1 = el("l1"), l2 = el("l2"), l3 = el("l3");
  let vals = [0, parseFloat(l1.value), parseFloat(l2.value), parseFloat(l3.value)];
  
  // enforce sum=1 by making l3 the slack variable (or l2 if l3 was changed)
  if (changedIdx === 3) {
    if (vals[3] + vals[1] > 1) { vals[1] = 1 - vals[3]; }
    vals[2] = 1.0 - vals[1] - vals[3];
  } else {
    if (vals[1] + vals[2] > 1) {
      if (changedIdx===1) vals[2] = 1 - vals[1];
      else vals[1] = 1 - vals[2];
    }
    vals[3] = 1.0 - vals[1] - vals[2];
  }
  
  l1.value = vals[1].toFixed(2); el("l1v").innerText = vals[1].toFixed(2);
  l2.value = vals[2].toFixed(2); el("l2v").innerText = vals[2].toFixed(2);
  l3.value = vals[3].toFixed(2); el("l3v").innerText = vals[3].toFixed(2);
  
  validateLambdas();
  schedSave();
}
function validateLambdas() {
  const sum = parseFloat(el("l1").value) + parseFloat(el("l2").value) + parseFloat(el("l3").value);
  const out = el("lambda-sum");
  out.innerText = `∑ = ${sum.toFixed(2)}${Math.abs(sum-1)<0.01 ? " ✓" : " ✗"}`;
  out.className = Math.abs(sum-1)<0.01 ? "lambda-sum" : "lambda-sum invalid";
}

// --- Pipeline Runner ---
function clearLog() { el("log-console").innerHTML = ""; }

async function runPipeline() {
  const btn = el("run-btn");
  btn.disabled = true;
  btn.classList.add("busy");
  btn.innerText = "⏳ Running Pipeline...";
  el("status-dot").className = "status-dot busy";
  el("status-label").innerText = "Running";
  
  clearLog();
  
  // Connect SSE
  if(evtSource) evtSource.close();
  evtSource = new EventSource("/api/status");
  evtSource.onmessage = (e) => {
    if(e.data === "[ping]") return;
    let html = e.data;
    if(html.includes("[DONE]")) {
      disconnectPipeline();
      setTimeout(loadWebData, 1000); // Reload maps/graphs!
    }
    appendLog(html);
  };
  
  const steps = [1,2,3,4,5].filter(s => el(`s${s}`).checked);
  try {
    await fetch("/api/run", { method: "POST", body: JSON.stringify({steps}) });
  } catch(e) {
    appendLog(`[FAIL] API error: ${e.message}`);
    disconnectPipeline();
  }
}

function disconnectPipeline() {
  if(evtSource) { evtSource.close(); evtSource=null; }
  const btn = el("run-btn");
  btn.disabled = false;
  btn.classList.remove("busy");
  btn.innerText = "⚡ Run Selected Steps";
  el("status-dot").className = "status-dot";
  el("status-label").innerText = "Idle";
}

function appendLog(line) {
  const c = el("log-console");
  const div = document.createElement("div");
  div.innerHTML = line;
  if(line.includes("✓")) div.className = "log-line-ok";
  if(line.includes("✗") || line.includes("Traceback") || line.includes("FAIL")) div.className = "log-line-fail";
  if(line.includes("▶")) div.className = "log-line-step";
  if(line.includes("[DONE]")) div.className = "log-line-done";
  c.appendChild(div);
  c.scrollTop = c.scrollHeight;
}

// --- Data Loaders & Visualization ---
let activeMapLayer = "risk";  // risk | congestion | time
let rawGeoJson = null;
let instanceData = null;

async function loadWebData() {
  try {
    const [resN, resI, resG] = await Promise.all([
      fetch("/api/network"), fetch("/api/instance"), fetch("/api/graph")
    ]);
    
    if (resN.ok) {
      rawGeoJson = await resN.json();
      renderMapNetwork();
      renderCharts();
    }
    if (resI.ok) {
      instanceData = await resI.json();
      renderMapMarkers();
    }
    if (resG.ok) {
      const gData = await resG.json();
      renderVisGraph(gData);
    }
  } catch(e) {
    console.error("Data load failed", e);
  }
}

// --- Leaflet Map ---
function initMap() {
  // Center roughly on Kharagpur IIT
  map = L.map("map", { zoomControl: false }).setView([22.314, 87.310], 13);
  L.control.zoom({ position: 'bottomright' }).addTo(map);
  
  // Dark basemap
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    subdomains: 'abcd',
    maxZoom: 19
  }).addTo(map);
  
  networkLayer = L.layerGroup().addTo(map);
  markerLayer = L.layerGroup().addTo(map);
}

function setLayer(layer, btn) {
  document.querySelectorAll(".layer-btn-group:first-of-type .layer-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  activeMapLayer = layer;
  renderMapNetwork();
}

function toggleMarker(type, btn) {
  btn.classList.toggle("active");
  renderMapMarkers();
}

function getColor(val, min=0, max=1) {
  // Simple green->yellow->red linear interp
  const v = Math.max(0, Math.min(1, (val - min)/(max - min || 1)));
  if (v < 0.5) {
    // green to yellow
    const r = Math.floor(34 + v*2 * (245-34));
    const g = Math.floor(197 - v*2 * (197-158));
    const b = Math.floor(94 + v*2 * (11-94));
    return `rgb(${r},${g},${b})`;
  } else {
    // yellow to red
    const r = Math.floor(245 + (v-0.5)*2 * (239-245));
    const g = Math.floor(158 - (v-0.5)*2 * (158-68));
    const b = Math.floor(11 + (v-0.5)*2 * (68-11));
    return `rgb(${r},${g},${b})`;
  }
}

function renderMapNetwork() {
  if (!rawGeoJson || !map) return;
  networkLayer.clearLayers();
  
  // find max bounds for active metric (for relative coloring)
  let maxVal = 0.01;
  const props = rawGeoJson.features.map(f => f.properties);
  if (activeMapLayer === "time") maxVal = Math.max(...props.map(p => p.t_ij || 0.1));
  
  L.geoJSON(rawGeoJson, {
    style: function(feature) {
      const p = feature.properties;
      let val = p.r_ij;
      let mw = 1.0;
      if (activeMapLayer === "congestion") val = p.c_ij; // c_ij mapped 0-1 range already? wait, speed proxy gives values up to 1. But lower speed = higher congestion.
      if (activeMapLayer === "time") { val = p.t_ij; mw = maxVal; }
      
      const c = getColor(val, 0, mw);
      return { color: c, weight: val > 0.05 ? 3 : 1.5, opacity: val > 0.05 ? 0.9 : 0.3 };
    },
    onEachFeature: function(feature, layer) {
      const p = feature.properties;
      const popup = `
        <div class="popup-title">${p.name || p.highway || "Road Segment"}</div>
        <div class="popup-row"><span class="popup-key">Risk r_ij:</span> <span class="popup-val ${p.r_ij>0.5?'high':''}">${p.r_ij.toFixed(4)}</span></div>
        <div class="popup-row"><span class="popup-key">Congestion c_ij:</span> <span class="popup-val">${p.c_ij.toFixed(4)}</span></div>
        <div class="popup-row"><span class="popup-key">Time t_ij:</span> <span class="popup-val">${p.t_ij.toFixed(3)} m</span></div>
        <div class="popup-row"><span class="popup-key">Length:</span> <span class="popup-val">${p.length_m} m</span></div>
      `;
      layer.bindPopup(popup);
      layer.on('mouseover', function(e) { this.setStyle({weight:5, opacity:1}); });
      layer.on('mouseout', function(e) { networkLayer.resetStyle(this); });
    }
  }).addTo(networkLayer);
}

function renderMapMarkers() {
  if (!instanceData || !map) return;
  markerLayer.clearLayers();
  
  const showDepot = el("mb-depot").classList.contains("active");
  const showCust  = el("mb-cust").classList.contains("active");
  
  if (showDepot) {
    const d = instanceData.depot;
    const depotIcon = L.divIcon({
      className: '',
      html: `<div style="background:var(--accent);border:2px solid #fff;color:#000;width:24px;height:24px;border-radius:50%;text-align:center;line-height:20px;font-weight:bold;box-shadow:0 0 10px var(--accent);">D</div>`,
      iconSize: [24,24]
    });
    L.marker([d.lat, d.lon], {icon: depotIcon, zIndexOffset: 1000})
     .bindPopup(`<b>Depot</b><br>TW: ${d.e_0} - ${d.l_0}`)
     .addTo(markerLayer);
  }
  
  if (showCust) {
    instanceData.customers.forEach((c, idx) => {
      const cIcon = L.divIcon({
        className: '',
        html: `<div style="background:#58a6ff;border:1px solid #000;color:#fff;width:18px;height:18px;border-radius:50%;text-align:center;line-height:16px;font-size:10px;">${idx+1}</div>`,
        iconSize: [18,18]
      });
      L.marker([c.lat, c.lon], {icon: cIcon})
       .bindPopup(`<b>Customer ${idx+1}</b><br>Demand (q): ${c.q_i}<br>Time Window: [${c.e_i}, ${c.l_i}]<br>Min TT (depot): ${c.tt_from_depot.toFixed(1)}m`)
       .addTo(markerLayer);
    });
  }
}

// --- Vis.js Graph Explorer ---
function initVisGraph() {
  // Container only
}

function renderVisGraph(data) {
  const container = el('graph-container');
  visNetwork = new vis.Network(container, data, data.options || {});
  
  // Populate stats
  if(instanceData) {
    el("gs-n").innerText = instanceData.customers.length;
    el("gs-k").innerText = instanceData.parameters.K;
    el("gs-q").innerText = instanceData.parameters.Q;
    el("gs-l1").innerText = instanceData.parameters.lambda1.toFixed(2);
    el("gs-l2").innerText = instanceData.parameters.lambda2.toFixed(2);
    el("gs-l3").innerText = instanceData.parameters.lambda3.toFixed(2);
  }
  if(rawGeoJson) {
    const riskey = rawGeoJson.features.filter(f=>f.properties.r_ij > 0).length;
    el("gs-risk").innerText = `${riskey} / ${rawGeoJson.features.length}`;
  }
}

// --- ChartJS Stats ---
let charts = {};
function createHist(id, data, label, color) {
  if (charts[id]) charts[id].destroy();
  const ctx = el(id).getContext('2d');
  
  // simple binning
  const bins = 20;
  const min = Math.min(...data), max = Math.max(...data);
  const step = (max - min) / bins || 1;
  const counts = new Array(bins).fill(0);
  data.forEach(v => {
    let b = Math.floor((v - min) / step);
    if (b >= bins) b = bins - 1;
    counts[b]++;
  });
  
  const labels = counts.map((_, i) => (min + i*step).toFixed(1));
  
  charts[id] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: label,
        data: counts,
        backgroundColor: color,
        borderWidth: 0,
        barPercentage: 1.0,
        categoryPercentage: 0.95
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#64748b", font: {size: 9} } },
        y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#64748b", font: {size: 9} } }
      }
    }
  });
}

function renderCharts() {
  if (!rawGeoJson || !instanceData) return;
  const props = rawGeoJson.features.map(f => f.properties);
  const r_ijs = props.map(p => p.r_ij).filter(v => v > 0); // only show >0 for clarity
  const t_ijs = props.map(p => p.t_ij);
  const c_ijs = props.map(p => p.c_ij);
  
  const slacks = instanceData.customers.map(c => c.l_i - c.tt_from_depot);
  
  createHist("chart-risk", r_ijs, "Risk > 0", "rgba(239, 68, 68, 0.8)");
  createHist("chart-time", t_ijs, "Travel Time", "rgba(0, 212, 255, 0.8)");
  createHist("chart-cong", c_ijs, "Congestion", "rgba(245, 158, 11, 0.8)");
  createHist("chart-slack", slacks, "TW Slack", "rgba(34, 197, 94, 0.8)");
}
