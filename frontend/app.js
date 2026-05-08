/**
 * app.js  v2 — PipeGuard AI Production Dashboard
 * ─────────────────────────────────────────────────
 * Changes from v1:
 *   - Auth guard: redirects to login.html if no JWT
 *   - All API calls include Authorization: Bearer header
 *   - 401 responses → automatic redirect to login
 *   - SSE stream passes token as query param
 *   - Logout button handler
 *   - Acknowledge anomaly event buttons in history log
 *   - CSV export buttons
 *   - /metrics endpoint populates aggregate stats
 */

"use strict";

const API_BASE   = "http://127.0.0.1:8000";
const MAX_POINTS = 80;
const SEQ_LEN    = 30;

// ──────────────────────────────────────────────────────────────────
//  Auth helpers
// ──────────────────────────────────────────────────────────────────

function getToken()    { return localStorage.getItem("pipeguard_token"); }
function getUsername() { return localStorage.getItem("pipeguard_username") || "operator"; }
function getRole()     { return localStorage.getItem("pipeguard_role") || "operator"; }
function getFullName() { return localStorage.getItem("pipeguard_full_name") || getUsername(); }

function authHeaders() {
  return {
    "Content-Type":  "application/json",
    "Authorization": `Bearer ${getToken()}`,
  };
}

function logout() {
  localStorage.removeItem("pipeguard_token");
  localStorage.removeItem("pipeguard_username");
  localStorage.removeItem("pipeguard_role");
  localStorage.removeItem("pipeguard_full_name");
  window.location.href = "login.html";
}

/** Wrap fetch — auto-redirect on 401 */
async function apiFetch(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: { ...authHeaders(), ...(options.headers || {}) },
  });
  if (res.status === 401) {
    logout();
    return null;
  }
  return res;
}

// ──────────────────────────────────────────────────────────────────
//  Auth guard — must run before anything else
// ──────────────────────────────────────────────────────────────────

(function authGuard() {
  if (!getToken()) {
    window.location.href = "login.html";
  }
})();

// ──────────────────────────────────────────────────────────────────
//  State
// ──────────────────────────────────────────────────────────────────

let sensorChart  = null;
let scoreChart   = null;
let sseSource    = null;
let isPaused     = false;
let totalReadings = 0;
let totalLeaks    = 0;

const readingBuffer = [];

// ──────────────────────────────────────────────────────────────────
//  DOM helpers
// ──────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

// Null-safe setter helpers
function setText(id, val) { const el = $(id); if (el) el.textContent = val; }
function setHTML(id, val) { const el = $(id); if (el) el.innerHTML = val; }

// ──────────────────────────────────────────────────────────────────
//  User display
// ──────────────────────────────────────────────────────────────────

function initUserDisplay() {
  setText("usernameDisplay", getFullName());
  const pill = $("rolePill");
  pill.textContent = getRole();
  if (getRole() === "admin") pill.classList.add("admin");
}

// ──────────────────────────────────────────────────────────────────
//  Clock
// ──────────────────────────────────────────────────────────────────

function tickClock() {
  const now = new Date();
  setText("clock",
    `${String(now.getUTCHours()).padStart(2,"0")}:` +
    `${String(now.getUTCMinutes()).padStart(2,"0")}:` +
    `${String(now.getUTCSeconds()).padStart(2,"0")} UTC`);
}
setInterval(tickClock, 1000);
tickClock();

// ──────────────────────────────────────────────────────────────────
//  Charts
// ──────────────────────────────────────────────────────────────────

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 200 },
  plugins: { legend: { display: false }, tooltip: { mode: "index", intersect: false } },
  scales: {
    x: {
      type: "category",
      ticks: { color: "#3d5060", font: { family: "Share Tech Mono", size: 10 }, maxTicksLimit: 8, maxRotation: 0 },
      grid:  { color: "rgba(255,255,255,0.04)" },
    },
    y: {
      ticks: { color: "#3d5060", font: { family: "Share Tech Mono", size: 10 } },
      grid:  { color: "rgba(255,255,255,0.04)" },
    },
  },
};

function initCharts() {
  const sCtx = $("sensorChart").getContext("2d");
  sensorChart = new Chart(sCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Pressure",    data: [], borderColor: "#00e676", backgroundColor: "rgba(0,230,118,0.05)", borderWidth: 1.5, pointRadius: 0, tension: 0.4, yAxisID: "yPressure" },
        { label: "Flow Rate",   data: [], borderColor: "#40c4ff", backgroundColor: "rgba(64,196,255,0.05)", borderWidth: 1.5, pointRadius: 0, tension: 0.4, yAxisID: "yFlow" },
        { label: "Temperature", data: [], borderColor: "#ffab00", backgroundColor: "rgba(255,171,0,0.05)",  borderWidth: 1.5, pointRadius: 0, tension: 0.4, yAxisID: "yTemp" },
        { label: "Anomaly",     data: [], borderColor: "transparent", backgroundColor: "#ff3d3d", pointRadius: d => d.raw !== null ? 6 : 0, showLine: false, yAxisID: "yPressure" },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        x: CHART_DEFAULTS.scales.x,
        yPressure: { ...CHART_DEFAULTS.scales.y, position: "left",  title: { display: true, text: "Pressure (bar)", color: "#3d5060", font: { family: "Share Tech Mono", size: 9 } } },
        yFlow:     { ...CHART_DEFAULTS.scales.y, position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "Flow / Temp", color: "#3d5060", font: { family: "Share Tech Mono", size: 9 } } },
        yTemp:     { display: false, position: "right" },
      },
    },
  });

  const aCtx = $("scoreChart").getContext("2d");
  scoreChart = new Chart(aCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Anomaly Score", data: [], borderColor: "#40c4ff", backgroundColor: ctx => { const g = ctx.chart.ctx.createLinearGradient(0,0,0,260); g.addColorStop(0,"rgba(64,196,255,0.3)"); g.addColorStop(1,"rgba(64,196,255,0)"); return g; }, borderWidth: 1.5, pointRadius: 0, tension: 0.4, fill: true },
        { label: "Threshold",     data: [], borderColor: "rgba(255,61,61,0.5)", borderWidth: 1, borderDash: [4,4], pointRadius: 0, fill: false },
      ],
    },
    options: { ...CHART_DEFAULTS, scales: { x: CHART_DEFAULTS.scales.x, y: { ...CHART_DEFAULTS.scales.y, min: 0 } } },
  });
}

// ──────────────────────────────────────────────────────────────────
//  Push data to charts
// ──────────────────────────────────────────────────────────────────

function pushToCharts(data) {
  const label = data.timestamp
    ? new Date(data.timestamp).toLocaleTimeString("en-GB", { hour12: false })
    : new Date().toLocaleTimeString("en-GB", { hour12: false });

  const isAnomaly = data.is_anomaly || data.status === "Leak Detected";

  function trim(arr, val) { arr.push(val); if (arr.length > MAX_POINTS) arr.shift(); }

  const sd = sensorChart.data;
  trim(sd.labels,           label);
  trim(sd.datasets[0].data, data.pressure);
  trim(sd.datasets[1].data, data.flow_rate);
  trim(sd.datasets[2].data, data.temperature);
  trim(sd.datasets[3].data, isAnomaly ? data.pressure : null);

  const ad = scoreChart.data;
  trim(ad.labels,           label);
  trim(ad.datasets[0].data, data.anomaly_score || 0);
  const thr = (data.score_ratio > 0 && data.anomaly_score > 0)
    ? (data.anomaly_score / data.score_ratio) : null;
  if (thr !== null) {
    while (ad.datasets[1].data.length < ad.datasets[0].data.length - 1) ad.datasets[1].data.push(thr);
    trim(ad.datasets[1].data, thr);
  } else {
    ad.datasets[1].data.push(null);
    if (ad.datasets[1].data.length > MAX_POINTS) ad.datasets[1].data.shift();
  }

  sensorChart.update("none");
  scoreChart.update("none");
}

// ──────────────────────────────────────────────────────────────────
//  Status UI
// ──────────────────────────────────────────────────────────────────

function updateStatus(data) {
  const status = data.status || "Collecting";
  const score  = data.anomaly_score || 0;
  const ratio  = data.score_ratio   || 0;

  let hex, bodyClass;
  if (status === "Leak Detected") { hex = "#ff3d3d"; bodyClass = "state-leak"; }
  else if (status === "Warning")  { hex = "#ffab00"; bodyClass = "state-warning"; }
  else                            { hex = "#00e676"; bodyClass = ""; }

  document.body.className = bodyClass;
  const heroEl = document.querySelector(".hero");
  heroEl.style.setProperty("--ring-color",   hex);
  heroEl.style.setProperty("--status-color", hex);

  $("statusIcon").textContent  = data.icon || "🟢";
  $("heroStatus").textContent  = status === "Collecting" ? "Initialising…" : status;
  $("heroSub").textContent     = {
    "Normal":        "All sensors within operating range",
    "Warning":       "Unusual readings — monitoring closely",
    "Leak Detected": "ALERT: Anomalous sensor pattern detected!",
    "Collecting":    `Buffering readings (${data.buffer_fill || 0}/${SEQ_LEN})`,
  }[status] || "";

  const barPct = Math.min(ratio * 66.6, 100);
  const fillEl = $("scoreBarFill");
  fillEl.style.width = `${barPct}%`;
  fillEl.style.background = ratio >= 1.0
    ? "linear-gradient(90deg,#ff3d3d,#ff6b6b)"
    : ratio >= 0.6
      ? "linear-gradient(90deg,#ffab00,#ffd54f)"
      : "linear-gradient(90deg,#00e676,#40c4ff)";
  $("scoreValue").textContent = score.toFixed(6);

  const p = data.pressure, f = data.flow_rate, t = data.temperature;
  if (p !== undefined) {
    $("metricPressure").textContent = p.toFixed(1);
    $("barPressure").style.width    = `${Math.min((p/120)*100,100)}%`;
    $("barPressure").style.background = p < 60 ? "var(--red)" : "var(--green)";
    $("trendPressure").textContent  = p < 65 ? "↓" : p > 90 ? "↑" : "→";
    $("cardPressure").classList.toggle("active", p < 65);
  }
  if (f !== undefined) {
    $("metricFlow").textContent   = f.toFixed(0);
    $("barFlow").style.width      = `${Math.min((f/700)*100,100)}%`;
    $("barFlow").style.background = Math.abs(f-500) > 80 ? "var(--amber)" : "var(--blue)";
    $("trendFlow").textContent    = f < 420 ? "↓" : f > 580 ? "↑" : "→";
  }
  if (t !== undefined) {
    $("metricTemp").textContent   = t.toFixed(1);
    $("barTemp").style.width      = `${Math.min(((t-30)/40)*100,100)}%`;
    $("barTemp").style.background = t > 50 ? "var(--red)" : "var(--amber)";
    $("trendTemp").textContent    = t > 50 ? "↑" : "→";
  }

  totalReadings++;
  if (status === "Leak Detected") totalLeaks++;
  $("statTotal").textContent  = totalReadings;
  $("statLeaks").textContent  = totalLeaks;
  $("statUptime").textContent = totalReadings
    ? (100 - (totalLeaks/totalReadings)*100).toFixed(1)+"%"
    : "100%";
}

// ──────────────────────────────────────────────────────────────────
//  History log
// ──────────────────────────────────────────────────────────────────

let alertCount   = 0;
let lastAlertTime = 0;

function addHistoryEvent(data) {
  const list  = $("historyList");
  const empty = list.querySelector(".history-empty");
  if (empty) empty.remove();

  const cls  = data.status === "Leak Detected" ? "leak" : "warning";
  const icon = data.status === "Leak Detected" ? "🔴" : "🟡";
  const time = new Date(data.timestamp || Date.now()).toLocaleTimeString("en-GB");
  const evId = data.id;

  const item = document.createElement("div");
  item.className = `history-item ${cls}`;
  item.dataset.id = evId || "";
  item.innerHTML = `
    <span class="history-item-icon">${icon}</span>
    <div class="history-item-info">
      <div class="history-item-title">${data.status}</div>
      <div class="history-item-meta">${time} · P:${(data.pressure||0).toFixed(1)} bar · F:${(data.flow_rate||0).toFixed(0)} m³/h</div>
    </div>
    <span class="history-item-score">${(data.anomaly_score||0).toFixed(4)}</span>
    ${evId ? `<button class="btn-ack" data-id="${evId}" title="Acknowledge">✓</button>` : ""}
  `;
  list.prepend(item);
  while (list.children.length > 30) list.lastChild.remove();

  alertCount++;
  $("badgeCount").textContent = alertCount;

  if (data.status === "Leak Detected" && Date.now() - lastAlertTime > 10000) {
    lastAlertTime = Date.now();
    showLeakModal(data);
  }
}

// Acknowledge via API
document.addEventListener("click", async e => {
  if (!e.target.classList.contains("btn-ack")) return;
  const evId = e.target.dataset.id;
  if (!evId) return;
  const res = await apiFetch(`${API_BASE}/events/${evId}/acknowledge`, {
    method: "POST",
    body:   JSON.stringify({ notes: "Acknowledged from dashboard" }),
  });
  if (res && res.ok) {
    const row = document.querySelector(`.history-item[data-id="${evId}"]`);
    if (row) { row.style.opacity = "0.5"; e.target.remove(); }
    showToast("✅ Event acknowledged");
  }
});

function showLeakModal(data) {
  $("alertBody").innerHTML = `
    Anomalous sensor readings detected at pipeline monitoring point.<br><br>
    <strong style="color:var(--red)">Score: ${(data.anomaly_score||0).toFixed(5)}</strong><br>
    Pressure: <strong>${(data.pressure||0).toFixed(1)} bar</strong> ·
    Flow: <strong>${(data.flow_rate||0).toFixed(0)} m³/h</strong> ·
    Temp: <strong>${(data.temperature||0).toFixed(1)}°C</strong><br><br>
    Dispatch field crew for inspection immediately.
  `;
  $("alertModal").classList.remove("hidden");
}

$("btnDismiss").addEventListener("click", () => $("alertModal").classList.add("hidden"));

// ──────────────────────────────────────────────────────────────────
//  Toast
// ──────────────────────────────────────────────────────────────────

function showToast(msg, duration = 3000) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.remove("hidden", "hide");
  t.classList.add("show");
  setTimeout(() => {
    t.classList.replace("show", "hide");
    setTimeout(() => t.classList.add("hidden"), 300);
  }, duration);
}

// ──────────────────────────────────────────────────────────────────
//  SSE stream — token passed as query param
// ──────────────────────────────────────────────────────────────────

function connectStream() {
  if (sseSource) { sseSource.close(); sseSource = null; }

  const token = getToken();
  if (!token) { logout(); return; }

  const url = `${API_BASE}/stream?anomaly_prob=0.12&token=${encodeURIComponent(token)}`;

  try {
    sseSource = new EventSource(url);

    sseSource.onopen = () => {
      setConnStatus(true);
      showToast("✅ Live stream connected");
    };

    sseSource.onmessage = event => {
      if (isPaused) return;
      try {
        const data = JSON.parse(event.data);
        updateStatus(data);
        pushToCharts(data);
        if (data.is_anomaly || data.status === "Warning") addHistoryEvent(data);
      } catch (e) { console.error("SSE parse:", e); }
    };

    sseSource.onerror = () => {
      setConnStatus(false);
      if (sseSource) { sseSource.close(); sseSource = null; }
      setTimeout(connectStream, 5000);
    };
  } catch (e) {
    setConnStatus(false);
    setTimeout(connectStream, 5000);
  }
}

function setConnStatus(live) {
  $("connDot").className    = "conn-dot" + (live ? " live" : " error");
  $("connText").textContent = live ? "Live Stream" : "Reconnecting…";
}

// ──────────────────────────────────────────────────────────────────
//  Pause / Resume
// ──────────────────────────────────────────────────────────────────

$("btnPause").addEventListener("click", () => {
  isPaused = !isPaused;
  $("btnPause").textContent = isPaused ? "▶" : "⏸";
  showToast(isPaused ? "⏸ Stream paused" : "▶ Stream resumed");
});

// ──────────────────────────────────────────────────────────────────
//  Manual Predict
// ──────────────────────────────────────────────────────────────────

$("btnManualPredict").addEventListener("click", async () => {
  const pressure    = parseFloat($("inPressure").value);
  const flow_rate   = parseFloat($("inFlow").value);
  const temperature = parseFloat($("inTemp").value);

  if (isNaN(pressure) || isNaN(flow_rate) || isNaN(temperature)) {
    showToast("⚠️ Please fill all three sensor fields");
    return;
  }

  const reading = { pressure, flow_rate, temperature };
  readingBuffer.push(reading);
  if (readingBuffer.length > SEQ_LEN) readingBuffer.shift();

  let data;
  if (readingBuffer.length < SEQ_LEN) {
    data = { status: "Collecting", color: "gray", icon: "⏳", is_anomaly: false, anomaly_score: 0, buffer_fill: readingBuffer.length, buffer_needed: SEQ_LEN, ...reading };
  } else {
    const res = await apiFetch(`${API_BASE}/predict`, {
      method: "POST",
      body:   JSON.stringify({ readings: readingBuffer }),
    });
    if (!res) return;
    data = { ...(await res.json()), ...reading };
  }

  const resEl = $("manualResult");
  resEl.classList.remove("hidden");
  $("manualIcon").textContent  = data.icon || "⏳";
  $("manualText").textContent  = data.status || "Collecting";
  $("manualScore").textContent = `Score: ${(data.anomaly_score||0).toFixed(6)}`;
  resEl.style.borderColor = data.color === "green" ? "var(--green)"
    : data.color === "red" ? "var(--red)" : data.color === "yellow" ? "var(--amber)" : "var(--border-hi)";

  if (readingBuffer.length >= SEQ_LEN) {
    updateStatus(data);
    pushToCharts({ ...reading, timestamp: new Date().toISOString(), ...data });
    if (data.is_anomaly || data.status === "Warning") addHistoryEvent({ ...data, ...reading, timestamp: new Date().toISOString() });
  }

  const need = SEQ_LEN - readingBuffer.length;
  if (need > 0) showToast(`⏳ ${need} more reading${need > 1 ? "s" : ""} needed to classify`);
});

// ──────────────────────────────────────────────────────────────────
//  Simulate batch
// ──────────────────────────────────────────────────────────────────

$("btnSimulate").addEventListener("click", async () => {
  const withAnomaly = $("chkAnomaly").checked;
  showToast("🔄 Fetching simulated pipeline batch…");

  const res = await apiFetch(`${API_BASE}/simulate/sequence?n=80&with_anomaly=${withAnomaly}`);
  if (!res) return;
  const payload = await res.json();

  clearCharts();
  let i = 0;
  const interval = setInterval(() => {
    if (i >= payload.data.length) { clearInterval(interval); showToast(`✅ Loaded ${payload.data.length} readings`); return; }
    const row  = payload.data[i];
    const data = {
      ...row,
      status:        row.is_anomaly ? "Leak Detected" : "Normal",
      color:         row.is_anomaly ? "red"           : "green",
      icon:          row.is_anomaly ? "🔴"            : "🟢",
      anomaly_score: row.is_anomaly ? 0.012 + Math.random()*0.02 : Math.random()*0.004,
      score_ratio:   row.is_anomaly ? 1.1   + Math.random()*0.5  : 0.2 + Math.random()*0.3,
      is_anomaly:    Boolean(row.is_anomaly),
    };
    updateStatus(data);
    pushToCharts(data);
    if (row.is_anomaly) addHistoryEvent({ ...data, timestamp: row.timestamp });
    i++;
  }, 40);
});

// ──────────────────────────────────────────────────────────────────
//  Clear charts
// ──────────────────────────────────────────────────────────────────

function clearCharts() {
  [sensorChart, scoreChart].forEach(chart => {
    chart.data.labels = [];
    chart.data.datasets.forEach(ds => ds.data = []);
    chart.update("none");
  });
}
$("btnClear").addEventListener("click", () => { clearCharts(); showToast("🗑 Charts cleared"); });

// ──────────────────────────────────────────────────────────────────
//  Logout
// ──────────────────────────────────────────────────────────────────

$("btnLogout").addEventListener("click", () => {
  if (confirm("Sign out of PipeGuard AI?")) logout();
});

// ──────────────────────────────────────────────────────────────────
//  Load DB history on startup
// ──────────────────────────────────────────────────────────────────

async function loadHistory() {
  const res = await apiFetch(`${API_BASE}/events?limit=20`);
  if (!res || !res.ok) return;
  const { events } = await res.json();
  if (!events || events.length === 0) return;
  events.forEach(ev => addHistoryEvent(ev));
}

// ──────────────────────────────────────────────────────────────────
//  Health check
// ──────────────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    if (!data.model_loaded) {
      showToast("⚠️ Model not loaded — run train.py first!", 6000);
      setConnStatus(false);
      return false;
    }
    return true;
  } catch {
    setConnStatus(false);
    showToast("❌ Backend offline — start FastAPI server first!", 6000);
    return false;
  }
}

// ──────────────────────────────────────────────────────────────────
//  Initialise
// ──────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
  initUserDisplay();
  initCharts();
  await loadHistory();
  const ok = await checkHealth();
  if (ok) {
    connectStream();
  } else {
    const retryId = setInterval(async () => {
      if (await checkHealth()) { clearInterval(retryId); connectStream(); }
    }, 8000);
  }
});
