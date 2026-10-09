"use strict";

// Same-origin requests work on localhost, Render, Railway and custom domains.
// Predictions always come from the backend. This file never synthesizes scores.
const $ = id => document.getElementById(id);
const names = ["mass_inventory", "dynamic_residual", "cnn_lstm", "hybrid"];
const labels = { mass_inventory: "Mass balance", dynamic_residual: "Dynamic predictor", cnn_lstm: "CNN–LSTM", hybrid: "Residual hybrid" };
const colors = { mass_inventory: "#368b7e", dynamic_residual: "#6280bc", cnn_lstm: "#b18b40", hybrid: "#986c95" };
const descriptions = { mass_inventory: "Inlet − outlet flow, corrected for gas storage.", dynamic_residual: "Forecast errors from lagged sensors and commands.", cnn_lstm: "Reconstruction error across the eight inputs.", hybrid: "Neural innovations and the dynamic check." };
let config, evidence, run = null, cursor = 719, timer = null, busy = false;
let coverage = "broad";

function escapeHTML(value) { return String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function clock(seconds) { const s = Math.max(0, Math.floor(seconds)); return `${String(Math.floor(s / 60)).padStart(2,"0")}:${String(s % 60).padStart(2,"0")}`; }
function valueText(value, digits = 2) { return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits); }
function message(text, error = false) { $("message").textContent = text; $("message").classList.toggle("error", error); $("message").hidden = !text; }
function stopReplay() { if (timer) clearInterval(timer); timer = null; $("play-button").textContent = "▶"; $("play-button").setAttribute("aria-label", "Replay simulated time"); }

async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      const text = await response.text();
      try { const data = JSON.parse(text); detail = typeof data.detail === "string" ? data.detail : data.detail?.map(v => v.msg).join("; ") || detail; } catch { detail = text || detail; }
      throw new Error(detail);
    }
    return await response.json();
  } catch (error) { if (error.name === "AbortError") throw new Error("The server took too long to respond. A sleeping deployment may need a moment; try again."); throw error; }
  finally { clearTimeout(timeout); }
}

function setBusy(value) {
  busy = value;
  $("run-button").disabled = value;
  $("csv-upload").disabled = value;
  $("run-button").firstElementChild.textContent = value ? "Computing all four methods…" : "Run experiment";
  $("scenario-form").setAttribute("aria-busy", String(value));
}

function createCards() {
  $("model-cards").innerHTML = names.map((name, i) => `<article class="card model-card" style="--model-color:${colors[name]}" id="card-${name}"><div class="model-title"><h3>${labels[name]}</h3><span class="model-number">0${i+1}</span></div><p class="description">${descriptions[name]}</p><div class="model-status warm" id="status-${name}"><span class="status-dot"></span><span>Waiting for readings</span></div><div class="score-value" id="value-${name}">—<small>× threshold</small></div><div class="score-note" id="score-${name}">Frozen healthy-calibration threshold</div><div class="model-summary" id="summary-${name}">No experiment yet</div></article>`).join("");
  $("score-legend").innerHTML = names.map(name => `<span><i class="legend-dot" style="background:${colors[name]}"></i>${labels[name]}</span>`).join("") + `<span class="legend-end">Threshold ratio = 1</span>`;
}

function setupScenario() {
  const scenario = $("scenario").value;
  const leak = ["leak", "gradual_leak"].includes(scenario);
  $("leak-controls").hidden = !leak;
  $("scenario-description").textContent = config?.scenarios[scenario]?.description || "";
}

function applyRun(data) {
  stopReplay();
  run = data;
  cursor = run.time_s.length - 1;
  $("timeline").max = String(cursor);
  $("timeline").value = String(cursor);
  $("timeline").disabled = false;
  $("play-button").disabled = false;
  $("show-truth").disabled = run.ground_truth == null;
  $("run-kind").textContent = run.source === "simulation" ? "SIMULATED READINGS" : "UPLOADED READINGS";
  $("duration").textContent = clock(run.time_s.at(-1) - run.time_s[0]);
  const scenarioLabel = config.scenarios[run.scenario]?.label || "Uploaded sensor readings";
  $("run-meta").textContent = run.source === "simulation" ? `${scenarioLabel} · seed ${run.seed} · ${run.compute_seconds.toFixed(3)} s compute` : `${run.time_s.length} uploaded readings · simulation reference`;
  for (const format of ["csv", "json"]) {
    const link = $(`export-${format}`);
    link.href = `/api/runs/${encodeURIComponent(run.id)}/export.${format}`;
    link.classList.remove("disabled"); link.setAttribute("aria-disabled", "false");
  }
  renderRun();
}

function renderRun() {
  if (!run) return;
  $("clock").textContent = clock(run.time_s[cursor] - run.time_s[0]);
  const row = run.readings[cursor];
  const truthVisible = $("show-truth").checked && run.ground_truth;
  const leakNow = truthVisible && run.ground_truth.leak_present[cursor];
  for (let i = 0; i < 3; i++) {
    $(`pressure-${i}`).textContent = `${valueText(row[i] == null ? null : row[i] / 1e6, 3)} MPa`;
    $(`volume-${i}`).classList.toggle("injected", Boolean(leakNow && run.ground_truth.injected_location === i));
  }
  $("flow-in").textContent = `${valueText(row[3], 3)} kg/s`;
  $("flow-out").textContent = `${valueText(row[4], 3)} kg/s`;
  $("truth-callout").classList.toggle("leak", Boolean(leakNow));
  if (!run.ground_truth) $("truth-callout").textContent = "No ground truth supplied. An anomaly does not establish a leak, its cause or its location. Check the units and reference suitability.";
  else if (!truthVisible) $("truth-callout").textContent = "Injected ground truth hidden. Model scores and alert states are unchanged. Location inference: unestablished.";
  else if (run.ground_truth.onset_s == null) $("truth-callout").textContent = `${config.scenarios[run.scenario].description} No leak was injected. Location inference: unestablished.`;
  else if (leakNow) $("truth-callout").textContent = `Known injection: volume ${"ABC"[run.ground_truth.injected_location]}, ${Math.round(run.leak_fraction * 100)}% nominal size, onset ${clock(run.ground_truth.onset_s)}. This is scenario truth, not an inferred location.`;
  else $("truth-callout").textContent = `Known injection begins at ${clock(run.ground_truth.onset_s)}. Before that point, no leak is present. Location inference: unestablished.`;
  for (const name of names) {
    const model = run.predictions.models[name], score = model.scores[cursor];
    const quality = run.predictions.quality[cursor];
    const status = $(`status-${name}`);
    const pending = score != null && score > model.threshold && !model.active[cursor];
    const state = quality === "data_quality_alert" ? "Sensor data missing" : score == null ? "Warming up" : model.active[cursor] ? "Anomaly alert" : pending ? "Confirming exceedance" : "Within reference";
    status.className = `model-status ${quality === "data_quality_alert" ? "quality" : score == null ? "warm" : model.active[cursor] ? "alert" : pending ? "quality" : ""}`;
    status.lastElementChild.textContent = state;
    $(`value-${name}`).innerHTML = `${valueText(score == null ? null : score / model.threshold, 2)}<small>× threshold</small>`;
    $(`score-${name}`).textContent = score == null ? `${run.predictions.warmup_readings} consecutive valid readings required` : `Score ${score.toPrecision(4)} / threshold ${model.threshold.toPrecision(4)}`;
    const events = model.event_times_s.filter(t => t <= run.time_s[cursor]);
    const after = run.ground_truth?.onset_s == null ? [] : events.filter(t => t >= run.ground_truth.onset_s);
    let summary = `${events.length} new alarm event${events.length === 1 ? "" : "s"} so far`;
    if (after.length) summary = `New post-onset alarm · delay ${Math.round(after[0] - run.ground_truth.onset_s)} s`;
    else if (run.ground_truth?.onset_s != null && run.time_s[cursor] >= run.ground_truth.onset_s) summary = "No new post-onset alarm so far";
    $(`summary-${name}`).textContent = summary;
  }
  renderCharts();
}

// Exact sensor/score traces, rendered locally without external chart libraries.
function chartSVG(series, yMin, yMax, {log = false, threshold = null, suffix = ""} = {}) {
  const width = 900, height = 205, left = 56, right = 18, top = 13, bottom = 30;
  const span = run.time_s.at(-1) - run.time_s[0] || 1;
  const x = i => left + (run.time_s[i] - run.time_s[0]) / span * (width-left-right);
  const transform = value => log ? Math.log10(Math.max(value, .01)) : value;
  const a = transform(yMin), b = transform(yMax);
  const y = value => top + (b - transform(value)) / (b-a || 1) * (height-top-bottom);
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${log ? "Model scores relative to threshold" : "Measured pressures"} across simulated time"><g font-family="monospace" font-size="10" fill="#93a391">`;
  const tickValues = log ? [.01,.1,1,10,100,1000,10000,100000].filter(v => v >= yMin && v <= yMax) : [0,1,2,3].map(i => yMin + (yMax-yMin)*i/3);
  for (const value of tickValues) svg += `<path d="M${left} ${y(value)} H${width-right}" stroke="#e8eee4" stroke-width="1"/><text x="${left-10}" y="${y(value)+3}" text-anchor="end">${log ? value : value.toFixed(2)}${suffix}</text>`;
  for (let t = 0; t <= 4; t++) { const px = left+t/4*(width-left-right); svg += `<text x="${px}" y="${height-7}" text-anchor="middle">${Math.round(span*t/4/60)} min</text>`; }
  svg += "</g>";
  if (threshold != null) svg += `<path d="M${left} ${y(threshold)} H${width-right}" stroke="#c1a05a" stroke-dasharray="5 5" stroke-width="1.3"/>`;
  if ($("show-truth").checked && run.ground_truth?.onset_s != null && run.ground_truth.onset_s <= run.time_s[cursor]) { const px = left+(run.ground_truth.onset_s-run.time_s[0])/span*(width-left-right); svg += `<path d="M${px} ${top} V${height-bottom}" stroke="#c1866a" stroke-dasharray="4 4" stroke-width="1"/>`; }
  for (const item of series) {
    let path = "", open = false;
    // Preserve all observations. Explicit missing points break the path.
    for (let i = 0; i <= cursor; i++) {
      const value = item.values[i];
      if (value == null || !Number.isFinite(value)) { open = false; continue; }
      path += `${open ? "L" : "M"}${x(i).toFixed(2)} ${y(value).toFixed(2)} `; open = true;
    }
    svg += `<path d="${path}" fill="none" stroke="${item.color}" stroke-width="1.5" stroke-linejoin="round"/>`;
  }
  svg += `<path d="M${x(cursor)} ${top} V${height-bottom}" stroke="#a9bcab" stroke-width="1" opacity=".45"/></svg>`;
  return svg;
}

function renderCharts() {
  const pressure = [0,1,2].map((column, i) => ({values:run.readings.map(row => row[column] == null ? null : row[column]/1e6), color:[colors.mass_inventory,colors.dynamic_residual,colors.cnn_lstm][i]}));
  const finite = pressure.flatMap(s => s.values).filter(v => v != null);
  if (finite.length) {
    const low = Math.min(...finite), high = Math.max(...finite), pad = Math.max(.008,(high-low)*.12);
    $("pressure-chart").innerHTML = chartSVG(pressure, low-pad, high+pad);
  } else $("pressure-chart").textContent = "Pressure data unavailable. Scoring is suspended while sensors are missing.";
  const scores = names.map(name => ({values:run.predictions.models[name].scores.map(v => v == null ? null : v/run.predictions.models[name].threshold),color:colors[name]}));
  const maxRatio = Math.max(10,...scores.flatMap(s => s.values).filter(v => v != null));
  $("score-chart").innerHTML = chartSVG(scores,.01,Math.pow(10,Math.ceil(Math.log10(maxRatio))),{log:true,threshold:1});
}

function play() {
  if (!run) return;
  if (timer) { stopReplay(); return; }
  if (cursor === run.time_s.length-1) cursor = 0;
  $("play-button").textContent = "Ⅱ"; $("play-button").setAttribute("aria-label","Pause replay");
  renderRun();
  timer = setInterval(() => {
    cursor = Math.min(run.time_s.length-1, cursor + Math.max(1,Math.round(Number($("speed").value)*.15/5)));
    $("timeline").value = String(cursor); renderRun();
    if (cursor === run.time_s.length-1) stopReplay();
  }, 150);
}

function renderEvidence() {
  const rows = evidence.simulation.summaries.filter(row => row.calibration_coverage === coverage);
  $("result-table").innerHTML = names.map(name => {
    const row = rows.find(v => v.model === name || (name === "mass_inventory" && v.model.startsWith("mass_inventory")));
    if (!row) return "";
    const ci = row.event_recall_95pct_ci.map(v => (v*100).toFixed(1)).join("–");
    return `<tr><td><div class="method-name"><i class="legend-dot" style="background:${colors[name]}"></i>${labels[name]}</div></td><td><strong>${row.detected_events} / ${row.leak_events}</strong></td><td><strong>${(row.event_recall*100).toFixed(1)}%</strong><span>${ci}%</span></td><td><strong>${row.false_alarms_per_hour.toFixed(3)}</strong><span>${row.false_alarm_events} events / ${row.healthy_hours.toFixed(2)} h</span></td><td><strong>${row.median_detected_delay_s == null ? "No detections" : `${row.median_detected_delay_s.toFixed(1)} s`}</strong></td><td><strong>${row.sensor_bias_alerted_runs} / 6</strong></td></tr>`;
  }).join("");
  document.querySelectorAll("[data-coverage]").forEach(button => { const active = button.dataset.coverage === coverage; button.classList.toggle("selected",active); button.setAttribute("aria-pressed",String(active)); });
  $("acoustic-results").innerHTML = evidence.acoustic.summaries.map(row => `<div class="acoustic-method" style="--model-color:${colors[row.model]}"><h3>${labels[row.model]}</h3><strong>${row.true_positive} / ${row.leak_test_clips}</strong><span>leak clips detected</span><small>Recall CI: ${row.recall_95pct_ci.map(v => (v*100).toFixed(1)).join("–")}%</small><p>${row.false_positive} / ${row.healthy_test_clips} healthy clips flagged</p></div>`).join("");
  $("references").innerHTML = evidence.references.map(ref => `<li><span>${escapeHTML(ref.id)}</span><a href="${escapeHTML(ref.url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(ref.title)} ↗</a></li>`).join("");
}

function selectTab(name, focus = false) {
  document.querySelectorAll("[data-tab]").forEach(button => {
    const active = button.dataset.tab === name;
    button.classList.toggle("selected", active); button.setAttribute("aria-selected", String(active)); button.tabIndex = active ? 0 : -1;
    $(`panel-${button.dataset.tab}`).hidden = !active;
    if (active && focus) button.focus();
  });
}

function parseCSV(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const fields = line => line.split(",").map(v => v.trim().replace(/^"(.*)"$/, "$1"));
  const header = fields(lines.shift());
  const required = ["timestamp_s", ...config.models.schema];
  const indices = required.map(name => header.indexOf(name));
  if (indices.some(i => i < 0)) throw new Error(`CSV requires these columns: ${required.join(", ")}`);
  if (new Set(header).size !== header.length) throw new Error("CSV column names must be unique.");
  const number = (token, allowMissing) => {
    if (!token || /^(na|nan|null)$/i.test(token)) { if (allowMissing) return null; throw new Error("Every row needs a numeric timestamp_s."); }
    const value = Number(token); if (!Number.isFinite(value)) throw new Error(`Invalid numeric reading: ${token.slice(0,30)}`); return value;
  };
  const readings = lines.filter(line => line.trim()).map(line => {
    const cols = fields(line); if (cols.length !== header.length) throw new Error("Every CSV row must have the same number of columns as the header.");
    return {timestamp_s:number(cols[indices[0]],false),values:indices.slice(1).map(i => number(cols[i],true))};
  });
  if (readings.length < 32 || readings.length > 1200) throw new Error("Upload between 32 and 1,200 readings. Use the template for the required SI units.");
  return {readings};
}

$("scenario-form").addEventListener("submit", async event => {
  event.preventDefault(); if (busy || !config || !$("scenario-form").reportValidity()) return;
  stopReplay(); setBusy(true); message("Running the simulator and all four frozen models. No ground-truth labels enter inference.");
  const payload = {scenario:$("scenario").value,leak_fraction:Number($("leak-fraction").value),location:Number($("location").value),regime:$("regime").value,seed:Number($("seed").value)};
  try { const data = await request("/api/runs", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}); applyRun(data); message(`Experiment complete. ${data.time_s.length} readings scored in ${data.compute_seconds.toFixed(3)} s. Replay the trace or export the complete record.`); }
  catch (error) { message(error.message,true); }
  finally { setBusy(false); }
});
$("csv-upload").addEventListener("change", async event => {
  const file = event.target.files[0]; if (!file || busy || !config) return;
  stopReplay(); setBusy(true);
  try {
    if (file.size > 400000) throw new Error("CSV exceeds 400 kB. Upload a shorter segment.");
    const payload = parseCSV(await file.text());
    message("Scoring the uploaded sensor readings against the simulation reference…");
    applyRun(await request("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}));
    message("Upload analyzed. Check SI units and reference suitability: these models have no field validation. Additional CSV columns were ignored.");
  } catch (error) { message(error.message,true); }
  finally { setBusy(false); event.target.value = ""; }
});
$("scenario").addEventListener("change",setupScenario);
$("play-button").addEventListener("click",play);
$("show-truth").addEventListener("change",renderRun);
$("timeline").addEventListener("input",event => { stopReplay(); cursor=Number(event.target.value); renderRun(); });
document.querySelectorAll("[data-tab]").forEach(button => {
  button.addEventListener("click",()=>selectTab(button.dataset.tab));
  button.addEventListener("keydown",event => {
    if (!["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) return;
    event.preventDefault(); const tabs=["lab","results","research"],i=tabs.indexOf(button.dataset.tab);
    const next=event.key === "Home" ? 0 : event.key === "End" ? 2 : (i+(event.key === "ArrowRight" ? 1 : 2))%3;
    selectTab(tabs[next],true);
  });
});
document.querySelectorAll("[data-coverage]").forEach(button => button.addEventListener("click",()=>{coverage=button.dataset.coverage;if(evidence)renderEvidence();}));
window.addEventListener("pagehide",stopReplay);
createCards(); selectTab("lab");
async function initialize() {
  try {
    const [settings, health, record, example] = await Promise.all([request("/api/config"),request("/health"),request("/api/evidence"),request("/api/example")]);
    config=settings;evidence=record;setupScenario();renderEvidence();applyRun(example);
    $("server-status").textContent=`${health.models_loaded} methods ready`;
    message("Loaded a reproducible example from the actual model runtime. Change the controls and run your own experiment.");
  } catch (error) { $("server-status").textContent="Engine unavailable";message(`Could not load the research lab. ${error.message}`,true); }
}
initialize();
