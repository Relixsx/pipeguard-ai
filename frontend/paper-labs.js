"use strict";
// Paper-lab estimates, screens and metrics are computed on the server.
let paperBusy = false;
const metric = (label, value, note = "") => `<div><span>${escapeHTML(label)}</span><strong>${escapeHTML(value)}</strong><small>${escapeHTML(note)}</small></div>`;
const exportLab = data => data.id ? `<a class="text-link" href="/api/runs/${encodeURIComponent(data.id)}/export.json">Export this experiment JSON ↓</a>` : "";

function labPlot(lines, indices, {label, intervals = null, flags = null, reference = null} = {}) {
  const width=880, height=220, l=50, r=20, top=18, bottom=36;
  const all=lines.flatMap(line=>line.values).concat(intervals ? intervals.flat() : []).filter(Number.isFinite);
  if (reference != null) all.push(reference);
  let lo=Math.min(...all),hi=Math.max(...all),pad=Math.max((hi-lo)*.15,.015);lo-=pad;hi+=pad;
  const xmin=Math.min(...indices),xmax=Math.max(...indices);
  const x=t=>l+(t-xmin)/(xmax-xmin||1)*(width-l-r),y=v=>top+(hi-v)/(hi-lo)*(height-top-bottom);
  let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHTML(label)}"><g fill="#728979" font-size="11" font-family="monospace">`;
  for(let j=0;j<=3;j++){const v=lo+(hi-lo)*j/3;svg+=`<path d="M${l} ${y(v)} H${width-r}" stroke="#e4ebe2"/><text x="${l-8}" y="${y(v)+4}" text-anchor="end">${v.toFixed(2)}</text>`;}
  for(let j=0;j<=4;j++){const t=xmin+(xmax-xmin)*j/4;svg+=`<text x="${x(t)}" y="${height-12}" text-anchor="middle">${Math.round(t)}</text>`;}
  svg+='</g>';
  if(reference!=null)svg+=`<path d="M${l} ${y(reference)} H${width-r}" stroke="#ad9155" stroke-dasharray="5 5"/>`;
  if(intervals)intervals.forEach((ci,i)=>{svg+=`<path d="M${x(indices[i])} ${y(ci[0])} V${y(ci[1])}" stroke="#86b3a4" stroke-width="3"/>`;});
  for(const line of lines)svg+=`<path d="${line.values.map((v,i)=>`${i?'L':'M'}${x(indices[i])} ${y(v)}`).join(' ')}" fill="none" stroke="${line.color}" stroke-width="2"/>`;
  if(flags)flags.forEach((flag,i)=>{if(flag)svg+=`<circle cx="${x(indices[i])}" cy="${y(lines[0].values[i])}" r="4" fill="#bb6550"/>`;});
  return svg+'</svg>';
}

function networkGraph(inputs) {
  const positions={w1:[100,110],w2:[300,110],w3:[200,35],w4:[400,185]};
  const edges=[['w1','w2'],['w1','w3'],['w3','w2'],['w2','w1'],['w2','w4'],['w4','w2']];
  let svg='<svg viewBox="0 0 530 230" role="img" aria-label="Target w1 to w2 with a parallel path through w3 and output loop through w4"><defs><marker id="network-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10" fill="#91a594"/></marker></defs>';
  for(const [a,b] of edges){const [ax,ay]=positions[a],[bx,by]=positions[b];const dx=bx-ax,dy=by-ay,d=Math.hypot(dx,dy),side=a==='w2'?-12:0;svg+=`<path d="M${ax+dx/d*27} ${ay+dy/d*27+side} L${bx-dx/d*29} ${by-dy/d*29+side}" stroke="${a==='w1'&&b==='w2'?'#ba7847':'#b2c1b0'}" stroke-width="2" marker-end="url(#network-arrow)"/>`;}
  for(const [name,[x,y]] of Object.entries(positions))svg+=`<circle cx="${x}" cy="${y}" r="25" fill="${name==='w2'?'#142f34':inputs.includes(name)?'#e2ede3':'#fafbf6'}" stroke="#91a594"/><text x="${x}" y="${y+5}" text-anchor="middle" fill="${name==='w2'?'#fff':'#3e6050'}" font-size="14">${name}</text>`;
  return svg+'<text x="178" y="132" fill="#9c743d" font-size="11">target connection</text><text x="18" y="205" fill="#758c77" font-size="11">r1, r3, r4: independent external references</text></svg>';
}

function renderNetwork(data) {
  const full=data.full_two_stage, omitted=data.omitted_two_stage;
  const supported=data.rolling.filter(r=>r.supported).length, flags=data.rolling.filter(r=>r.target_change_flag).length;
  $('network-output').innerHTML=`<div class="network-layout"><div class="network-graph">${networkGraph(data.graph.used_inputs)}</div><div><div class="method-metrics">${metric('Two-stage · full inputs',valueText(full.target_estimate,3),'Last 2,048 measurements')}${metric('Ordinary regression',valueText(full.naive_ols_estimate,3),'Same full measured inputs')}${metric('Two-stage · omit paths',valueText(omitted.target_estimate,3),'Graph does not preserve target')}</div><div class="guard-status ${supported?'':'failed'}">${supported ? `${supported}/${data.rolling.length} windows support attribution in the declared model` : 'Attribution withheld: graph or excitation requirements fail'}</div><p class="evidence-note">Used inputs: ${data.graph.used_inputs.join(', ')}. Target preserved: ${data.preservation.target_preserved?'yes':'no'}. ${data.preservation.unblocked_parallel_path?`Unblocked path: ${data.preservation.unblocked_parallel_path.join(' → ')}.`:''} ${data.preservation.unblocked_output_loop?`Unblocked loop: ${data.preservation.unblocked_output_loop.join(' → ')}.`:''} ${flags} approximate change flags.</p></div></div><div class="method-chart">${labPlot([{values:data.rolling.map(r=>r.estimate),color:'#368b7e'}],data.rolling.map(r=>r.endpoint_sample),{label:'Rolling target estimates with approximate 95 percent HAC intervals',intervals:data.rolling.map(r=>r.interval_95),flags:data.rolling.map(r=>r.target_change_flag),reference:data.evaluation.known_healthy_target})}</div><p class="evidence-note">Horizontal axis: sample index. Bars: approximate 95% HAC intervals; red dots: Bonferroni-guarded change flags. Dashed 0.65 line is known simulation truth, used only for evaluation. Finite-sample coverage and causal physical attribution outside this declared model are not guaranteed. Realized controller commands are not automatically valid external instruments.</p>${exportLab(data)}`;
}

function renderInformation(data) {
  const c=data.d6,s=data.structural;
  $('information-output').innerHTML=`<div class="method-metrics">${metric('Active independent signals',data.active_excitations.join(' + '))}${metric('Disconnecting set',c.disconnecting_set.join(', '),'Separates target u1 from other inputs')}${metric('Required / available paths',`${c.required_disjoint_paths} / ${c.available_disjoint_paths}`)}${metric('Target spectral minimum',valueText(data.minimum_target_spectral_eigenvalue,3),'Known model · 64 frequency points')}</div><div class="guard-status ${c.sufficient_condition?'':'failed'}">${c.sufficient_condition?'Target condition satisfied; the entire four-module row remains ambiguous':'Target condition fails at this excitation location'}</div><p class="method-explain">Input graph: u1 → u2 → u3, u3 → u2, u4 → u3. ${c.excluded_excitations.length?`Signal ${c.excluded_excitations.join(', ')} reaches the remaining inputs without passing u2 and is excluded from the sufficient target-information test.`:'Independent signals reach u1 and separator u2 through two vertex-disjoint paths.'} Target generic rank increment: ${s.full_row_generic_rank} − ${s.nuisance_generic_rank} = ${s.full_row_generic_rank-s.nuisance_generic_rank}; whole-row rank: ${s.full_row_generic_rank}/4.</p><p class="evidence-note">D4 Figure 1 also gives a unique target with an ambiguous full row. These graph results assume known topology, independently parameterized modules and admissible independent excitation. The finite frequency grid supplements the graph test; it is not a standalone proof for arbitrary transfer functions.</p>${exportLab(data)}`;
}

function rmsHeatmap(rows, title) {
  const channels=rows[0]?.length||0, width=860,left=55,cell=(width-left-20)/rows.length;
  const max=Math.max(1,...rows.flat()),low=0;
  let svg=`<svg viewBox="0 0 ${width} ${40+channels*35}" role="img" aria-label="${escapeHTML(title)} by channel across sample bins"><text x="${left}" y="15" font-size="11" fill="#738779">${escapeHTML(title)} · normalized RMS (0–${max.toFixed(1)})</text>`;
  for(let c=0;c<channels;c++){svg+=`<text x="8" y="${49+c*35}" font-size="11" fill="#617e69">Ch ${c+1}</text>`;rows.forEach((r,i)=>{const a=Math.min(1,(r[c]-low)/max);const color=`rgb(${Math.round(236-193*a)},${Math.round(240-114*a)},${Math.round(225-112*a)})`;svg+=`<rect x="${left+i*cell}" y="${28+c*35}" width="${cell-.8}" height="28" fill="${color}"><title>Channel ${c+1}, bin ${i+1}: RMS ${r[c].toFixed(3)}</title></rect>`;});}
  return svg+'</svg>';
}

function renderAcoustic(data) {
  const v=data.verification,s=data.separator;
  $('acoustic-output').innerHTML=`<div class="method-metrics">${metric('Raw cross-correlation',valueText(data.raw_maximum_absolute_correlation,3),'Maximum across channel pairs and lags')}${metric('Residual cross-correlation',valueText(v.maximum_absolute_correlation,3),'Separate healthy verification segment')}${metric('Shift-screen p-value',valueText(v.familywise_surrogate_p,2),'99 shifts · max over pairs and lags')}${metric('Inverse spectral radius',valueText(s.inverse_spectral_radius,3),'Below 1 means stable fitted inverse')}</div><div class="guard-status ${data.dependence_screen_passed?'':'failed'}">${data.dependence_screen_passed?'Stable inverse; held-out cross-correlation screen did not reject':'Source interpretation unsupported by the stability or dependence screen'}</div><div class="rms-heatmap">${rmsHeatmap(data.measured_rms,'Measured mixtures')}${rmsHeatmap(data.source_rms,'Estimated source residuals')}</div><p class="evidence-note">Each cell uses 128 subsequent samples. Heatmaps have separate displayed scales. FIR order ${s.order}, neighbor radius ${s.radius}, adjacent delay ${s.adjacent_delay_samples} samples; Eq.8 lag convention. Regularization uses the smallest nonzero design singular value. Passing a correlation screen does not prove independence or true source recovery.</p>${data.evaluation?`<details><summary>Inspect simulation truth used only for evaluation</summary><p class="method-explain">Known-source correlations by channel: ${data.evaluation.known_source_correlations.map(v=>valueText(v,3)).join(', ')}. Sources are ${data.evaluation.true_sources_dependent?'dependent':'independent'} and ${data.evaluation.true_sources_white?'white':'colored'}. ${data.evaluation.source_change_onset_sample!=null?`Source 3 amplitude changes at sample ${data.evaluation.source_change_onset_sample}.`:''} The dependent-but-uncorrelated case deliberately exposes a limitation of correlation verification.</p></details>`:''}${exportLab(data)}`;
}

function summaryTable(rows) {
  return `<div class="table-scroll"><table><thead><tr><th>Method</th><th>New leak detections</th><th>False alarms / healthy hour</th><th>Median detected delay</th></tr></thead><tbody>${rows.map(r=>`<tr><td><strong>${escapeHTML(labels[r.method]||'Global refiner')}</strong></td><td>${r.detected_events}/${r.leak_events}</td><td>${valueText(r.false_alarms_per_hour,3)}<span>${r.false_alarm_events} events / ${r.healthy_hours.toFixed(2)} h</span></td><td>${r.median_detected_delay_s==null?'No detections':`${r.median_detected_delay_s.toFixed(1)} s`}</td></tr>`).join('')}</tbody></table></div>`;
}

function renderPaperEvidence(data) {
  $('method-benchmarks').innerHTML=`<h3>Two-stage identification · 80 independent runs per condition</h3><div class="table-scroll"><table><thead><tr><th>Disturbance / excitation</th><th>Two-stage target RMSE</th><th>Ordinary regression RMSE</th><th>Approximate 95% interval coverage</th><th>Attribution supported</th></tr></thead><tbody>${data.benchmarks.network.map(r=>`<tr><td>${escapeHTML(r.scenario.replaceAll('_',' '))}</td><td>${r.two_stage_rmse.toFixed(4)}</td><td>${r.naive_ols_rmse.toFixed(4)}</td><td>${(r.hac_interval_coverage*100).toFixed(1)}%</td><td>${r.attribution_supported_runs}/${r.replications}</td></tr>`).join('')}</tbody></table></div><p class="evidence-note">True target gain: 0.65. The guard correctly withholds all weak-excitation estimates in this cohort. Their poor interval coverage is retained above, rather than presented as meaningful confidence.</p><h3>Spatial FIR · 12 runs per source condition</h3><div class="table-scroll"><table><thead><tr><th>Sources</th><th>Mean raw → residual cross-correlation</th><th>Known-source correlation</th><th>Dependence-screen rejections</th></tr></thead><tbody>${data.benchmarks.acoustic.map(r=>`<tr><td>${escapeHTML(r.scenario.replaceAll('_',' '))}</td><td>${r.mean_raw_cross_correlation.toFixed(3)} → ${r.mean_residual_cross_correlation.toFixed(3)}</td><td>${r.mean_known_source_correlation.toFixed(3)}</td><td>${r.dependence_screen_rejections}/${r.replications}</td></tr>`).join('')}</tbody></table></div><p class="evidence-note">Independent-source runs were rejected 2/12 times: nominal screen level is not established finite-sample calibration. Colored sources can produce innovations rather than faithful waveforms. Dependent-but-uncorrelated sources often pass; correlation is not an independence certificate.</p>`;
  $('context-results').innerHTML=`<h3>First fresh cohort · 66 runs, seeds 105000–105065</h3>${summaryTable(data.context.summaries)}<p class="evidence-note">The refiner had additional healthy training and 60-reading warm-up. The original four retained their weights and 32-reading warm-up. Zero false events over 26.58 h does not imply a zero population rate (one-sided 95% Poisson upper limit ≈ 0.113/hour).</p><h3>Matched-data ablation · 66 new runs, seeds 106000–106065</h3>${summaryTable(data.ablation.summaries)}<div class="guard-status failed">Clustering superiority is not established: one extra detection, equal false alarms.</div><p class="evidence-note">Both refiners use the same initial model, training runs, augmented forecasts, penalty, calibration runs and 60-reading warm-up. Only clustering/routing and groupwise residual normalization differ. Paired exact McNemar p = ${data.ablation.paired_leak_detection.two_sided_exact_mcnemar_p.toFixed(2)}. Thresholds were frozen before test generation. This experimental method remains available for inspection in the main gas lab.</p>`;
  const r=data.recorded_transfer;
  $('recorded-transfer').innerHTML=`<div class="method-metrics">${metric('Unique waveform rows',r.unique_recording_rows,'12 conditions · 57 rows per condition')}${metric('Median raw correlation',valueText(r.median_raw_correlation,3))}${metric('Median separated correlation',valueText(r.median_separated_correlation,3))}${metric('Median SI-SDR gain',`${r.median_si_sdr_improvement_db.toFixed(1)} dB`,'Compared with known recorded reference')}</div><p class="method-explain">All GPLA waveforms were kept out of separator training. Each was placed into an imposed five-channel propagation model with recorded blower interference. This tests transfer to recorded waveforms under a controlled mixing model. The 684 rows are not claimed to be independent acquisition sessions.</p><p class="evidence-note">GPLA-12 contains gas-leak recordings under pressure/noise/microphone conditions, with no healthy class. Its channels are not released as synchronized arrays. This experiment therefore measures reference-waveform recovery, not field leak accuracy or location. Sampling uses an artificial common index. <a href="https://github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12" target="_blank" rel="noopener noreferrer">Inspect the source data ↗</a></p>`;
}

async function executePaper(form, path, payload, render) {
  if(paperBusy)return;paperBusy=true;const buttons=document.querySelectorAll('.method-form button');buttons.forEach(b=>b.disabled=true);form.setAttribute('aria-busy','true');
  message('Computing the paper method from measured data on the server…');
  try{const data=await request(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});render(data);message(`Experiment complete in ${data.compute_seconds.toFixed(3)} s. Inspect the assumptions and export its results.`);}
  catch(error){message(error.message,true);}finally{paperBusy=false;buttons.forEach(b=>b.disabled=false);form.setAttribute('aria-busy','false');}
}

$('network-form').addEventListener('submit',event=>{event.preventDefault();executePaper(event.currentTarget,'/api/labs/network',{scenario:$('network-scenario').value,sensor_mode:$('network-sensors').value,seed:Number($('network-seed').value)},renderNetwork);});
$('information-form').addEventListener('submit',event=>{event.preventDefault();executePaper(event.currentTarget,'/api/labs/information',{excitation:$('information-excitation').value},renderInformation);});
$('acoustic-form').addEventListener('submit',event=>{event.preventDefault();executePaper(event.currentTarget,'/api/labs/acoustic',{scenario:$('acoustic-scenario').value,seed:Number($('acoustic-seed').value)},renderAcoustic);});
$('acoustic-upload').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  try{
    if(file.size>400000)throw new Error('Acoustic CSV exceeds 400 kB.');
    const lines=(await file.text()).replace(/^\uFEFF/,'').trim().split(/\r?\n/),headers=lines.shift().split(',').map(s=>s.trim());
    if(headers.length<2||headers.length>8||headers.some((h,i)=>h!==`m${i+1}`))throw new Error('Use 2–8 synchronized columns named m1,m2,… in order.');
    const measurements=lines.filter(s=>s.trim()).map(line=>{const parts=line.split(',');if(parts.length!==headers.length||parts.some(p=>!p.trim()||!Number.isFinite(Number(p))))throw new Error('Every row must contain the same finite numeric channels.');return parts.map(Number);});
    const healthy_samples=Number($('acoustic-healthy').value);
    if(measurements.length<768||measurements.length>4096||!Number.isInteger(healthy_samples)||healthy_samples<512||healthy_samples>3840||measurements.length-healthy_samples<256)throw new Error('Use 768–4,096 rows, at least 512 healthy rows and 256 later monitoring rows.');
    await executePaper($('acoustic-form'),'/api/labs/acoustic/analyze',{measurements,healthy_samples},renderAcoustic);
  }catch(error){message(error.message,true);}finally{event.target.value='';}
});

request('/api/paper-methods').then(data=>{renderNetwork(data.examples.network);renderInformation(data.examples.information);renderAcoustic(data.examples.acoustic);renderPaperEvidence(data);}).catch(error=>{ $('method-benchmarks').textContent=`Paper-method evidence unavailable: ${error.message}`; });
