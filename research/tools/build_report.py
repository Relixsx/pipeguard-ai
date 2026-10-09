from pathlib import Path
import ast,csv,html,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT
RESULTS=PROJECT/'results'
FIG=PROJECT/'figures';FIG.mkdir(exist_ok=True)
OUT=ROOT/'PipeGuard_Executed_Research_and_Results.pdf'
OUT.parent.mkdir(exist_ok=True)
primary=json.loads((RESULTS/'benchmark_results.json').read_text())
regime=json.loads((RESULTS/'regime_results.json').read_text())
inventory=json.loads((RESULTS/'inventory_results.json').read_text())
audio=json.loads((RESULTS/'acoustic/acoustic_results.json').read_text())
manifest=json.loads((PROJECT/'data/simulation/manifest.json').read_text())
audit=json.loads((PROJECT/'evidence/source_audit.json').read_text())

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.labelcolor':'#334155','text.color':'#122438',
                     'axes.edgecolor':'#94a3b8','savefig.facecolor':'white'})
palette=['#94a3b8','#2d5c88','#137e80','#db9148']
names=['Original 3','CNN-LSTM 8','Dynamic','Hybrid']
keys=['original_three_features','cnn_lstm','dynamic_residual','hybrid']
fig,axes=plt.subplots(1,2,figsize=(8,3.1),layout='constrained')
for ax,metric,multiplier,title in zip(axes,['event_recall','false_alarms_per_hour'],[100,1],
                                    ['Leak events detected (%)','False alarm events / healthy hour']):
    for i,key in enumerate(keys):
        values=np.array([r[metric] for r in primary['summaries'] if r['model']==key])*multiplier
        avg=values.mean()
        ax.bar(i,avg,color=palette[i],width=.58)
        if len(values)>1:ax.errorbar(i,avg,yerr=[[avg-values.min()],[values.max()-avg]],color='#122438',capsize=4,fmt='none')
        ax.text(i,avg+(.05 if multiplier==1 else 2),f'{avg:.2f}' if multiplier==1 else f'{avg:.1f}',ha='center',fontsize=9)
    ax.set_xticks(range(4),names,rotation=15,ha='right')
    ax.set_title(title,loc='left',fontsize=10,fontweight='bold')
    ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
axes[0].set_ylim(0,110);axes[1].set_ylim(0,4.55)
fig.savefig(FIG/'primary_comparison.png',dpi=220);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,3.55),layout='constrained')
for name,color,label in [('cnn_lstm','#94a3b8','CNN-LSTM'),('dynamic_residual','#2d5c88','Dynamic'),
                         ('hybrid','#db9148','Hybrid'),('mass_inventory','#137e80','Inventory')]:
    rows=[next(r for r in regime['summaries'] if r['model']==name and r['calibration_coverage']==coverage) for coverage in ['narrow','broad']]
    x=[r['false_alarms_per_hour'] for r in rows];y=[r['event_recall']*100 for r in rows]
    ax.plot(x,y,color=color,lw=1.7,alpha=.9)
    ax.scatter(x[0],y[0],color=color,s=55,marker='o',facecolors='none',linewidths=1.5)
    ax.scatter(x[1],y[1],color=color,s=65,marker='s',label=label)
ax.axvline(.1,color='#b91c1c',ls='--',lw=1,label='Target: 0.1/h')
ax.set(xlabel='False alarm events / healthy hour (lower is better)',ylabel='Leak event recall (%)',xlim=(-.10,4.45),ylim=(-8,108))
ax.grid(alpha=.15);ax.legend(loc='center right',fontsize=8,frameon=False)
ax.set_title('Healthy calibration coverage changes the operating trade-off',loc='left',fontsize=11,fontweight='bold')
fig.savefig(FIG/'calibration_tradeoff.png',dpi=220);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(8,2.9),layout='constrained')
labels=['Dynamic','CNN-LSTM','Hybrid']
axes[0].bar(labels,[r['true_positive'] for r in audio['summaries']],color=[palette[2],palette[1],palette[3]])
axes[0].set(ylim=(0,22),ylabel='Leak clips detected / 20')
axes[1].bar(labels,[r['false_positive'] for r in audio['summaries']],color=[palette[2],palette[1],palette[3]])
axes[1].set(ylim=(0,4),ylabel='Healthy clips flagged / 10')
for ax in axes:ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
fig.savefig(FIG/'acoustic_pilot.png',dpi=220);plt.close(fig)

pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVuBold',italic='DejaVu',boldItalic='DejaVuBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle('BodyPG',fontName='DejaVu',fontSize=10.2,leading=15,textColor=colors.HexColor('#24364b'),spaceAfter=9))
styles.add(ParagraphStyle('SmallPG',parent=styles['BodyPG'],fontSize=8.8,leading=12.6,spaceAfter=7))
styles.add(ParagraphStyle('CellPG',parent=styles['BodyPG'],fontSize=8.5,leading=12,spaceAfter=0))
styles.add(ParagraphStyle('H1PG',fontName='DejaVuBold',fontSize=22,leading=27,textColor=colors.HexColor('#102b46'),spaceAfter=13))
styles.add(ParagraphStyle('H2PG',fontName='DejaVuBold',fontSize=12,leading=16,textColor=colors.HexColor('#137e80'),spaceBefore=9,spaceAfter=7))
styles.add(ParagraphStyle('KickerPG',fontName='DejaVuBold',fontSize=9,leading=13,textColor=colors.HexColor('#137e80'),spaceAfter=9))
story=[];pages=[]

def p(text,style='BodyPG'):
    story.append(Paragraph(text,styles[style]))
def h(text):p(text,'H2PG')
def page(label,title):
    if story:story.append(PageBreak())
    pages.append(title)
    p(label.upper(),'KickerPG');p(title,'H1PG')
def table(headers,rows,widths):
    data=[[Paragraph(str(c),styles['CellPG']) for c in row] for row in [headers]+rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e0eef1')),
                          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f4f7fa'),colors.white]),
                          ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),
                          ('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),
                          ('BOTTOMPADDING',(0,0),(-1,-1),8),
                          ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#83b4ba'))]))
    story.extend([t,Spacer(1,10)])
def note(text):
    t=Table([[Paragraph(text,styles['BodyPG'])]],colWidths=[491])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#eaf4f2')),
                          ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),
                          ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.extend([t,Spacer(1,10)])
def image(name,height):story.extend([Image(str(FIG/name),width=491,height=height),Spacer(1,8)])
def link(url,label):return f'<link href="{html.escape(url,quote=True)}" color="#137e80">{html.escape(label)}</link>'

page('01 / Execution outcome','PipeGuard: research converted into a working prototype')
p('Source audit, ten-paper translation, trained comparisons, failure analysis and interview material. Prepared for a research discussion with Dr. Arne Dankers. Execution date: 9 October 2026.')
note('<b>Main finding:</b> adding CNN-LSTM to the dynamic residual did not establish a benefit. The strongest simulation follow-up combined gas-inventory balance with healthy operating-regime coverage.')
table(['Requested step','Concrete execution'],[
    ['1. Suitable data','150 real gas-acoustic files acquired and audited. Closest field SCADA and multiphase sources qualified; raw access remains unresolved.'],
    ['2. Paper-derived method','All ten papers mapped to explicit implementation choices and limits.'],
    ['3. Build comparisons','CNN-LSTM, dynamic ARX and hybrid trained; three neural seeds, plus an original-input ablation.'],
    ['4. Test failures','Independent simulation runs, real-audio pilot, inventory sensitivity and two fresh-seed follow-ups.'],
    ['5. Reviewable deliverable','Source, weights, data generator/cohort, result tables, local API, ten passing tests and interview brief.']], [128,363])
p('<b>Strongest follow-up:</b> 36/36 simulated leaks detected, seven false alarm events in 28.58 healthy hours, median detected-event delay 94.5 s. The observed 0.245 alarms/h exceeds the 0.1/h target. Known geometry and gas assumptions make this an optimistic physics comparison.','SmallPG')
p('<b>Evidence boundary:</b> this package is a reproducible research prototype. Field pressure/flow validation, calibrated component attribution and synchronized multichannel source separation remain open. No GitHub push, publication or message to an author was made.','SmallPG')

page('02 / Source audit','Fix the evidence pipeline before choosing a larger model')
p('Repository inspected at <b>f3acfde210161a396cb1835bcabd2a413ba20e78</b>. The following findings are computed from the code snapshot and CSV, not inferred from the project description.')
table(['Observed issue','Implication / executed extension'],[
 ['1,000 rows, 50 segments, only 17 timestamps; 408 duplicate segment-time rows.','Maximum 23 normal rows per segment cannot supply a valid 30-reading normal window for an asset.'],
 ['All 665 original normal windows mix segments.','New experiments retain run identity and chronology; no window crosses a run or split.'],
 ['Scaling precedes splitting; 29 raw rows overlap the window split; validation is synthetic.','Healthy training scaling, separate validation/calibration runs, and independent final test runs.'],
 ['One shared prediction buffer; API accepts three floats without identity/time.','Asset-specific buffers and alarm state; ordered timestamps, unit-labelled schema, missing-data reset.'],
 ['Batch demo uses supplied anomaly labels and random scores; anomaly-only logging.','New local API calls saved models and logs all accepted readings, including healthy exposure.'],
 ['Threshold metadata contains two inconsistent values; original weights are LFS pointers.','Calibrated thresholds, complete checkpoints and frozen result protocols included.']], [235,256])
p('The CNN-LSTM topology is preserved: two temporal convolution layers, a two-layer LSTM encoder, 16-dimensional latent state, repeated-latent LSTM decoder and reconstruction loss. We retrained it; the missing original weights were not used.','SmallPG')
p('Source audit and file hashes are in <b>evidence/source_audit.json</b>. This is a standalone research extension, with an integration adapter; it does not silently rewrite the upstream application.','SmallPG')

page('03 / Data qualification','The right task and modality come first')
table(['Source','Fit to your task','Actual access result'],[
 ['NGPOD / MKTCN [S1]','Natural-gas pipeline SCADA, multiple stations and boundary flow. Strongest field lead.','Paper found. Linked notebook expects missing pipeline_datasets arrays. Raw chronological data not acquired.'],
 ['OLGA multiphase study [S2]','Pressure, temperature and inlet/outlet mass flow; controlled pipeline leaks.','Processed data/code require an author request; redistribution may be restricted. Files not acquired.'],
 ['Meng gas-acoustic data [S3]','Real gas-leak acoustics with healthy background, hole and valve folders.','150 unique mono, 96 kHz, one-second files downloaded and hashed. Session identifiers and full release not verified.'],
 ['Declared gas simulator','Aligned multi-point telemetry and known physical events for method development.','142 primary runs generated, with separate fresh-seed follow-ups. Synthetic, explicitly labelled.']], [118,176,197])
h('Why the acoustic dataset is a separate experiment')
p('Audio can support gas-leak detection, but it does not supply pressure, mass flow or gas temperature. Its recordings cannot be inserted into the SCADA predictor while retaining the same physical interpretation. The pilot uses its own spectral feature schema.')
h('What a useful larger dataset must include')
p('Require synchronized multi-point pressure, both boundary mass flows, temperature, units, asset/run identifiers, controls, topology/geometry, confirmed leak timing and healthy operating changes. A larger row count cannot repair missing event provenance or unobserved boundary flow.')
p('The access packet names the closest providers and requests raw runs before any overlapping-window preparation. Oil-well operational and water-network datasets were excluded as the main gas-pipeline benchmark.','SmallPG')

page('04 / Method','Retain a four-stage workflow with stronger contracts')
table(['Stage','Required behaviour'],[
 ['1. Align measurements','Validate asset/time, units, sampling and sensor quality. Retain relevant pressure nodes, flows and controls.'],
 ['2. Learn a healthy reference','Fit scaling and dynamics on healthy training runs. Validate the predictor and inspect available target information.'],
 ['3. Compute residuals','Compare observed behaviour against dynamics and mass inventory; evaluate neural residual scoring as an ablation.'],
 ['4. Decide and explain','Calibrate alarms on healthy regimes, apply persistence, log the healthy denominator and keep attribution separate.']], [142,349])
h('What is physically different from raw reconstruction')
p('A legitimate outlet-valve change can create inlet/outlet imbalance because the gas stored in the pipe is changing. The inventory residual subtracts that storage change before calling the remaining imbalance anomalous. [R2]')
note('<b>Cell storage:</b> M<sub>i</sub> = V<sub>i</sub> p<sub>i</sub> / (Z R T).<br/><b>Mass balance:</b> dM<sub>i</sub>/dt = q<sub>in,i</sub> - q<sub>out,i</sub> - q<sub>leak,i</sub>.<br/><b>Inventory score:</b> |average(q<sub>in</sub> - q<sub>out</sub>) - change(total M)/window duration|.')
p('The simulator uses three fixed 21.2 m3 cells, absolute pressure in Pa, flow in kg/s, temperature in K, constant Z = 0.9 and R = 518.3 J/(kg K). Inter-cell flow follows a declared quasi-steady square-pressure relationship. External excitation and realized feedback commands are distinguished.')
p('Temperature is constant within a generated run. The model does not reproduce full gas wave propagation, non-isothermal dynamics or multiphase flow. Its conservation check is numerical verification, not field validation.','SmallPG')

page('05 / Papers D1-D6','Separate forecasting, identifiability and valid diagnosis')
table(['Paper insight','Applied decision','Claim boundary'],[
 ['D1: consistent module estimation needs justified inputs/noise assumptions.','Fit a frozen healthy multivariate dynamic predictor.','ARX forecast accuracy alone does not establish a consistent physical-module estimate.'],
 ['D2: omitted paths can change a target model.','Retain all measured pressure nodes, both flows and controls; compare a reduced-input baseline.','A field sensor graph must be assessed before removing channels.'],
 ['D3: distinguishable model structure comes before parameter interpretation.','Declare the simulated graph and exposed variables.','No field-graph identifiability result is claimed.'],
 ['D4: target identifiability differs from whole-network ambiguity.','Report target-block partial regression alongside full-design conditioning.','A low-rank larger design need not invalidate a target.'],
 ['D5: confounding and feedback bias naive causal estimates.','Expose realized feedback and external excitation separately.','A controller command is not automatically a valid exogenous instrument.'],
 ['D6: actual information can be target-specific.','Evaluate weak excitation, keep reference frozen and attribution separate.','A singular-value cutoff is not the paper theorem or a calibrated confidence bound.']], [166,160,165])
p('<b>Implemented diagnostic:</b> the standardized design condition number rises from about 116 in training to 810 in weak-excitation runs. Conditional target-block eigenvalues are reported separately. Measurement noise and nuisance structure matter; this panel does not justify physical component attribution.','SmallPG')
p('An anomaly channel remains active even when attribution is unsupported. The monitor returns <b>not_established</b> for component attribution; it does not label every deviation as a leak.','SmallPG')

page('06 / Papers D7-D10','Use acoustic and context results where their inputs exist')
table(['Paper','What transfers','What cannot yet be reproduced'],[
 ['D7: acoustic separation via identification','Synchronised mixing measurements and a suitable channel model can separate interfering sources.','Our acquired WAVs are single-channel. Temporal AR forecasting is not blind source separation.'],
 ['D8: pipeline field study','Study spatial inverse-FIR filtering, normalization, regularization and latency under the actual data layout.','Paired simultaneous DAS/microphone channels are absent. Multiple mono clips are not spatial channels.'],
 ['D9: context-aware refinement','Normal behaviour depends on operating regimes and known actions.','Future maintenance/event labels and retrospective regime assignments must not enter a live score.'],
 ['D10: batched decomposition','Fit the same transformation available at deployment and obey the forecast boundary.','Future-looking decomposition can distort alarm evidence; no such transform was used here.']], [123,181,187])
h('A concrete discussion point from the field-study method')
p('D8 uses an inverse FIR approach with a local spatial neighbourhood and a regularization choice that depends on singular-value scale. A reproduction should examine sensitivity to channel scaling, regularization and held-out interference. A timing number for one filtering batch does not establish full alarm latency.')
p('Mutual-information estimation is not a statistical independence p-value. This prototype does not use it as one. For the next acoustic study, keep the downstream classifier fixed while comparing separation methods on independent sessions and noise conditions.')
p('D9-inspired healthy-regime coverage was actually tested after the narrow-regime failure. D10-inspired time boundaries were checked by matching offline and per-asset streamed scores.','SmallPG')

page('07 / Experimental contract','A model comparison that can be audited and rerun')
table(['Partition / setting','Frozen role'],[
 ['40 training runs, 12 validation runs','Healthy only. Training scales and fits models; validation chooses ARX order/penalty and neural stopping.'],
 ['24 calibration runs','Healthy only. Choose thresholds for the declared 0.1 alarm/h budget. Effective scored exposure: 22.97 h.'],
 ['66 primary test runs','36 abrupt/gradual leak events at 2%, 5%, 10% nominal-flow magnitude, three locations and two regimes; 30 other runs.'],
 ['Input access','Main neural/dynamic comparison shares three pressures, two flows, temperature and two controls. Original three-feature baseline is separate.'],
 ['Neural fitting','16-epoch limit, three seeds (11, 23, 37), 4,600 training windows/model. About 143,000 parameters.'],
 ['Alarm logic','30 readings at 5 s cadence; three high scores to confirm, five low scores to clear, 120 s notification cooldown.']], [173,318])
h('Event detection must not reward an existing false alarm')
p('Only a new confirmed alarm at or after physical leak onset counts as a detected event. An alarm active before onset receives no detection credit. This was corrected during execution and the saved weights/thresholds were rescored; a regression test protects the definition.')
p('False alarm events use healthy and pre-leak exposure. Sensor bias and missing data are separate failure tests. Missing windows are suppressed with a data-quality alert rather than silently classified as healthy.')
p('Raw CNN-LSTM reconstructs eight variables; ARX forecasts six physical channels; the hybrid monitors six innovations. Access is matched, while objectives differ. Three seeds reuse the same 36 events; report initialization ranges separately from event-recall intervals.','SmallPG')

page('08 / Primary comparison','The dynamic residual helped; the neural hybrid did not')
table(['Model','New leak detections','False alarms/h','Median detected delay'],[
 ['Original three features, one seed','26/36 (72.2%)','3.19','388 s'],
 ['CNN-LSTM, shared 8 inputs, 3 seeds','28-31/36 (77.8%-86.1%)','3.22-3.33','338-352 s'],
 ['Dynamic residual','35/36 (97.2%)','3.92','99 s'],
 ['Residual CNN-LSTM hybrid, 3 seeds','35/36 (97.2%)','3.92-3.95','89-104 s']], [192,116,85,98])
image('primary_comparison.png',190)
p('Bars show means; whiskers show initialization ranges, using the same test runs. All methods miss the operating false-alarm target. The dynamic model improves event detection, but its linear approximation and narrow calibration do not explain the full range of normal operating changes.','SmallPG')
p('The hybrid preserves dynamic detection but supplies no demonstrated event-recall gain, and slightly increases false notifications for one seed. There is no basis here for retaining its extra complexity as the default method.','SmallPG')
p('Dynamic recall 35/36 has a conditional exact 95% interval of about 85.5%-99.9%. Full per-run alarms, misses, p90 delays and point metrics are in <b>benchmark_results.json</b>.','SmallPG')

page('09 / Fresh-seed follow-up','Physical structure and normal-regime coverage change the trade-off')
p('After inspecting the primary failures, we added a mass-inventory residual, then 30 independent healthy runs covering wider controls, transients and weak excitation. We froze weights, calibrated thresholds on 51.68 scored healthy hours, then generated a third 66-run cohort (seeds 81000-81065). This is an exploratory in-simulator follow-up.')
table(['Broader-calibrated model','New detections','False alarms/h','Median delay'],[
 ['CNN-LSTM, seed 11','0/36','0.070','No detections'],
 ['Dynamic residual','25/36','0.210','212 s'],
 ['Hybrid, seed 11','25/36','0.210','212 s'],
 ['Mass-inventory residual','36/36','0.245','94.5 s']], [227,90,90,84])
image('calibration_tradeoff.png',214)
p('Open circles: narrow calibration. Filled squares: broader healthy calibration. Threshold changes use healthy data only. Raising the neural-only threshold suppresses false alarms by suppressing all leak detections. Inventory balance retains detection, but seven false alarms still exceed the observed target.','SmallPG')
p('The inventory model uses the simulator geometry/gas law; this is favourable to it. Separate +/-20% volume sensitivity and a fresh replication are included. Those experiments do not establish field robustness.','SmallPG')
p('Broader inventory calibration flagged 0/6 sensor-bias cases; the dynamic/hybrid models flagged 6/6. A sensor-quality branch still needs separate validation and a shared alarm budget.','SmallPG')

page('10 / Real gas-recording pilot','Relevant recordings, limited validation strength')
p('Downloaded 150 unique WAVs: 50 healthy-background, 50 hole-leak and 50 valve-leak files, using source-folder labels. Every file is mono, 16-bit PCM, 96 kHz and one second long. The archived manifest retains file IDs, hashes and header checks. [S3]')
p('Prefix blocks 0-14 train, 15-19 validate, 20-24 test. Both suffix files and all classes of each prefix remain in one split. Only healthy clips train the three anomaly models. Features are 16 log-power bands from 100 Hz to 48 kHz; 2,048-sample STFT windows with 512-sample hops.')
table(['Model','Leak clips caught / 20','Healthy false positives / 10'],[
 ['Dynamic spectral predictor','20/20','1/10'],['CNN-LSTM','8/20','0/10'],['Residual hybrid','20/20','3/10']], [201,137,153])
image('acoustic_pilot.png',170)
p('<b>Do not read this as 100% field leak accuracy.</b> The 20/20 recall interval is approximately 83.2%-100%; the dynamic false-positive interval from 1/10 is approximately 0.3%-44.5%. Prefixes are provisional groups; independent acquisition sessions, pressure/SNR metadata and the full release are unverified.','SmallPG')
p('Maximum observed clipped-sample fraction is 1.28% in a clip. There is no event timeline for delay measurement and only ten healthy test seconds. No spatial BSS or location estimate was made.','SmallPG')

page('11 / Five solution paths','Choose a route by its measurable research question')
table(['Solution / paper link','Executed evidence','Next decisive evidence'],[
 ['1. Dynamic residual monitoring<br/>D1, D2, D5','35/36 primary simulated detections; excess alarms during operating changes.','Graph/noise-appropriate estimator on aligned field telemetry; held-out control changes.'],
 ['2. Inventory balance plus operating coverage<br/>D2, D9 + gas physics','Strongest follow-up: 36/36, 0.245 alarms/h, 94.5 s median.','Thermal/geometry/flow-meter uncertainty, higher-fidelity simulation and field events.'],
 ['3. Residual CNN-LSTM hybrid<br/>D1, D9, D10','No demonstrated gain over dynamic predictor; audio hybrid adds false positives.','Independent improvement at a matched alarm budget. Retain only if this test supports it.'],
 ['4. Information-aware diagnosis<br/>D3, D4, D6','Target/full-design diagnostics and separate attribution status implemented.','Validated estimator and calibrated target uncertainty under weak excitation/confounding.'],
 ['5. Multichannel acoustic separation<br/>D7, D8','Single-channel gas pilot completed. Full separation remains data-blocked.','Synchronized mixtures, independent recording groups and a fixed-classifier separation ablation.']], [171,165,155])
note('<b>Recommended near-term build:</b> aligned gas telemetry, inventory/dynamic residual channels and verified healthy operating coverage. Add component attribution when the target is identifiable and informative. Retain a neural branch only when it contributes independent held-out evidence.')
p('A plausible research contribution is finite-data, target-specific diagnostic validity under weak excitation and shared disturbances. Basic dynamic-network fault diagnosis already exists [R1]; a hybrid architecture alone does not establish novelty.','SmallPG')

page('12 / Interview brief','Lead with an experiment and a question you can explain')
h('A concise opening')
note('"My original PipeGuard prototype used CNN-LSTM reconstruction. Your papers led me to examine which measured relationships can actually be learned, what normal control changes look like, and when a fault can be attributed to a component. I audited the original windows, rebuilt the experiment with independent runs, and compared neural reconstruction, dynamic residuals and a hybrid. The hybrid did not add a clear benefit. A gas-inventory residual with broader healthy calibration gave the strongest simulation result, but the alarm rate still needs work. I want to investigate target-specific information and uncertainty using aligned real pipeline data."')
h('Three findings to be ready to explain')
p('<b>1. Stored gas:</b> inlet flow can exceed outlet flow during a normal inventory increase. A leak decision must account for that change and measurement uncertainty.')
p('<b>2. Detection versus diagnosis:</b> detecting an unexplained residual does not identify a physical cause. Graph structure, excitation, confounding and the chosen estimator determine what can be attributed.')
p('<b>3. Model complexity:</b> in these tests, adding a neural residual branch did not improve event detection. A reproducible negative result identifies the next useful experiment.')
h('Specific questions for Dr. Dankers')
p('Which estimator and noise assumptions would you consider defensible for this graph with realized feedback commands and pressure/flow measurement error?')
p('How could the target-specific informativity conditions be turned into a finite-window diagnostic that supports uncertainty without disabling anomaly detection?')
p('For your acoustic pipeline approach, what synchronized-channel and healthy-reference data would be needed to reproduce the filtering stage and then test it against independently confirmed leak events?')
p('Describe this as AI-assisted research that you can reproduce and explain. Present measured limitations and the proposed experiment; the source papers supply the theory, and this prototype tests an engineering approximation.','SmallPG')

page('13 / Data request and next experiment','Make the field-data gap concrete')
table(['Provider / route','Specific request'],[
 ['NGPOD / MKTCN authors [S1]','Raw chronological station telemetry, IDs, units/aggregation, original labels and confirmed leak timing; controls, topology and geometry. Pre-windowed trainX/testX alone cannot audit overlap.'],
 ['OLGA study corresponding author and original providers [S2]','Original scenario runs and baseline, onset/severity/location, boundary conditions, healthy operating changes, code and research reuse terms.'],
 ['Meng data providers [S3]','Complete release; original recording IDs/cut times; pressure and noise conditions; synchronized microphone channels and research reuse terms.'],
 ['Research discussion with Dr. Dankers','Ask whether an appropriate instrumented lab/field dataset or collaboration could support the target-information and diagnostic-validity experiment.']], [155,336])
h('A prospective experiment, specified before accessing the final test')
p('Select a physical target and document the sensor graph. Match units, sampling and known controls. Train only on healthy runs; validate dynamics across normal regimes. Freeze preprocessing, estimator, uncertainty procedure, alarm budget and attribution rule before opening held-out events.')
p('Compare raw CNN-LSTM, dynamic residual, inventory residual and any retained hybrid with the same sensor access. Hold out complete events and acquisition sessions; where feasible also hold out sites or periods. Report misses, new-event delay, healthy exposure, false notifications and uncertainty.')
p('Keep a frozen healthy reference. Any adaptive model should update through a separately validated shadow process, with flagged periods quarantined so a persistent leak is not learned as normal. This safeguard is specified for future adaptation; no online parameter update is enabled in the delivered monitor.')
p('No access request has been sent. These requests are reviewable material for your research discussion and provider contact.','SmallPG')

page('14 / Paper references','The ten supplied papers and their implementation links')
drefs=[(r['id'],r['title'],r['url']) for r in json.loads((PROJECT/'evidence/publication_references.json').read_text())]
for rid,title,url in drefs:
    p('<b>'+rid+'</b> '+html.unescape(title)+' '+link(url,'Publication / supplied manuscript'),'SmallPG')
p('The 119-page reading ledger covers every supplied research-paper page. Two supporting project/study documents add 45 pages. These readings precede the execution phase; the new report focuses on implemented transfers and measured results rather than repeating their full summaries.','SmallPG')
(PROJECT/'evidence/publication_references.json').write_text(json.dumps([{'id':rid,'title':html.unescape(title),'url':url} for rid,title,url in drefs],indent=2))

page('15 / Sources and reproduction','Open the result, then trace it to code and data')
refs=[
 ('S1','Li et al. MKTCN natural-gas pipeline preprint, arXiv:2411.06214; data described, raw acquisition unresolved.','https://arxiv.org/html/2411.06214v1'),
 ('S1a','Author code repository; inspected tree contains README and notebook, with missing data folder.','https://github.com/MrLiXv/MKTCN'),
 ('S2','Naanouh and Henry. Multiphase pipeline leak monitoring, Digital 2026, 6(2), 45. Data/code on request.','https://doi.org/10.3390/digital6020045'),
 ('S3','Meng et al. gas-acoustic dataset repository and its linked public data folder.','https://github.com/mengdinet/Gas-pipeline-leakage-data-set'),
 ('S3a','MSFAT study, Sensors 2025, 25(20), 6390. Larger reported dataset use; full release/grouping not verified here.','https://doi.org/10.3390/s25206390'),
 ('R1','Shi, Fonken and Van den Hof (2024). Dynamic-network fault detection/diagnosis prior art, IFAC 58(15), 384-389.','https://doi.org/10.1016/j.ifacol.2024.08.559'),
 ('R2','Chevalier and Wu (2020). Dynamic Linepack Depletion Models for Natural Gas Pipeline Networks. General physical context, not the identical simulator.','https://arxiv.org/abs/2001.11496'),
 ('CODE','Relixsx/pipeguard-ai, audited commit f3acfde210161a396cb1835bcabd2a413ba20e78.','https://github.com/Relixsx/pipeguard-ai/tree/f3acfde210161a396cb1835bcabd2a413ba20e78')]
for rid,title,url in refs:p('<b>'+rid+'</b> '+title+' '+link(url,'Primary source'),'SmallPG')
h('Reproducible package contents')
p('<b>run_benchmark.py</b>: generation, fitting and narrow-calibration comparison.<br/><b>run_inventory_check.py</b>: physics residual and geometry sensitivity.<br/><b>run_regime_calibration.py</b>: healthy coverage and final fresh cohort.<br/><b>run_acoustic_pilot.py</b>: separate real-recording pilot.<br/><b>serve_research.py</b>: local per-asset model scoring and all-reading log.<br/><b>results/</b>: exact checkpoints, thresholds and per-run results.','SmallPG')
p('From the package directory, install the README dependencies, then run <b>python -m unittest discover -s tests -v</b>. Ten tests cover conservation, causal design, chronology, client isolation, quality resets, event accounting, logging and offline/streaming equivalence.','SmallPG')
p('Synthetic cohort and generator are included. Third-party audio is excluded; its download utility, IDs and hashes are included. CPU Python 3.12 / PyTorch 2.14.1 was used. See README for complete replay/training commands and the field-access limits.','SmallPG')

def decorate(c,doc):
    w,h=doc.pagesize
    c.saveState()
    c.setFillColor(colors.HexColor('#102b46'));c.rect(0,h-16,w,16,fill=1,stroke=0)
    c.setFont('DejaVuBold',8);c.setFillColor(colors.HexColor('#102b46'))
    c.drawString(52,h-43,'PIPEGUARD  /  EXECUTED RESEARCH')
    c.setFont('DejaVu',8);c.setFillColor(colors.HexColor('#60738a'))
    c.drawRightString(w-52,h-43,'9 OCT 2026')
    c.setStrokeColor(colors.HexColor('#d4e1e9'));c.line(52,48,w-52,48)
    c.setFont('DejaVu',7.5);c.drawString(52,33,'Research prototype  |  simulation and small acoustic pilot')
    c.drawRightString(w-52,33,f'{doc.page:02d}')
    c.restoreState()

doc=SimpleDocTemplate(str(OUT),pagesize=(595.28,841.89),leftMargin=52,rightMargin=52,
                      topMargin=70,bottomMargin=65,title='PipeGuard: Executed Research and Results',
                      author='PipeGuard research prototype / AI-assisted execution')
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
(PROJECT/'evidence/report_page_plan.json').write_text(json.dumps(pages,indent=2))
print(OUT)
