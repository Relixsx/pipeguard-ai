"""Register deliberately refitted context artifacts; never invoked at startup."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
models = ROOT / 'deployment/models'
protocol = json.loads((ROOT / 'deployment/evidence/context_refinement_protocol.json').read_text())
raw = (models / 'context_refinement.npz').read_bytes()
if hashlib.sha256(raw).hexdigest() != protocol['model_sha256']:
    raise RuntimeError('Context artifact does not match its calibration protocol')
config = json.loads((models / 'config.json').read_text())
config['thresholds']['context_refinement'] = protocol['selected_threshold']
config['version'] = '3.0.0'
config['warmup_by_method'] = {n: 60 if n == 'context_refinement' else 32 for n in config['thresholds']}
(models / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
manifest = json.loads((models / 'manifest.json').read_text())
manifest['format'] = 'ONNX and NumPy'
for name in ['config.json', 'context_refinement.npz', 'global_refinement_ablation.npz']:
    p = models / name
    if p.exists():
        blob = p.read_bytes()
        manifest['files'][name] = {'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
(models / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('Context threshold and artifact hashes registered')
