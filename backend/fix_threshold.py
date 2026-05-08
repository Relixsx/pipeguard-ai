"""
Run this from the backend/ folder:
  python fix_threshold.py
"""
import json
from pathlib import Path

path = Path("../models/threshold.json")
with open(path) as f:
    cfg = json.load(f)

mean = cfg["stats"]["mean"]
std  = cfg["stats"]["std"]
old  = cfg["threshold"]

# Raise to 8σ — streaming single readings always score higher than
# training sequences due to lack of temporal continuity in the buffer
new = mean + 8.0 * std
cfg["threshold"] = round(new, 6)
cfg["stats"]["sigma"] = 8.0

with open(path, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"Old threshold : {old:.6f}")
print(f"New threshold : {new:.6f}  (mean={mean:.4f} + 8σ={std:.4f})")
print("Restart the server now.")
