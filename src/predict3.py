"""Physics submission + SandSnap sand model for samples that are sand by measurement (H16).

Base: submission_physics.csv (LB-best, 49.448): SAM-calibrated coarse part (linear, w=1) + ensemble
fines shape. Gate (automatic): SAM median area % of grains > 2 mm (4.6 px/mm pipeline) < GATE_PCT
→ sand. For gated samples the part of the curve below 2 mm is replaced (weight W) by the sand model
(cache/sand_preds.csv, trained on SandSnap only), rescaled to the base curve's F(2 mm):
    F(d) = (1-W)·F_base(d) + W·F_sand(d)·F_base(2)/100      for d < 2 mm
No human-chosen values enter the prediction (rule §4.b).

usage: python src/predict3.py W [GATE_PCT]   -> submissions/submission_sand_w{W*100}[_g{GATE}].csv
(refuses to overwrite files already scored on the LB — see submissions/scored/)
"""
import os
import sys

import numpy as np
import pandas as pd

from common import CACHE_DIR, COLS, DATA_DIR, SUB_DIR, make_monotone
from grains_eval import load_grains
from sam_grains import SIZES_MM

W = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
GATE_PCT = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

base = pd.read_csv(f"{SUB_DIR}/submission_physics.csv").set_index("sample_id")
SAND_VER = os.environ.get("SAND_VER", "v1")  # v1 produced the LB-scored 30.389 file
# "v1+v2" averages the curves of several sand-model versions (EMD is convex -> the average is never
# worse than the mean of the individual errors; reduces variance where versions disagree)
sand = sum(pd.read_csv(f"{CACHE_DIR}/sand_preds_{v}.csv").set_index("sample_id")[COLS] for v in SAND_VER.split("+"))
sand = sand / len(SAND_VER.split("+"))
d = np.load(f"{CACHE_DIR}/feats_dinov2base.npz", allow_pickle=True)
te = d["split"] == "test"
G = 100 * load_grains(d["path"])["frac"][te]
j2 = int(np.argmin(np.abs(SIZES_MM - 2)))
k2 = COLS.index("2")

rows = []
for s in pd.read_csv(f"{DATA_DIR}/sample_submission.csv").sample_id:
    b = base.loc[s, COLS].values.astype(float)
    gravel_area = np.median(G[d["sample"][te] == s, j2])
    p = b.copy()
    if gravel_area < GATE_PCT:
        fs = sand.loc[s, COLS].values.astype(float)
        p[:k2] = (1 - W) * b[:k2] + W * fs[:k2] * b[k2] / 100
    p = make_monotone(p)[0]
    rows.append([s, *np.round(p, 4)])
    print(f"{s:28s} SAM>2mm={gravel_area:5.1f}%  {'SAND' if gravel_area < GATE_PCT else '    '}  "
          f"F(0.063) {b[3]:5.1f}->{p[3]:5.1f}  F(0.2) {b[4]:5.1f}->{p[4]:5.1f}  F(0.63) {b[5]:5.1f}->{p[5]:5.1f}")
tag = (f"submission_sand_w{int(round(W * 100))}" + ("" if GATE_PCT == 20.0 else f"_g{int(GATE_PCT)}")
       + ("" if SAND_VER == "v1" else "_" + SAND_VER.replace("+", "")))
out = f"{SUB_DIR}/{tag}.csv"
if os.path.exists(f"{SUB_DIR}/scored/{tag}.csv"):
    sys.exit(f"refusing to overwrite LB-scored {tag}.csv (copy in submissions/scored/)")
pd.DataFrame(rows, columns=["sample_id", *COLS]).to_csv(out, index=False)
print("wrote", out)
