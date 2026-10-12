"""Compare feature sets x regressors under all CV protocols AND show what each predicts on test.

Motivation (EDA, H12/H14): in DINOv2 space the sandy iPhone test photos sit next to H493 only (52 %
fines), while hand-crafted mm-scale texture features place them inside the training cloud. A model
that is good in CV but leans on the shifted features can still be wrong on test, so we print, per
model, the mean predicted % < 0.063 mm for the 7 sandy test samples and the nearest training sample.
Results are appended to results/experiments.csv for the research log.

usage: FEATS=feats_xxx.npz python compare_models.py [tag]
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from common import COLS, RES_DIR, make_monotone
from model import FEATS, FEATURE_SETS, Yimg, d, evaluate, fit_predict, regressors

TAG = sys.argv[1] if len(sys.argv) > 1 else os.path.splitext(FEATS)[0]
P = ("loso", "xcam", "group", "xgroup")
te = d["split"] == "test"
tsamp = d["sample"][te]
SANDY = [s for s in sorted(set(tsamp)) if "Airbus" in s or "Kleinkummerfeld" in s]
TEST_SETS = {"hand": d["hand"][te], "emb": d["emb"][te], "hand+emb": np.hstack([d["hand"][te], d["emb"][te]])}
i063, i2 = COLS.index("0.063"), COLS.index("2")

rows = []
print(f"{'features':10s} {'model':12s} " + " ".join(f"{p:>7s}" for p in P) + "   test: sandy%<63um  Münster%<2mm")
for fname, X in FEATURE_SETS.items():
    for rname, make in regressors().items():
        cv = {p: evaluate(X, make, p) for p in P}
        allm = np.ones(len(X), bool)
        pt = {s: make_monotone(fit_predict(make, X, Yimg, TEST_SETS[fname][tsamp == s]).mean(0))[0]
              for s in sorted(set(tsamp))}
        sandy_fines = np.mean([pt[s][i063] for s in SANDY])
        mu_f2 = pt["HPC_Muenster_BS6_9_0-10m"][i2]
        print(f"{fname:10s} {rname:12s} " + " ".join(f"{cv[p]:7.2f}" for p in P)
              + f"   {sandy_fines:16.1f}  {mu_f2:12.1f}", flush=True)
        rows.append(dict(time=datetime.now().isoformat(timespec="seconds"), feats=TAG, features=fname, model=rname,
                         **cv, test_sandy_fines=round(sandy_fines, 2), test_muenster_F2=round(mu_f2, 2)))
EXP = f"{RES_DIR}/experiments.csv"
pd.DataFrame(rows).to_csv(EXP, mode="a", index=False, header=not os.path.exists(EXP))
print("appended", len(rows), "rows to", EXP)
