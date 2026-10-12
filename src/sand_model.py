"""H16: sand-range model trained on SandSnap (external, CC-BY-4.0) at 12 px/mm.

1. Domain check: is each test sample inside the SandSnap feature distribution? (nearest-SandSnap
   distance vs SandSnap's own leave-site-out nearest-neighbour distances)
2. Leave-one-SITE-out CV on SandSnap (same-beach samples look alike) for F(0.2) and F(0.63) and the
   EMD over the 11-point curve.
3. Predictions for the test samples -> cache/sand_preds.csv (curve + in-domain flag).

usage: python src/sand_model.py
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CACHE_DIR, COLS, emd, make_monotone
from sand_features import VER

d = np.load(f"{CACHE_DIR}/sand_feats_{VER}.npz", allow_pickle=True)
curves = pd.read_csv(f"{CACHE_DIR}/sandsnap_curves_{VER}.csv", index_col=0)
ss = d["src"] == "sandsnap"
X_all = np.hstack([d["emb"], d["tex"]])
# site = beach folder; tiles_20210823_set1 files are named <SITECODE><n>_..., use the letter prefix
site = np.array([(lambda f: re.match(r"[A-Za-z]+", os.path.basename(f)).group(0) if "/tiles_" in f else f.split("/images/")[1].split("/")[0])(p) if s else "" for p, s in zip(d["path"], ss)])
key = d["key"]
Y = np.stack([curves.loc[k].values if s else np.full(len(COLS), np.nan) for k, s in zip(key, ss)])
i02, i063 = COLS.index("0.2"), COLS.index("0.63")


def logit(p):
    p = np.clip(p / 100, 0.005, 0.995)
    return np.log(p / (1 - p))


def model():
    return make_pipeline(StandardScaler(), PCA(20), Ridge(alpha=10))


def fit_predict(trm, Xte):
    """Predict logit F at every sieve with variance (all sizes 0.063..2 mm), then curve."""
    cols = [COLS.index(c) for c in ("0.063", "0.2", "0.63", "2")]
    m = model().fit(X_all[trm], logit(Y[trm][:, cols]))
    P = 100 / (1 + np.exp(-m.predict(Xte)))
    out = np.zeros((len(Xte), len(COLS)))
    out[:, cols] = P
    out[:, COLS.index("6.3"):] = 100
    return make_monotone(out)


# ---- 1. domain check (PCA space fitted on SandSnap) ----
pre = make_pipeline(StandardScaler(), PCA(20)).fit(X_all[ss])
Z = pre.transform(X_all)
loso_nn = []
for s in np.unique(site[ss]):
    a, b = Z[ss & (site == s)], Z[ss & (site != s)]
    loso_nn.append(np.linalg.norm(a[:, None] - b[None], axis=2).min(1))
loso_nn = np.concatenate(loso_nn)
thr = np.percentile(loso_nn, 95)
print(f"SandSnap leave-site-out NN distance: median {np.median(loso_nn):.2f}, p95 {thr:.2f}")
te = d["src"] == "test"
tkeys = sorted(set(key[te]))
nn_te = {k: np.median(np.linalg.norm(Z[te & (key == k)][:, None] - Z[ss][None], axis=2).min(1)) for k in tkeys}

# ---- 2. leave-one-site-out CV ----
rows = []
for s in np.unique(site[ss]):
    tm = ss & (site == s)
    P = fit_predict(ss & ~tm, X_all[tm])
    for k in np.unique(key[tm]):
        m = key[tm] == k
        p = P[m].mean(0)
        t = curves.loc[k].values
        rows.append(dict(site=s, key=k, emd=emd(t, p)[0], err02=p[i02] - t[i02], err063=p[i063] - t[i063]))
cv = pd.DataFrame(rows)
base = curves.values.mean(0)
cv_base = np.mean([emd(curves.loc[k].values, base)[0] for k in cv.key])
print(f"leave-site-out CV on {len(cv)} sands: EMD {cv.emd.mean():.2f} (mean-curve baseline {cv_base:.2f}); "
      f"MAE F(0.2) {cv.err02.abs().mean():.1f} pts, F(0.63) {cv.err063.abs().mean():.1f} pts")

# ---- 3. test predictions ----
P = fit_predict(ss, X_all[te])
out = []
print(f"\n{'test sample':28s} {'NN dist':>7s} in-domain  F(0.063) F(0.2) F(0.63) F(2)")
for k in tkeys:
    p = P[key[te] == k].mean(0)
    ind = nn_te[k] <= thr
    out.append(dict(sample_id=k, nn_dist=nn_te[k], in_domain=ind, **dict(zip(COLS, p.round(2)))))
    print(f"{k:28s} {nn_te[k]:7.2f} {str(ind):9s}  {p[COLS.index('0.063')]:7.1f} {p[i02]:6.1f} {p[i063]:7.1f} {p[COLS.index('2')]:5.1f}")
pd.DataFrame(out).to_csv(f"{CACHE_DIR}/sand_preds_{VER}.csv", index=False)
print(f"saved cache/sand_preds_{VER}.csv")
