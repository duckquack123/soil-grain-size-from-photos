"""Leave-one-sample-out evaluation of feature sets x regressors on the competition metric.

Two protocols:
  loso      - hold out one sample (all its photos), train on the rest (all cameras)
  xcam      - additionally train only on the *other* camera's photos -> proxy for the
              unseen-camera (iPhone) shift of the test set
"""
import os
import sys

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CACHE_DIR, emd, in_dir, label_groups, load_labels, make_monotone

FEATS = (sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1
         else os.environ.get("FEATS", "feats_dinov2base.npz"))
d = np.load(in_dir(CACHE_DIR, FEATS), allow_pickle=True)
Y = load_labels()
tr = d["split"] == "train"
samples, cams = d["sample"][tr], d["cam"][tr]
cams = np.where(cams == "Motorola_Edge_60_fusion", "Motorola_Edge", cams)
Yimg = Y.loc[samples].values
uniq = np.array(sorted(set(samples)))

FEATURE_SETS = {
    "hand": d["hand"][tr],
    "emb": d["emb"][tr],
    "hand+emb": np.hstack([d["hand"][tr], d["emb"][tr]]),
}


def regressors():
    return {
        "ridge_a10": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "ridge_a100": lambda: make_pipeline(StandardScaler(), Ridge(alpha=100)),
        "ridge_a1000": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1000)),
        "pca5_ridge": lambda: make_pipeline(StandardScaler(), PCA(5), Ridge(alpha=1)),
        "pls2": lambda: make_pipeline(StandardScaler(), PLSRegression(2, scale=False)),
        "pls3": lambda: make_pipeline(StandardScaler(), PLSRegression(3, scale=False)),
        "knn3": "knn3",
        "knn5": "knn5",
    }


def fit_predict(make, Xtr, Ytr, Xte):
    if isinstance(make, str):  # cosine kNN on standardised features, mean of neighbour curves
        k = int(make[3:])
        sc = StandardScaler().fit(Xtr)
        a, b = sc.transform(Xtr), sc.transform(Xte)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        idx = np.argsort(-b @ a.T, axis=1)[:, :k]
        return Ytr[idx].mean(1)
    m = make().fit(Xtr, Ytr[:, :10])
    return np.hstack([m.predict(Xte), np.full((len(Xte), 1), 100.0)])


GROUPS = label_groups()


def folds(protocol):
    """Yield (held-out sample, train mask, test-row mask) for each CV protocol.

    loso   - leave one sample out
    xcam   - loso, but train only on the other camera's photos (camera-shift proxy)
    group  - leave the whole near-duplicate label group out (no same-site leakage)
    xgroup - group + other camera only (strictest)
    """
    for s in uniq:
        te_mask = samples == s
        out = np.isin(samples, [x for x in uniq if GROUPS[x] == GROUPS[s]]) if "group" in protocol else te_mask
        if protocol in ("loso", "group"):
            yield s, ~out, te_mask
        else:
            for c in sorted(set(cams[te_mask])):
                yield s, (~out) & (cams != c), te_mask & (cams == c)


def evaluate(X, make, protocol):
    scores = [emd(Y.loc[s].values, make_monotone(fit_predict(make, X[trm], Yimg[trm], X[rows]).mean(0)))[0]
              for s, trm, rows in folds(protocol)]
    return np.mean(scores)


if __name__ == "__main__":
    P = ("loso", "xcam", "group", "xgroup")
    print(f"{'features':10s} {'model':12s} " + " ".join(f"{p:>7s}" for p in P))
    for fname, X in FEATURE_SETS.items():
        for rname, make in regressors().items():
            print(f"{fname:10s} {rname:12s} " + " ".join(f"{evaluate(X, make, p):7.2f}" for p in P), flush=True)
