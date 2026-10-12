"""H18 evaluation: does a second embedding (SigLIP) / DINOv2 mid-layer help the sand model, and do SigLIP
concept axes track real factors?

A. SandSnap v2 sand model with feature sets {dino_last(+tex), dino_mid, siglip, concat, late fusion,
   CCA shared subspace}; protocols: leave-site-out and dry->wet (train on dry photos of samples that have
   no wet photos, test on wet photos of the others) = a systematic split across physical state.
B. Test sands: per-method F(0.2) and the spread across methods (embedding disagreement).
C. Concept axes: score = cos(img, text_pos) - cos(img, text_neg); Spearman vs labels on the 24
   competition samples; where SandSnap (dry/wet), competition train and each test sample fall.
usage: python src/embed_eval.py
"""
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CACHE_DIR, COLS, emd, load_labels, make_monotone
from multi_embed import CONCEPTS

M = np.load(f"{CACHE_DIR}/multi_embed.npz", allow_pickle=True)
S = np.load(f"{CACHE_DIR}/sand_feats_v2.npz", allow_pickle=True)
curves = pd.read_csv(f"{CACHE_DIR}/sandsnap_curves_v2.csv", index_col=0)
TARGET = [COLS.index(c) for c in ("0.063", "0.2", "0.63", "2")]
i02 = COLS.index("0.2")

# align sand-scale arrays by path
pos = {p: i for i, p in enumerate(S["path"])}
order = [pos[p] for p in M["sand_path"]]
src, key, dry, path = M["sand_src"], M["sand_key"], M["sand_dry"], M["sand_path"]
FEATS = {
    "dino_last": np.hstack([S["emb"][order], S["tex"][order]]),
    "dino_mid": M["sand_dino_mid"],
    "siglip": M["sand_siglip"],
}
FEATS["concat"] = np.hstack([FEATS["dino_last"], FEATS["siglip"]])
ss = src == "sandsnap"
te = src == "test"
site = np.array([(re.match(r"[A-Za-z]+", os.path.basename(p)).group(0) if "/tiles_" in p
                  else p.split("/images/")[1].split("/")[0]) if s else "" for p, s in zip(path, ss)])
Y = np.stack([curves.loc[k].values if s else np.full(len(COLS), np.nan) for k, s in zip(key, ss)])


def logit(p):
    p = np.clip(p / 100, 0.005, 0.995)
    return np.log(p / (1 - p))


def to_curve(P):
    out = np.zeros((len(P), len(COLS)))
    out[:, TARGET] = 100 / (1 + np.exp(-P))
    out[:, COLS.index("6.3"):] = 100
    return make_monotone(out)


def fit_predict(name, trm, tem):
    if name == "late":
        return np.mean([fit_predict(n, trm, tem) for n in ("dino_last", "siglip")], 0)
    if name == "late3":
        return np.mean([fit_predict(n, trm, tem) for n in ("dino_last", "dino_mid", "siglip")], 0)
    if name == "cca":
        a = make_pipeline(StandardScaler(), PCA(30)).fit(FEATS["dino_last"][trm])
        b = make_pipeline(StandardScaler(), PCA(30)).fit(FEATS["siglip"][trm])
        cca = CCA(10, max_iter=2000).fit(a.transform(FEATS["dino_last"][trm]), b.transform(FEATS["siglip"][trm]))
        za = lambda m: np.hstack(cca.transform(a.transform(FEATS["dino_last"][m]), b.transform(FEATS["siglip"][m])))
        r = Ridge(alpha=10).fit(za(trm), logit(Y[trm][:, TARGET]))
        return to_curve(r.predict(za(tem)))
    m = make_pipeline(StandardScaler(), PCA(20), Ridge(alpha=10)).fit(FEATS[name][trm], logit(Y[trm][:, TARGET]))
    return to_curve(m.predict(FEATS[name][tem]))


def score(name, protocol):
    rows = []
    if protocol == "site":
        splits = [(ss & (site != s), ss & (site == s)) for s in np.unique(site[ss])]
    else:  # dry -> wet: samples with any wet photo are held out entirely from training
        wet_keys = set(key[ss & (dry == 0)])
        splits = [(ss & (dry == 1) & ~np.isin(key, list(wet_keys)), ss & (dry == 0))]
    for trm, tem in splits:
        P = fit_predict(name, trm, tem)
        for k in np.unique(key[tem]):
            p = P[key[tem] == k].mean(0)
            t = curves.loc[k].values
            rows.append((emd(t, p)[0], abs(p[i02] - t[i02])))
    r = np.array(rows)
    return r[:, 0].mean(), r[:, 1].mean(), len(r)


METHODS = ["dino_last", "dino_mid", "siglip", "concat", "late", "late3", "cca"]
print("=== A. SandSnap sand model: embedding comparison ===")
print(f"{'method':10s} {'site EMD':>9s} {'site MAE F.2':>12s} {'dry->wet EMD':>13s} {'MAE F.2':>8s}")
res = {}
for m in METHODS:
    a, b, n1 = score(m, "site")
    c, e, n2 = score(m, "drywet")
    res[m] = (a, b, c, e)
    print(f"{m:10s} {a:9.2f} {b:12.1f} {c:13.2f} {e:8.1f}", flush=True)
print(f"(site: {n1} samples; dry->wet: {n2} wet samples)")

print("\n=== B. Test sands: F(0.2) per method, spread = disagreement ===")
tk = sorted(set(key[te]))
P = {m: fit_predict(m, ss, te) for m in METHODS}
print(f"{'sample':28s} " + " ".join(f"{m:>9s}" for m in METHODS) + "   spread(dl,sg,dm)")
for k in tk:
    v = {m: P[m][key[te] == k].mean(0)[i02] for m in METHODS}
    print(f"{k:28s} " + " ".join(f"{v[m]:9.1f}" for m in METHODS)
          + f"   {np.std([v['dino_last'], v['siglip'], v['dino_mid']]):6.1f}")
pd.DataFrame([dict(sample_id=k, method=m, **dict(zip(COLS, P[m][key[te] == k].mean(0).round(2))))
              for k in tk for m in METHODS]).to_csv(f"{CACHE_DIR}/sand_preds_multi.csv", index=False)

print("\n=== C. SigLIP concept axes ===")
Lab = load_labels()
cs, csamp = M["comp_siglip"], M["comp_sample"]
ctr = M["comp_split"] == "train"


def axis_score(X, name):
    return X @ M[f"text_{name}_pos"] - X @ M[f"text_{name}_neg"]


targets = {"fines%<0.063": Lab["0.063"], "%<0.2": Lab["0.2"], "gravel%>2": 100 - Lab["2"]}
print("Spearman of per-sample axis score vs labels (24 competition samples):")
for name in CONCEPTS:
    sc = pd.Series(axis_score(cs[ctr], name)).groupby(csamp[ctr]).mean()
    print(f"  {name:20s} " + "  ".join(f"{t}: {spearmanr(sc.loc[Lab.index], v)[0]:+.2f}" for t, v in targets.items()))
print("\nmean axis score by group (sand scale: SandSnap; comp scale: competition train / test samples):")
for name in CONCEPTS:
    line = f"  {name:20s} SS-dry {axis_score(M['sand_siglip'][ss & (dry == 1)], name).mean():+.3f}  SS-wet {axis_score(M['sand_siglip'][ss & (dry == 0)], name).mean():+.3f}  comp-train {axis_score(cs[ctr], name).mean():+.3f} |"
    for k in sorted(set(csamp[~ctr])):
        line += f" {k.replace('HPC_', '')[:10]} {axis_score(cs[~ctr & (csamp == k)], name).mean():+.3f}"
    print(line)
