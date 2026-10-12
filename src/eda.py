"""Deeper EDA: labels, near-duplicates, per-camera image quality, feature-space shift, feature/target
correlations, CV error decomposition, and same-scale galleries. Figures -> figures/eda/*.png"""
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import CACHE_DIR, COLS, DIAM, FIG_DIR, LOGD, emd, list_images, load_labels, camera_of
from features import TILE_PX, load_scaled, valid_tiles

Y = load_labels()
ids = list(Y.index)
V = Y.values
out = {}

# ---------------------------------------------------------------- 1. labels
gravel = 100 - Y["2"]
sand = Y["2"] - Y["0.063"]
silt = Y["0.063"] - Y["0.002"]
clay = Y["0.002"]


def d_at(row, p):  # diameter where the curve reaches p % (log-linear interpolation)
    return 10 ** np.interp(p, row, LOGD) if row.max() >= p else np.nan


d10 = np.array([d_at(r, 10) for r in V])
d50 = np.array([d_at(r, 50) for r in V])
d60 = np.array([d_at(r, 60) for r in V])
print("=== 1. LABELS (per training sample) ===")
print(f"{'id':6s} {'gravel':>6s} {'sand':>6s} {'silt':>6s} {'clay':>6s} {'D50mm':>7s} {'Cu':>6s}")
for i, s in enumerate(ids):
    cu = d60[i] / d10[i] if d10[i] > 0 else np.nan
    print(f"{s:6s} {gravel[s]:6.1f} {sand[s]:6.1f} {silt[s]:6.1f} {clay[s]:6.1f} {d50[i]:7.3f} {cu:6.1f}")

# pairwise label EMD -> near-duplicate groups
P = np.array([[emd(V[i], V[j])[0] for j in range(len(ids))] for i in range(len(ids))])
Z = linkage(squareform(P, checks=False), "average")
grp = fcluster(Z, t=15, criterion="distance")
print("\nnear-duplicate label groups (avg-link EMD < 15):")
for g in sorted(set(grp)):
    members = [ids[i] for i in np.where(grp == g)[0]]
    if len(members) > 1:
        sub = P[np.ix_(grp == g, grp == g)]
        print(f"  group {g}: {members}  max within-group EMD {sub.max():.1f}")
nn = np.sort(P + np.eye(len(ids)) * 1e9, 1)[:, 0]
print(f"label nearest-neighbour EMD: median {np.median(nn):.1f}; samples with NN<10: "
      f"{[ids[i] for i in np.where(nn < 10)[0]]}")
np.save(f"{CACHE_DIR}/eda_label_groups.npy", np.array([ids, grp], dtype=object))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
cmap = plt.get_cmap("tab10")
for i, s in enumerate(ids):
    ax[0].plot(DIAM, V[i], "-o", ms=3, color=cmap(grp[i] % 10), alpha=0.8)
    ax[0].text(DIAM[6] * 1.05, V[i, 6], s, fontsize=6, color=cmap(grp[i] % 10))
ax[0].set_xscale("log"); ax[0].set_xlabel("diameter (mm)"); ax[0].set_ylabel("% finer")
ax[0].set_title("training curves, coloured by label group"); ax[0].grid(alpha=0.3)
im = ax[1].imshow(P[np.ix_(np.argsort(grp), np.argsort(grp))], cmap="viridis")
ax[1].set_xticks(range(len(ids))); ax[1].set_xticklabels(np.array(ids)[np.argsort(grp)], rotation=90, fontsize=7)
ax[1].set_yticks(range(len(ids))); ax[1].set_yticklabels(np.array(ids)[np.argsort(grp)], fontsize=7)
ax[1].set_title("pairwise label EMD (sorted by group)"); plt.colorbar(im, ax=ax[1])
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eda/eda_labels.png", dpi=110); plt.close()

# ---------------------------------------------------------------- 2. image quality by camera
print("\n=== 2. IMAGE QUALITY at 4.6 px/mm (soil tiles only) ===")
tr, te = list_images()
rows = []
for f in tr + te:
    img = load_scaled(f)
    t = valid_tiles(img)[:12]
    crops = [img[y:y + TILE_PX, x:x + TILE_PX] for y, x in t]
    q = []
    for c in crops:
        g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lab = cv2.cvtColor(c, cv2.COLOR_RGB2LAB).astype(np.float32)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        # JPEG blockiness: mean abs jump at 8-px boundaries vs elsewhere
        dx = np.abs(np.diff(g, axis=1))
        block = dx[:, 7::8].mean() / (dx.mean() + 1e-6)
        # noise: MAD of residual after 3x3 median
        noise = np.median(np.abs(g - cv2.medianBlur(g.astype(np.uint8), 3)))
        q.append([lap.var() / (g.mean() ** 2 + 1e-6) * 1e3, noise, block, lab[..., 0].mean(),
                  lab[..., 1].mean() - 128, lab[..., 2].mean() - 128, g.std() / g.mean()])
    rows.append((camera_of(f), "test" if f in te else "train", np.mean(q, 0)))
names = ["sharp(LapVar)", "noise", "blockiness", "L*", "a*", "b*", "contrast"]
cams_q = sorted(set(r[0] for r in rows))
print(f"{'camera':26s} {'n':>3s} " + " ".join(f"{n:>13s}" for n in names))
for c in cams_q:
    m = np.array([r[2] for r in rows if r[0] == c])
    print(f"{c:26s} {len(m):3d} " + " ".join(f"{v:13.2f}" for v in m.mean(0)))
Q = np.array([r[2] for r in rows])
qcam = np.array([r[0] for r in rows])
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for k, j in enumerate([0, 2, 4, 5]):
    ax[k].boxplot([Q[qcam == c, j] for c in cams_q], tick_labels=[c.replace("_", "\n") for c in cams_q])
    ax[k].set_title(names[j]); ax[k].tick_params(labelsize=7)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eda/eda_quality.png", dpi=110); plt.close()

# ---------------------------------------------------------------- 3. feature-space shift
print("\n=== 3. FEATURE SPACE (image level) ===")
d = np.load(f"{CACHE_DIR}/feats_dinov2base.npz", allow_pickle=True)
cam = np.where(d["cam"] == "Motorola_Edge_60_fusion", "Motorola_Edge", d["cam"])
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for k, key in enumerate(["emb", "hand"]):
    X = StandardScaler().fit(d[key][d["split"] == "train"]).transform(d[key])
    pca = PCA(2).fit(X[d["split"] == "train"])
    Zp = pca.transform(X)
    for c, mk in zip(sorted(set(cam)), "os^vD"):
        m = cam == c
        ax[k].scatter(Zp[m, 0], Zp[m, 1], marker=mk, s=28, label=c, alpha=0.75)
    for s in sorted(set(d["sample"][d["split"] == "test"])):
        m = d["sample"] == s
        ax[k].annotate(s.replace("HPC_", "")[:14], Zp[m].mean(0), fontsize=6)
    ax[k].set_title(f"{key}: PCA fit on train (var {pca.explained_variance_ratio_.sum():.0%})")
    ax[k].legend(fontsize=7)
    # how separable are cameras? (between-camera vs between-sample variance on train)
    trm = d["split"] == "train"
    print(f"{key}: train PC1 corr with camera(Samsung=1): "
          f"{np.corrcoef(Zp[trm, 0], cam[trm] == 'Samsung_A52')[0, 1]:+.2f}, with fines%: "
          f"{np.corrcoef(Zp[trm, 0], Y.loc[d['sample'][trm], '0.063'])[0, 1]:+.2f}")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/eda/eda_featurespace.png", dpi=110); plt.close()

# ---------------------------------------------------------------- 4. feature/target correlations
print("\n=== 4. HAND FEATURES vs TARGETS (sample-level Spearman) ===")
from scipy.stats import spearmanr

hand_names = ([f"logE{s}" for s in [0.25, 0.5, 1, 2, 4, 8, 16, 32]] + [f"logV{s}" for s in [0.25, 0.5, 1, 2, 4, 8, 16, 32]]
              + [f"p{s}" for s in [0.25, 0.5, 1, 2, 4, 8, 16, 32]] + ["gq50", "gq75", "gq90", "gq97", "sat", "vcontrast", "gstd"])
trm = d["split"] == "train"
Hs = np.stack([d["hand"][trm & (d["sample"] == s)].mean(0) for s in ids])
targets = {"%<0.002": Y["0.002"], "%<0.063": Y["0.063"], "%<0.63": Y["0.63"], "%<2": Y["2"], "%<20": Y["20"]}
C = np.array([[spearmanr(Hs[:, j], t)[0] for t in targets.values()] for j in range(Hs.shape[1])])
best = np.argsort(-np.abs(C).max(1))[:12]
print(f"{'feature':10s} " + " ".join(f"{k:>8s}" for k in targets))
for j in best:
    print(f"{hand_names[j]:10s} " + " ".join(f"{v:+8.2f}" for v in C[j]))
# within-sample camera consistency of hand features (ICC-like): between-sample var / total var
both = [s for s in ids if len(set(cam[trm & (d["sample"] == s)])) == 2]
print("\nfeature reproducibility across the 2 cameras (corr of per-sample means, Samsung vs Motorola):")
for j in best[:8]:
    a = [d["hand"][trm & (d["sample"] == s) & (cam == "Samsung_A52"), j].mean() for s in both]
    b = [d["hand"][trm & (d["sample"] == s) & (cam == "Motorola_Edge"), j].mean() for s in both]
    print(f"  {hand_names[j]:10s} r={np.corrcoef(a, b)[0, 1]:+.2f}  mean shift Sam-Mot={np.mean(a) - np.mean(b):+.3f}")

# ---------------------------------------------------------------- 5. CV error decomposition (grouped vs LOSO)
print("\n=== 5. CV ERROR: LOSO vs GROUP-out (holding out whole near-duplicate groups) ===")
from model import FEATURE_SETS, Yimg, fit_predict, regressors, samples
from common import make_monotone

regs = regressors()
ENS = [("hand+emb", "pca5_ridge"), ("emb", "pls2"), ("hand", "knn5"), ("hand+emb", "knn5")]
gmap = dict(zip(ids, grp))
w = np.diff(LOGD)
res = {}
for mode in ("loso", "group"):
    per, contrib = {}, []
    for s in ids:
        tm = samples == s
        trm_ = ~np.isin(samples, [x for x in ids if gmap[x] == gmap[s]]) if mode == "group" else ~tm
        p = make_monotone(np.mean([fit_predict(regs[r], FEATURE_SETS[f][trm_], Yimg[trm_], FEATURE_SETS[f][tm]).mean(0)
                                   for f, r in ENS], 0))[0]
        per[s] = emd(Y.loc[s].values, p)[0]
        e = np.abs(Y.loc[s].values - p)
        contrib.append((e[:-1] + e[1:]) / 2 * w)
    res[mode] = per
    print(f"{mode:6s} mean EMD {np.mean(list(per.values())):.2f}")
    if mode == "group":
        cm = np.mean(contrib, 0)
        print("  mean EMD contribution per interval:",
              {f"{COLS[i]}-{COLS[i + 1]}": round(v, 1) for i, v in enumerate(cm)})
print(f"{'sample':6s} {'loso':>6s} {'group':>6s}")
for s in ids:
    print(f"{s:6s} {res['loso'][s]:6.1f} {res['group'][s]:6.1f}")

# ---------------------------------------------------------------- 6. galleries at the same scale
def gallery(items, fname, ncol=8, S=220):
    nrow = int(np.ceil(len(items) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 2.45))
    for a in ax.ravel():
        a.axis("off")
    for a, (title, f) in zip(ax.ravel(), items):
        img = load_scaled(f)
        t = valid_tiles(img)
        y, x = t[len(t) // 2]
        a.imshow(cv2.resize(img[y:y + TILE_PX, x:x + TILE_PX], (S, S)))
        a.set_title(title, fontsize=7)
    plt.tight_layout(); plt.savefig(fname, dpi=100); plt.close()


order = np.argsort(Y["0.063"].values)
items = []
for i in order:
    s = ids[i]
    f = next(f for f in tr if f"_{s}_" in f and "Samsung" in f) if any(f"_{s}_" in f and "Samsung" in f for f in tr) \
        else next(f for f in tr if f"_{s}_" in f)
    items.append((f"{s} fines{Y.loc[s, '0.063']:.0f}% grav{gravel[s]:.0f}%", f))
gallery(items, f"{FIG_DIR}/eda/eda_gallery_train.png")
from common import test_name_to_id

seen, titems = set(), []
for f in te:
    s = test_name_to_id(f)
    if s not in seen:
        seen.add(s); titems.append((s.replace("HPC_", "")[:22], f))
gallery(titems, f"{FIG_DIR}/eda/eda_gallery_test.png", ncol=5)
print("\nfigures written to figures/eda/")
