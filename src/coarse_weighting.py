"""H9: weight measured grains by volume, not just visible area, for the split of the gravel fraction.

Current physics (predict2, linear w=1) calibrates each coarse sieve independently from the area % of
grains > d. Large stones protrude from the surface, so their mass share exceeds their area share.
Here the total gravel C(>2 mm) keeps the existing calibration; only its split over sizes changes:

    ratio_k(d) = Σ_{sieve>d} area·sieve^k / Σ_{sieve>2mm} area·sieve^k      (k=0 area, k=1 ≈ volume)
    C(>d) = C(>2) · ratio_k(d)                                              for d = 6.3, 20, 63 mm

k is chosen on the gravelly training samples (true C(>2) ≥ 10 %) by leave-one-label-group-out, scoring
the true ratio C(>d)/C(>2) at 6.3/20/63 mm. Uses per-grain SAM output (cache/grains_v2_shard*.npz).
usage: python src/coarse_weighting.py
"""
import glob
import os

import numpy as np

from common import CACHE_DIR, COLS, label_groups, load_labels

KS = [0.0, 0.5, 1.0, 1.5, 2.0]
SIEVES = {"6.3": 6.3, "20": 20.0, "63": 63.0}


def load_v2():
    rec = {}
    for f in sorted(glob.glob(f"{CACHE_DIR}/grains_v2_shard*.npz")):
        z = np.load(f, allow_pickle=True)
        for i, p in enumerate(z["path"]):
            rec[os.path.basename(p)] = dict(sample=z["sample"][i], split=z["split"][i], grains=z["grains"][i])
    return rec


def ratios(grains, k):
    g = grains[grains[:, 0] > 2.0] if len(grains) else grains
    if len(g) == 0:
        return {c: 0.0 for c in SIEVES}
    w = g[:, 2] * g[:, 0] ** k
    return {c: w[g[:, 0] > d].sum() / w.sum() for c, d in SIEVES.items()}


def sample_ratios(rec, sample, k):
    """Median over the sample's photos."""
    rs = [ratios(r["grains"], k) for r in rec.values() if r["sample"] == sample]
    return {c: float(np.median([r[c] for r in rs])) for c in SIEVES}


if __name__ == "__main__":
    rec = load_v2()
    Y = load_labels()
    groups = label_groups()
    grav = [s for s in Y.index if 100 - Y.loc[s, "2"] >= 10]
    true = {s: {c: (100 - Y.loc[s, c]) / (100 - Y.loc[s, "2"]) for c in SIEVES} for s in grav}
    print(f"{len(grav)} gravelly training samples; true ratio C(>d)/C(>2) ranges:",
          {c: (round(min(t[c] for t in true.values()), 2), round(max(t[c] for t in true.values()), 2)) for c in SIEVES})
    pred = {k: {s: sample_ratios(rec, s, k) for s in grav} for k in KS}
    print(f"\n{'k':>4s} " + " ".join(f"MAE {c:>4s}mm" for c in SIEVES) + "   mean")
    mae = {}
    for k in KS:
        e = {c: np.mean([abs(pred[k][s][c] - true[s][c]) for s in grav]) for c in SIEVES}
        mae[k] = np.mean(list(e.values()))
        print(f"{k:4.1f} " + " ".join(f"{100 * e[c]:11.1f}" for c in SIEVES) + f"   {100 * mae[k]:.1f}")
    # leave-one-group-out choice of k (honest estimate of the selection)
    lo = []
    for s in grav:
        keep = [x for x in grav if groups[x] != groups[s]]
        kb = min(KS, key=lambda k: np.mean([abs(pred[k][x][c] - true[x][c]) for x in keep for c in SIEVES]))
        lo.append(np.mean([abs(pred[kb][s][c] - true[s][c]) for c in SIEVES]))
    print(f"\nleave-one-group-out selected-k MAE: {100 * np.mean(lo):.1f} pts (k=0 area-only: {100 * mae[0.0]:.1f})")
    print("\ntest samples (ratio of gravel coarser than d), k=0 vs k=1:")
    for s in sorted({r["sample"] for r in rec.values() if r["split"] == "test"}):
        r0, r1 = sample_ratios(rec, s, 0.0), sample_ratios(rec, s, 1.0)
        print(f"  {s:28s} " + "  ".join(f">{c}: {r0[c]:.2f}->{r1[c]:.2f}" for c in SIEVES))
