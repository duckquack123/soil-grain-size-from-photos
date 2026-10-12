"""How well do SAM grain-area fractions track the true coarse fractions? (per training sample)"""
import glob
import os

import numpy as np

from common import CACHE_DIR, COLS, load_labels
from sam_grains import SIZES_MM


def load_grains(paths):
    """Grain features aligned to `paths` (order of the feature npz); matched by file name so caches
    stay valid when the data folder moves."""
    shards = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{CACHE_DIR}/grains_shard*.npz"))]
    by_path = {}
    for s in shards:
        for i, p in enumerate(s["path"]):
            by_path[os.path.basename(p)] = {k: s[k][i] for k in ("frac", "frac_eq", "ngrains", "grain_area")}
    paths = [os.path.basename(p) for p in paths]
    missing = [p for p in paths if p not in by_path]
    assert not missing, f"{len(missing)} images missing grain features, e.g. {missing[:2]}"
    return {k: np.stack([by_path[p][k] for p in paths]) for k in ("frac", "frac_eq", "ngrains", "grain_area")}


if __name__ == "__main__":
    d = np.load(f"{CACHE_DIR}/feats_dinov2base.npz", allow_pickle=True)
    g = load_grains(d["path"])
    Y = load_labels()
    tr = d["split"] == "train"
    smp = d["sample"]
    print("per-sample: true % coarser than d  vs  SAM area % of grains with sieve size > d")
    for dcol, dmm in (("2", 2), ("6.3", 6.3), ("20", 20), ("63", 63)):
        j = int(np.argmin(np.abs(SIZES_MM - dmm)))
        true = np.array([100 - Y.loc[s, dcol] for s in sorted(set(smp[tr]))])
        meas = np.array([100 * g["frac"][tr & (smp == s), j].mean() for s in sorted(set(smp[tr]))])
        r = np.corrcoef(true, meas)[0, 1]
        slope = (true @ meas) / (meas @ meas + 1e-9)
        print(f"  d={dmm:>4} mm  r={r:5.2f}  true≈{slope:4.2f}*meas   true range {true.min():.0f}-{true.max():.0f}%  "
              f"meas range {meas.min():.0f}-{meas.max():.0f}%")
    print("\ntest samples: SAM area % coarser than", SIZES_MM.tolist())
    for s in sorted(set(smp[~tr])):
        print(f"  {s:28s}", np.round(100 * g["frac"][(~tr) & (smp == s)].mean(0), 1).tolist())
    print("\ntrain samples:")
    for s in sorted(set(smp[tr])):
        m = tr & (smp == s)
        print(f"  {s:6s} true%coarser(2,6.3,20,63)={[round(100 - Y.loc[s, c], 1) for c in ('2', '6.3', '20', '63')]}"
              f"  SAM={np.round(100 * g['frac'][m].mean(0)[[0, 3, 6, 9]], 1).tolist()}")
