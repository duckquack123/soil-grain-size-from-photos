"""Convert SandSnap percentile labels (d10..d90, mm) to our 11-point cumulative curves.

Inside [d10, d90]: log-linear interpolation through the 8 percentiles (exact data).
Outside: log-normal fitted to the 8 percentiles (tails; for beach sands d10 >= 0.086 mm, so the
fines < 0.063 mm come out ~0-5 % — an assumption carried by the data, not a measurement).
Physical samples = unique percentile sets (images of the same sieved sample share labels).
Writes <SANDSNAP>/sandsnap_index.csv (adds sample_key) and cache/sandsnap_curves.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import CACHE_DIR, COLS, DIAM, make_monotone
from sand_features import SANDSNAP, VER

P = np.array([10, 16, 25, 50, 65, 75, 84, 90])
DCOLS = ["d10", "d16", "d25", "d50", "d65", "d75", "d84", "d90"]


def curve(dvals):
    ld = np.log10(dvals)
    z = norm.ppf(P / 100)
    sigma, mu = np.polyfit(z, ld, 1)                  # log10 d = mu + sigma * z
    out = norm.cdf((np.log10(DIAM) - mu) / sigma) * 100
    inside = (DIAM >= dvals[0]) & (DIAM <= dvals[-1])
    out[inside] = np.interp(np.log10(DIAM[inside]), ld, P)
    return make_monotone(out)[0]


if __name__ == "__main__":
    idx = pd.read_csv(f"{SANDSNAP}/sandsnap_index.csv")
    keys = idx[DCOLS].round(4).astype(str).agg("|".join, axis=1)
    idx["sample_key"] = ["SS%03d" % i for i in pd.factorize(keys)[0]]
    idx.to_csv(f"{SANDSNAP}/sandsnap_index.csv", index=False)
    smp = idx.drop_duplicates("sample_key")
    C = np.stack([curve(np.sort(r[DCOLS].values.astype(float))) for _, r in smp.iterrows()])
    df = pd.DataFrame(C, columns=COLS, index=smp.sample_key).round(3)
    df.to_csv(f"{CACHE_DIR}/sandsnap_curves_{VER}.csv")
    print(f"{len(idx)} images, {len(df)} samples, images/sample median {idx.sample_key.value_counts().median():.0f}")
    print(df.describe().loc[["min", "50%", "max"]].round(1))
