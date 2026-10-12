"""Two-stage model: SAM-calibrated coarse fractions + learned (DINOv2/texture) fine-curve shape.

  coarse%(d) for d in {2, 6.3, 20} mm  = linear calibration of SAM grain-area % (fit on training photos)
  coarse%(63)                          = slope of the 20 mm calibration x SAM area % > 63 mm
                                         (no training sample has measurable grains > 63 mm)
  F(d) for d < 2 mm                    = ensemble curve shape, rescaled to the new F(2 mm)

Evaluated with leave-one-sample-out (loso) and cross-camera loso (xcam); writes submission.
usage: python predict2.py [OUT.csv [FORM W]]   (default: form/weight with best mean CV)
"""
import sys

import numpy as np
import pandas as pd

from common import COLS, DATA_DIR, SUB_DIR, emd, in_dir, make_monotone
from grains_eval import load_grains
from model import FEATURE_SETS, Y, Yimg, cams, d, fit_predict, regressors, samples, uniq
from sam_grains import SIZES_MM

ENSEMBLE = [("hand+emb", "pca5_ridge"), ("emb", "pls2"), ("hand", "knn5"), ("hand+emb", "knn5")]
CAL = {"2": 2.0, "6.3": 6.3, "20": 20.0}  # sieve column -> SAM size used for calibration
regs = regressors()
tr = d["split"] == "train"
te = ~tr
G = 100 * load_grains(d["path"])["frac"]  # (n_img, len(SIZES_MM)) area % coarser than size
Gtr, Gte = G[tr], G[te]
TEST_SETS = {"hand": d["hand"][te], "emb": d["emb"][te], "hand+emb": np.hstack([d["hand"][te], d["emb"][te]])}
col = {c: i for i, c in enumerate(COLS)}
size_idx = {c: int(np.argmin(np.abs(SIZES_MM - mm))) for c, mm in CAL.items()}
j63 = int(np.argmin(np.abs(SIZES_MM - 63)))


def ensemble(trm, X_sets, rows):
    return np.mean([fit_predict(regs[r], FEATURE_SETS[f][trm], Yimg[trm], X_sets[f][rows]).mean(0)
                    for f, r in ENSEMBLE], 0)


def logit(p):
    p = np.clip(np.asarray(p, float) / 100, 0.01, 0.99)
    return np.log(p / (1 - p))


def calibrate(trm, form="linear"):
    """Per sieve, fit true coarse% from SAM area% over training photos.

    linear: coarse = a + b*x with a <= 0 (SAM false positives - clods - only inflate the
            measured area, so a positive offset is unphysical; it would cap sandy samples)
    logit : logit(coarse) = a + b*logit(x) - bounded, saturates instead of exceeding 100 %
    The 'slope' entry is the through-origin slope, reused for the 63 mm point.
    """
    out = {}
    for c, j in size_idx.items():
        x, y = Gtr[trm, j], 100 - Yimg[trm, col[c]]
        if form == "logit":
            b, a = np.polyfit(logit(x), logit(y), 1)
        else:
            b, a = np.polyfit(x, y, 1)
            if a > 0:
                a, b = 0.0, (x @ y) / (x @ x + 1e-9)
        out[c] = dict(a=a, b=max(b, 0.0), slope=(x @ y) / (x @ x + 1e-9), form=form)
    return out


def apply_cal(p, x):
    if p["form"] == "logit":
        return 100 / (1 + np.exp(-(p["a"] + p["b"] * logit(x))))
    return np.clip(p["a"] + p["b"] * x, 0, 100)


def two_stage(ens, g_rows, cal, w=1.0, agg=np.median):
    """Replace coarse part of the ensemble curve by the calibrated SAM estimate (blend weight w)."""
    g = agg(g_rows, 0)
    F = ens.copy()
    for c, j in size_idx.items():
        F[col[c]] = (1 - w) * F[col[c]] + w * (100 - apply_cal(cal[c], g[j]))
    F[col["63"]] = (1 - w) * F[col["63"]] + w * (100 - np.clip(cal["20"]["slope"] * g[j63], 0, 100))
    F = make_monotone(np.minimum.accumulate(F[::-1])[::-1])[0]  # coarse points bound finer ones
    # rescale the learned fines shape (d < 2 mm) to the new passing-2 mm value
    k2 = col["2"]
    if ens[k2] > 1e-6:
        F[:k2] = ens[:k2] * F[k2] / ens[k2]
    return make_monotone(F)[0]


_ens_cache = {}


def cv(protocol, form="linear", **kw):
    scores = []
    for s in uniq:
        tm = samples == s
        groups = [None] if protocol == "loso" else sorted(set(cams[tm]))
        for c in groups:
            trm = ~tm if c is None else (~tm) & (cams != c)
            rows = tm if c is None else tm & (cams == c)
            key = (s, c)
            if key not in _ens_cache:
                _ens_cache[key] = make_monotone(ensemble(trm, FEATURE_SETS, rows))[0]
            ens = _ens_cache[key]
            p = ens if kw.get("w", 1.0) == 0 else two_stage(ens, Gtr[rows], calibrate(trm, form), **kw)
            scores.append(emd(Y.loc[s].values, p)[0])
    return np.mean(scores)


if __name__ == "__main__":
    print(f"{'variant':34s} {'loso':>7s} {'xcam':>7s} {'avg':>7s}")
    results = {}
    for form in ("linear", "logit"):
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            if w == 0 and form == "logit":
                continue
            lo, xc = cv("loso", form, w=w), cv("xcam", form, w=w)
            results[(form, w)] = (lo + xc) / 2
            name = "ensemble only" if w == 0 else f"two-stage {form} w={w}"
            print(f"{name:34s} {lo:7.2f} {xc:7.2f} {(lo + xc) / 2:7.2f}", flush=True)
    FORM, W = min(results, key=results.get)
    if len(sys.argv) > 2:  # explicit override: predict2.py OUT.csv FORM W
        FORM, W = sys.argv[2], float(sys.argv[3])
    print(f"\nusing form={FORM} w={W}")
    all_tr = np.ones(tr.sum(), bool)
    cal = calibrate(all_tr, FORM)
    print("calibration:", {c: (round(p["a"], 2), round(p["b"], 2)) for c, p in cal.items()},
          "63mm slope:", round(cal["20"]["slope"], 2))
    test_samples = d["sample"][te]
    sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    rows_out = []
    print(f"\n{'test sample':28s} ensemble -> two-stage (% finer at {', '.join(COLS)})")
    for s in sub.sample_id:
        m = test_samples == s
        ens = make_monotone(ensemble(all_tr, TEST_SETS, m))[0]
        p = two_stage(ens, Gte[m], cal, w=W)
        rows_out.append([s, *np.round(p, 4)])
        print(f"{s:28s} {np.round(ens, 1).tolist()}\n{'':28s} {np.round(p, 1).tolist()}")
    out = in_dir(SUB_DIR, sys.argv[1] if len(sys.argv) > 1 else "submission_two_stage.csv")
    pd.DataFrame(rows_out, columns=["sample_id", *COLS]).to_csv(out, index=False)
    print("\nwrote", out)
