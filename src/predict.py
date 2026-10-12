"""Ensemble CV + test prediction + out-of-distribution diagnostics; writes submission.csv."""
import sys

import numpy as np
import pandas as pd

from common import COLS, DATA_DIR, SUB_DIR, emd, in_dir, make_monotone
from model import FEATURE_SETS, Y, Yimg, cams, d, fit_predict, regressors, samples, tr, uniq

ENSEMBLE = [("hand+emb", "pca5_ridge"), ("emb", "pls2"), ("hand", "knn5"), ("hand+emb", "knn5")]
te = ~tr
TEST_SETS = {
    "hand": d["hand"][te],
    "emb": d["emb"][te],
    "hand+emb": np.hstack([d["hand"][te], d["emb"][te]]),
}
regs = regressors()


def ens_predict(trm, X_sets, rows):
    return np.mean([fit_predict(regs[r], FEATURE_SETS[f][trm], Yimg[trm], X_sets[f][rows]).mean(0)
                    for f, r in ENSEMBLE], 0)


# --- CV of the ensemble (both protocols) ---
for protocol in ("loso", "xcam"):
    scores = []
    for s in uniq:
        tm = samples == s
        groups = [None] if protocol == "loso" else sorted(set(cams[tm]))
        for c in groups:
            trm = ~tm if c is None else (~tm) & (cams != c)
            rows = tm if c is None else tm & (cams == c)
            scores.append(emd(Y.loc[s].values, make_monotone(ens_predict(trm, FEATURE_SETS, rows)))[0])
    print(f"ensemble {protocol}: {np.mean(scores):.2f}  (median {np.median(scores):.2f}, max {np.max(scores):.1f})")

# --- fit on all training images, predict test samples ---
test_samples = d["sample"][te]
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
all_tr = np.ones(tr.sum(), bool)
preds = {}
for s in sub.sample_id:
    preds[s] = make_monotone(ens_predict(all_tr, TEST_SETS, test_samples == s))[0]

# --- OOD diagnostics: distance of each test sample to the nearest training sample (hand features) ---
H = FEATURE_SETS["hand"]
mu, sd = H.mean(0), H.std(0) + 1e-9
tr_s = np.stack([((H[samples == s] - mu) / sd).mean(0) for s in uniq])
nn_tr = np.sort(np.linalg.norm(tr_s[:, None] - tr_s[None], axis=2), 1)[:, 1]
print(f"\ntrain-to-train NN distance: median {np.median(nn_tr):.2f}, max {nn_tr.max():.2f}")
print(f"{'test sample':28s} {'NN train':8s} {'dist':>5s}  prediction (% finer at {', '.join(COLS)})")
for s in sub.sample_id:
    v = ((TEST_SETS["hand"][test_samples == s] - mu) / sd).mean(0)
    dist = np.linalg.norm(tr_s - v, axis=1)
    print(f"{s:28s} {uniq[dist.argmin()]:8s} {dist.min():5.2f}  {np.round(preds[s], 1).tolist()}")

out = in_dir(SUB_DIR, sys.argv[1] if len(sys.argv) > 1 else "submission.csv")
pd.DataFrame([[s, *np.round(preds[s], 4)] for s in sub.sample_id], columns=["sample_id", *COLS]).to_csv(out, index=False)
print("\nwrote", out)
