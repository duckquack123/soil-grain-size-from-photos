"""Shared helpers: project paths, metric, data loading, ppm handling."""
import glob
import os
import re

import numpy as np
import pandas as pd
from PIL import Image

# ---- project layout (see README.md) ----
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = f"{ROOT}/data"            # raw competition files (read-only)
CACHE_DIR = f"{ROOT}/cache"          # feature caches (*.npz)
FIG_DIR = f"{ROOT}/figures"          # figures/eda, figures/debug
LOG_DIR = f"{ROOT}/logs"             # job logs
SUB_DIR = f"{ROOT}/submissions"      # submission csvs
RES_DIR = f"{ROOT}/results"          # experiments.csv etc.
TRAIN_DIR = f"{DATA_DIR}/Training-All_Photos_updated/Training-All_Photos_updated"
TEST_DIR = f"{DATA_DIR}/Test_All_Photos/Test_All_Photos"


def in_dir(directory, name):
    """Resolve a bare file name into `directory`; absolute/relative paths with a folder are kept."""
    return name if os.path.isabs(name) or os.path.dirname(name) else os.path.join(directory, name)
COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
DIAM = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
LOGD = np.log10(DIAM)

# native resolution and ppm per camera (ppm_updated.csv); ppm refers to native width
CAMERAS = {
    "iPhone14": (4032, 13.942),
    "iPhone16": (5712, 19.525),
    "Motorola_Edge_60_fusion": (4096, 12.465),
    "Motorola_Edge": (4000, 11.492),
    "Samsung_A52": (9248, 26.33),
}


def emd(true, pred):
    """Log-weighted EMD (area between cumulative curves on log10 axis); lower is better.

    Matches the competition definition: sum over the 10 intervals of the mean abs error
    at the interval ends times the log10 width of the interval.
    """
    true, pred = np.atleast_2d(true), np.atleast_2d(pred)
    err = np.abs(true - pred)
    w = np.diff(LOGD)
    return ((err[:, :-1] + err[:, 1:]) / 2 * w).sum(1)


def make_monotone(p):
    p = np.clip(np.atleast_2d(p).astype(float), 0, 100)
    p = np.maximum.accumulate(p, axis=1)
    p[:, -1] = 100.0
    return p


def camera_of(fname):
    b = os.path.basename(fname)
    for key in ("Motorola_Edge_60_fusion", "Motorola_Edge", "Samsung_A52"):
        if b.startswith(key):
            return key
    if b.startswith("iPhone14"):
        return "iPhone14"
    if b.startswith("iPhone16") or b.startswith("iPhone_16"):
        return "iPhone16"
    raise ValueError(b)


def effective_ppm(path):
    """px/mm of the file actually on disk (training images were downscaled).

    ppm_updated.csv gives px/mm at native resolution; scale by the long-side ratio so the
    result is independent of EXIF/portrait-vs-landscape orientation.
    """
    native_long, ppm = CAMERAS[camera_of(path)]
    return ppm * max(Image.open(path).size) / native_long


def sample_id_train(fname):
    m = re.search(r"_([FGH]\d{3})_\d+\.jpg$", os.path.basename(fname))
    return m.group(1)


def test_name_to_id(fname):
    """Map photo file name to the sample_submission id (e.g. Münster -> Muenster)."""
    b = os.path.basename(fname)
    b = re.sub(r"^(iPhone14|iPhone16|iPhone_16)_", "", b)
    b = re.sub(r"(\s*\(\d+\))?\.JPG$", "", b)
    b = b.replace("Münster_BS6_9,0-10m", "Muenster_BS6_9_0-10m")
    return b


def load_labels():
    df = pd.read_csv(f"{DATA_DIR}/Training_labels_updated.csv")
    return df.set_index("sample_id")[COLS]


def label_groups(threshold=15.0):
    """Near-duplicate sample groups (average-linkage on pairwise label EMD < threshold).

    EDA showed e.g. {H366,H367,H371,H372,H374}, {H368,H615}, {H181,H183} have near-identical curves
    (same site), so leave-one-sample-out CV leaks; hold out whole groups instead.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    Y = load_labels()
    V = Y.values
    P = np.array([[emd(a, b)[0] for b in V] for a in V])
    grp = fcluster(linkage(squareform(P, checks=False), "average"), t=threshold, criterion="distance")
    return dict(zip(Y.index, grp))


def list_images():
    tr = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))
    te = sorted(glob.glob(f"{TEST_DIR}/*.JPG"))
    return tr, te
