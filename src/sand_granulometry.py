"""H15: measure sand grading physically at native resolution (morphological granulometry).

The 4.6 px/mm pipeline cannot resolve sand (EDA: test sands are clean loose sands, but every learned
model predicts 35-50 % fines because no training photo shows loose sand). Test iPhone 16 photos are
19.5 px/mm (0.05 mm/px), so grains >= ~0.15 mm are 3+ px wide.

Granulometry by opening: opening an image with a disk of diameter D removes bright structures narrower
than D. The bright "volume" removed between successive D is the amount of grain with width (≈ sieve
size, the intermediate dimension seen from above) in that range. Two variants:
  gray   - grayscale volume above the 3 mm opening (background), illumination-flattened
  binary - area of an adaptive-threshold grain mask (pure area weighting, Delesse-like)
Cumulative G(d) = fraction of measured grain material with width < d, for d up to 2 mm (coarser
grains are handled by the SAM pipeline). Below the resolution limit G is extrapolated with a
log-normal fitted to the resolved points (truncated fit).

Output: cache/sand_granulometry.npz with per-photo G at WIDTHS_MM and the fitted (mu, sigma).
usage: python src/sand_granulometry.py
"""
import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import least_squares
from scipy.stats import norm

from common import CACHE_DIR, TRAIN_DIR, camera_of, effective_ppm, list_images, test_name_to_id
from features import TARGET_PPM, load_scaled
from sam_grains import soil_mask

WIDTHS_MM = np.array([0.15, 0.2, 0.3, 0.45, 0.63, 1.0, 1.4, 2.0])
D_MAX_MM = 3.0
CROP_MM = 25.0
N_CROPS = 12


def native_gray(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    return np.asarray(im, np.float32)


def crop_positions(path, shape, ppm):
    """Crop corners (native px) fully inside the soil mask, spread over the image."""
    small = load_scaled(path)
    soil = soil_mask(small)
    s = ppm / TARGET_PPM
    c = int(CROP_MM * TARGET_PPM)
    ii = cv2.integral(soil.astype(np.uint8))
    H, W = soil.shape
    cands = [(y, x) for y in range(0, H - c, c // 2) for x in range(0, W - c, c // 2)
             if ii[y + c, x + c] - ii[y, x + c] - ii[y + c, x] + ii[y, x] == c * c]
    pick = [cands[i] for i in np.linspace(0, len(cands) - 1, min(N_CROPS, len(cands))).astype(int)]
    return [(int(y * s), int(x * s)) for y, x in pick if int(y * s) + int(CROP_MM * ppm) <= shape[0]
            and int(x * s) + int(CROP_MM * ppm) <= shape[1]]


def disk(d_px):
    d = max(1, int(round(d_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def granulometry(g, ppm):
    """Return (G_gray, G_binary): cumulative fraction of grain material with width < WIDTHS_MM."""
    flat = g / (cv2.GaussianBlur(g, (0, 0), 3 * ppm) + 1e-3)          # flatten illumination
    base = cv2.morphologyEx(flat, cv2.MORPH_OPEN, disk(D_MAX_MM * ppm))  # background / > 3 mm
    tot = (flat - base).clip(0).sum()
    gray = [(flat - cv2.morphologyEx(flat, cv2.MORPH_OPEN, disk(w * ppm))).clip(0).sum() / (tot + 1e-9)
            for w in WIDTHS_MM]
    u8 = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask = cv2.adaptiveThreshold(u8, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                 int(1.5 * ppm) | 1, -2)                # grains brighter than local mean
    big = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(D_MAX_MM * ppm)).sum()
    A = mask.sum() - big
    binary = [(mask.sum() - cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(w * ppm)).sum()) / (A + 1e-9)
              for w in WIDTHS_MM]
    return np.clip(gray, 0, 1), np.clip(binary, 0, 1)


def fit_lognormal(G):
    """Fit Phi((log10 d - mu)/sigma) to the resolved cumulative points; returns (mu, sigma)."""
    x = np.log10(WIDTHS_MM)
    r = least_squares(lambda p: norm.cdf((x - p[0]) / p[1]) - G, x0=[np.log10(0.3), 0.3],
                      bounds=([-2, 0.05], [1, 2]))
    return r.x


def main():
    tr, te = list_images()
    files = [f for f in tr if camera_of(f) == "Motorola_Edge_60_fusion"] + te  # only photos >= 12 px/mm
    rec = []
    for f in files:
        ppm = effective_ppm(f)
        g = native_gray(f)
        pos = crop_positions(f, g.shape, ppm)
        c = int(CROP_MM * ppm)
        Gs = np.array([granulometry(g[y:y + c, x:x + c], ppm) for y, x in pos])  # (n, 2, len(W))
        Gg, Gb = np.median(Gs[:, 0], 0), np.median(Gs[:, 1], 0)
        sid = test_name_to_id(f) if TRAIN_DIR not in f else f.split("_")[-2]
        rec.append(dict(path=f, sample=sid, ppm=ppm, ncrops=len(pos), G_gray=Gg, G_bin=Gb,
                        fit_gray=fit_lognormal(Gg), fit_bin=fit_lognormal(Gb)))
        fg, fb = rec[-1]["fit_gray"], rec[-1]["fit_bin"]
        print(f"{sid:28s} ppm={ppm:5.1f} crops={len(pos):2d}  G_gray={np.round(Gg, 2).tolist()}  "
              f"D50 gray={10 ** fg[0]:.2f}mm bin={10 ** fb[0]:.2f}mm  "
              f"<63um gray={norm.cdf((np.log10(0.063) - fg[0]) / fg[1]):.2f} bin={norm.cdf((np.log10(0.063) - fb[0]) / fb[1]):.2f}",
              flush=True)
    np.savez(f"{CACHE_DIR}/sand_granulometry.npz", widths=WIDTHS_MM,
             **{k: np.array([r[k] for r in rec]) for k in rec[0]})
    print("saved cache/sand_granulometry.npz")


if __name__ == "__main__":
    main()
